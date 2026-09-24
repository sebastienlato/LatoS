"""Frozen Phase 17 CUDA attempt; development failure never opens final inputs."""

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch

from latos.data.manifest import canonical_json
from latos.evaluation.access import evaluation_access
from latos.evaluation.compare import compare
from latos.evaluation.runner import run as evaluate_fixed
from latos.model import ModelConfig, create_model
from latos.model.storage import file_hash
from latos.training.checkpoint import load_checkpoint, save_checkpoint
from latos.training.config import TrainingConfig
from latos.training.engine import Trainer
from latos.training.full_data import FullDataset, build_cache, expected_accounting
from latos.training.probe import (
    TOKENIZER_SHA256,
    cuda_environment,
    host_peak_bytes,
    inference_measure,
    inventory,
    memory,
    record_failure,
    snapshot_source,
    verify_inputs,
)

PLAN_HASH = "0939804d662a65fcc915d6587fe3e4f994b3bdefdc988464d500128bde06f263"
MODEL_HASH = "bed0bbcc8b9b145485584ac9475d0c450c196649cb797a09d1c1c47fddc14c66"
LAYOUT_HASH = "c53a9a8d60ecc3f5d80d2776de0544dd54f4fd484ce82b42a888def9024ae171"
BASE_HASH = "f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    """Atomic status replacement; metrics and checkpoints remain append-only."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(value))
    temporary.replace(path)


def frozen(root):
    for name, expected in (
        ("configs/training2/phase17-plan.json", PLAN_HASH),
        ("configs/training2/selected-model.json", MODEL_HASH),
        ("experiments/phase-16/full-corpus-layout.json", LAYOUT_HASH),
    ):
        if file_hash(root / name) != expected:
            raise ValueError("Frozen Phase 17 contract changed")
    return read(root / "configs/training2/phase17-plan.json")


def training_config(plan):
    keys = set(TrainingConfig.__dataclass_fields__) - {"schema_version", "sequence_length"}
    return TrainingConfig(sequence_length=plan["context_length"], **{k: plan[k] for k in keys})


def tree_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def artifact_bytes(transfer, output):
    """Include transfer and validation artifacts, reserving another transfer copy.

    Runtime environments are reported separately. The duplicate transfer allowance
    conservatively covers the compressed handoff archive outside the extracted root.
    """
    manifest = read(transfer / "handoff.json")
    transferred = sum(record["bytes"] for record in manifest["files"].values())
    return 2 * transferred + tree_bytes(transfer / "validation") + tree_bytes(output)


def resource_guard(output, env):
    rss = host_peak_bytes()
    if rss is None or rss > 8 * 1024**3:
        raise ValueError("Host memory unavailable or exceeds 8 GiB")
    if torch.cuda.max_memory_reserved() > 0.85 * env["total_bytes"]:
        raise ValueError("CUDA reserved memory exceeds 85%")
    if tree_bytes(output) > 20 * 1024**3:
        raise ValueError("New attempt artifacts exceed 20 GiB")


def verify_transfer(root):
    """Bind actual source, models and input bytes to the reviewed local transfer."""
    manifest = read(root / "handoff.json")
    if manifest.get("phase") != 17 or manifest.get("execution_authorized") is not True:
        raise ValueError("Not an authorized Phase 17 transfer")
    for name, record in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Unsafe transfer path")
        path = root / relative
        if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
            raise ValueError(f"Reviewed transfer differs: {name}")
    # The worker must import precisely the packaged runtime, including installed-wheel use.
    package = Path(__file__).resolve().parents[1]
    actual = {p.relative_to(package).as_posix(): file_hash(p) for p in package.rglob("*.py")}
    expected = {
        name.removeprefix("source/src/latos/"): record["sha256"]
        for name, record in manifest["files"].items()
        if name.startswith("source/src/latos/") and name.endswith(".py")
    }
    if actual != expected:
        raise ValueError("Imported runtime differs from reviewed source")
    return manifest


def prepare_data(transfer, output, source):
    verify_inputs(transfer / "inputs")
    layout = read(source / "experiments/phase-16/full-corpus-layout.json")
    datasets, groups = [], []
    for split in ("train", "development"):
        destination = output / ("cache-" + split)
        expected = expected_accounting(layout, split)
        if destination.exists():
            dataset = FullDataset(destination, expected, TOKENIZER_SHA256, split)
        else:
            dataset = build_cache(
                transfer / "inputs/corpus",
                transfer / "inputs/tokenizer",
                split,
                destination,
                expected,
            )
        datasets.append(dataset)
        groups.append(set(read(destination / "cache.json")["groups"]))
    if groups[0] & groups[1]:
        raise ValueError("Cross-split groups")
    return datasets


def train_worker(args, env, source, plan):
    datasets = prepare_data(args.transfer, args.output, source)
    resource_guard(args.output, env)
    config = training_config(plan)
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    existing = sorted(checkpoints.glob("step-*"))
    if args.segment:
        if not existing:
            raise ValueError("No valid checkpoint exists for same-attempt recovery")
        trainer = load_checkpoint(
            existing[-1],
            *datasets,
            "cuda",
            expected_config=config,
            expected_precision="bfloat16",
            expected_single_pass=True,
        )
    else:
        if existing:
            raise ValueError("Fresh attempt cannot reuse any checkpoint")
        model = create_model(ModelConfig.load(source / plan["model_config"]), plan["seed"])
        trainer = Trainer(model, config, *datasets, "cuda", precision="bfloat16", single_pass=True)
    segment_dir = args.output / f"segment-{args.segment}"
    rollback_targets = 0
    if args.segment:
        previous_log = args.output / f"segment-{args.segment - 1}" / "updates.jsonl"
        if previous_log.exists():
            lines = previous_log.read_text().splitlines()
            for index, line in enumerate(lines):
                try:
                    old = json.loads(line)
                except json.JSONDecodeError:
                    if index != len(lines) - 1:
                        raise
                    continue  # A torn final record remains preserved, with unknown exposure.
                if old["step"] > trainer.step:
                    rollback_targets += old["targets"]
    write(
        segment_dir / "start.json",
        {
            "start_step": trainer.step,
            "tokens_seen": trainer.tokens_seen,
            "windows_seen": trainer.windows_seen,
            "datasets": trainer.data_identities,
            "config": config.to_dict(),
            "sampling_policy": "single permutation; no epoch wrap",
            "previous_segment_logged_targets_rolled_back": rollback_targets,
            "unlogged_partial_update_exposure": "unknown after interruption; not unique data"
            if args.segment
            else "none",
        },
    )
    with (segment_dir / "updates.jsonl").open("x", encoding="utf-8") as log:
        if not args.segment:
            initial = trainer.validate()
            save_checkpoint(trainer, checkpoints / "step-00000000")
            write(segment_dir / "initial.json", initial)
        while trainer.step < config.max_steps:
            resource_guard(args.output, env)
            metric = trainer.update()
            metric.update(memory())
            log.write(json.dumps(metric, allow_nan=False) + "\n")
            log.flush()
            resource_guard(args.output, env)
            if trainer.step in plan["checkpoint_steps"]:
                validation = trainer.validate()
                save_checkpoint(trainer, checkpoints / f"step-{trainer.step:08d}")
                write(
                    segment_dir / f"boundary-{trainer.step:08d}.json",
                    {
                        "validation": validation,
                        "resources": memory(),
                    },
                )
                resource_guard(args.output, env)
    if (
        trainer.tokens_seen != plan["data"]["targets_with_EOS"]
        or trainer.windows_seen != plan["data"]["windows"]
        or trainer.stream.epoch != 0
    ):
        raise ValueError("One-pass conservation failed")
    candidate = checkpoints / "step-00007485/model"
    write(
        args.output / "training-result.json",
        {
            "status": "complete",
            "candidate_weights": file_hash(candidate / "model.safetensors"),
            "step": trainer.step,
            "tokens_seen": trainer.tokens_seen,
            "windows_seen": trainer.windows_seen,
            "resources": memory(),
            "selection": "final planned update only; no checkpoint search",
        },
    )


def evaluation_worker(args, env, source, plan):
    candidate = args.output / "checkpoints/step-00007485/model"
    trained = read(args.output / "training-result.json")
    if trained["status"] != "complete" or trained["step"] != plan["max_steps"]:
        raise ValueError("Only a complete frozen run may be evaluated")
    phase = args.worker
    selection = args.output / "acceptance-plan.json" if phase == "final" else None
    if selection is not None:
        # A separate reviewed declaration is required; this runner never invents it.
        if not selection.is_file():
            raise ValueError("Final evaluation requires the reviewed acceptance declaration")
    for name, model, tokenizer, digest in (
        (
            "reference",
            args.transfer / "evaluation/base",
            args.transfer / "evaluation/tokenizer",
            BASE_HASH,
        ),
        ("candidate", candidate, args.transfer / "inputs/tokenizer", trained["candidate_weights"]),
    ):
        resource_guard(args.output, env)
        evaluate_fixed(
            SimpleNamespace(
                protocol=source / "configs/evaluation/protocol-v1.json",
                suite=source / "configs/evaluation/suite-v1.json",
                external_manifest=source / "configs/evaluation/external-v1.json",
                external_dir=args.transfer / "evaluation/external",
                data_manifest=source / "data/manifests/english-books-v1.json",
                corpus_dir=args.transfer / "evaluation/corpus",
                model_dir=model,
                tokenizer_dir=tokenizer,
                weights_sha256=digest,
                output=args.output / f"{phase}-{name}",
                device="cuda",
                threads=4,
                batch_size=8,
                acceptance_plan=selection,
            )
        )
        resource_guard(args.output, env)
        gc.collect()
        torch.cuda.empty_cache()
    report = compare(args.output / f"{phase}-reference", args.output / f"{phase}-candidate", "base")
    write(args.output / f"{phase}-comparison.json", report)
    if phase == "development":
        from latos.model.storage import load_model

        model = load_model(candidate).to("cuda")
        inference = inference_measure(model)
        resource_guard(args.output, env)
        write(
            args.output / "inference.json",
            {
                **inference,
                "resources": memory(),
                "passed": inference["median_first_token_seconds"] <= 1
                and inference["median_decode_tokens_per_second"] >= 10,
            },
        )
    write(
        args.output / f"{phase}-result.json",
        {
            "learned_gates_passed": report["passed"],
            "phase18_authorized": False,
            "next": "Mac review; final access only after development and resource gates pass"
            if phase == "development" and report["passed"]
            else "Mac review; retain all results",
        },
    )


def worker(args):
    with attempt_lock(args.output, "worker.lock"):
        worker_locked(args)


def worker_locked(args):
    source = args.transfer / "source"
    segment = (
        args.output / f"segment-{args.segment}"
        if args.worker == "train"
        else args.output / args.worker
    )
    segment.mkdir(exist_ok=False)
    try:
        snapshot_source(segment)
        manifest = verify_transfer(args.transfer)
        receipt = read(args.transfer / "preflight.json")
        if (
            receipt.get("passed") is not True
            or receipt.get("source_commit") != manifest["source_commit"]
            or receipt.get("handoff_sha256") != file_hash(args.transfer / "handoff.json")
        ):
            raise ValueError("Missing matching Windows checkout/wheel preflight")
        for kind in ("checkout", "wheel"):
            if file_hash(args.transfer / "validation" / f"{kind}.xml") != receipt[kind]["sha256"]:
                raise ValueError("Preflight test evidence changed or is missing")
        plan = frozen(source)
        # The 20 GiB free-space prerequisite applies before the initial attempt,
        # not again after its own retained checkpoints have consumed disk space.
        initial = args.worker == "train" and args.segment == 0
        env = cuda_environment(args.output, min_free_bytes=20 * 1024**3 if initial else 0)
        if (
            platform.python_version() != plan["runtime_policy"]["python"]
            or str(torch.__version__) != plan["runtime_policy"]["torch"]
        ):
            raise ValueError("Runtime differs from frozen CUDA environment")
        if env["driver_report"] is None or "616.92" not in env["driver_report"]:
            raise ValueError("Expected the measured driver 616.92")
        write(segment / "environment.json", env)
        torch.cuda.reset_peak_memory_stats()
        if args.worker == "train":
            train_worker(args, env, source, plan)
        else:
            evaluation_worker(args, env, source, plan)
    except Exception as exc:
        record_failure(segment, exc)
        raise
    finally:
        inventory(segment)


@contextmanager
def attempt_lock(output, name="active.lock"):
    """OS-held lock prevents concurrent recovery; process exit releases the lease."""
    with (output / name).open("a+b") as lease:
        if lease.tell() == 0:
            lease.write(b"0")
            lease.flush()
        lease.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def supervise(args):
    """One attempt ledger; hard subprocess time bounds survive external interruptions."""
    if args.stage == "train" and not args.resume_external_interruption:
        args.output.mkdir(parents=True, exist_ok=False)
        ledger = {"training_seconds": 0.0, "evaluation_seconds": 0.0, "segments": 0, "jobs": []}
        write(args.output / "ledger.json", ledger)
    else:
        ledger = read(args.output / "ledger.json")
    with attempt_lock(args.output):
        with attempt_lock(args.output, "worker.lock"):
            pass  # Refuse an orphaned worker before editing the attempt ledger.
        # Reload after obtaining the lock so a concurrent writer cannot race us.
        supervise_locked(args, read(args.output / "ledger.json"))


def supervise_locked(args, ledger):
    if args.resume_external_interruption:
        if args.stage != "train" or not ledger["jobs"] or ledger["segments"] >= 3:
            raise ValueError("Only two same-attempt external-interruption resumptions are allowed")
        last = ledger["jobs"][-1]
        previous = args.output / f"segment-{ledger['segments'] - 1}"
        if (
            last["stage"] != "train"
            or last.get("status") not in ("running", "external-interruption")
            or (previous / "failure.json").exists()
            or (args.output / "training-result.json").exists()
        ):
            raise ValueError("Numerical/resource/completed attempts cannot be retried")
        last["external_interruption_explanation"] = args.resume_external_interruption
        last["status"] = "external-interruption"
    elif args.stage != "train":
        if not (args.output / "training-result.json").exists() or not any(
            j["stage"] == "train" and j["status"] == "complete" for j in ledger["jobs"]
        ):
            raise ValueError("Training incomplete")
        if any(j["stage"] == args.stage for j in ledger["jobs"]):
            raise ValueError("Evaluation stage already attempted; preserve it")
        if args.stage == "final" and (
            read(args.output / "development-comparison.json").get("passed") is not True
            or read(args.output / "inference.json").get("passed") is not True
            or not any(
                j["stage"] == "development" and j["status"] == "complete" for j in ledger["jobs"]
            )
        ):
            raise ValueError("Final access blocked by development/resource gates")
        if args.stage == "final":
            source = args.transfer / "source"
            selection = args.output / "acceptance-plan.json"
            protocol = read(source / "configs/evaluation/protocol-v1.json")
            candidate = read(args.output / "training-result.json")["candidate_weights"]
            for digest in (BASE_HASH, candidate):
                evaluation_access(
                    protocol, source / "configs/evaluation/suite-v1.json", digest, selection
                )
    key = "training_seconds" if args.stage == "train" else "evaluation_seconds"
    limit = 7200 if args.stage == "train" else 3600
    remaining = min(
        limit - ledger[key], 10800 - ledger["training_seconds"] - ledger["evaluation_seconds"]
    )
    if remaining <= 0:
        raise ValueError("Frozen active budget exhausted")
    segment = ledger["segments"]
    if args.stage == "train":
        ledger["segments"] += 1
    job = {"stage": args.stage, "segment": segment, "status": "running", "seconds": 0.0}
    ledger["jobs"].append(job)
    write(args.output / "ledger.json", ledger)
    command = [
        sys.executable,
        "-m",
        "latos.training.base2",
        "--transfer",
        str(args.transfer),
        "--output",
        str(args.output),
        "--worker",
        args.stage,
        "--segment",
        str(segment),
    ]
    started, prior = time.monotonic(), ledger[key]
    with (args.output / f"{args.stage}-{segment}.log").open("x", encoding="utf-8") as log:
        try:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        except OSError as exc:
            job["status"] = "startup-failed"
            job["error"] = str(exc)
            write(args.output / "ledger.json", ledger)
            raise
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - started
                job["seconds"] = elapsed
                ledger[key] = prior + elapsed
                write(args.output / "ledger.json", ledger)
                if (
                    elapsed >= remaining
                    or artifact_bytes(args.transfer, args.output) > 20 * 1024**3
                ):
                    job["status"] = "budget-exceeded"
                    child.kill()
                    child.wait()
                    break
                time.sleep(0.5)
            if job["status"] == "running":
                job["status"] = "complete" if child.returncode == 0 else "failed"
            job["returncode"] = child.returncode
        except KeyboardInterrupt:
            child.terminate()
            child.wait()
            job["status"] = "external-interruption"
            raise
        except Exception as exc:
            if child.poll() is None:
                child.kill()
                child.wait()
            job["status"] = "supervisor-failure"
            job["error"] = str(exc)
            raise
        finally:
            job["seconds"] = time.monotonic() - started
            ledger[key] = prior + job["seconds"]
            if job["seconds"] > remaining and job["status"] == "complete":
                job["status"] = "budget-exceeded"
            write(args.output / "ledger.json", ledger)
    if job["status"] != "complete":
        raise ValueError(f"Stage stopped: {job['status']}; preserve evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "development", "final"), default="train")
    parser.add_argument("--resume-external-interruption")
    parser.add_argument("--worker", choices=("train", "development", "final"))
    parser.add_argument("--segment", type=int, default=0)
    args = parser.parse_args()
    args.transfer, args.output = args.transfer.resolve(), args.output.resolve()
    worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    main()
