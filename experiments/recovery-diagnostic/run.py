"""Two frozen CUDA mechanics pairs. No historical model or evaluation access."""

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_once(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def file_size(path):
    try:
        return path.stat().st_size
    except FileNotFoundError:
        # A completed checkpoint may be atomically renamed during a monitoring scan.
        # Its final path is counted on the next scan; other I/O failures still stop.
        return 0


def size(path):
    return sum(file_size(p) for p in Path(path).rglob("*") if p.is_file())


def artifact_bytes(output, bundle, plan):
    siblings = sum(file_size(p) for p in output.parent.glob(output.name + ".*") if p.is_file())
    return (
        size(output) + 2 * size(bundle) + siblings + plan["administrative_artifact_reserve_bytes"]
    )


def verify_bundle(bundle):
    manifest = read(bundle / "bundle.json")
    for name, record in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Unsafe bundle member")
        path = bundle / relative
        if path.stat().st_size != record["bytes"] or hash_file(path) != record["sha256"]:
            raise ValueError("Diagnostic bundle identity mismatch")
    plan = read(bundle / "plan.json")
    if (
        hash_file(bundle / "lengths.json") != plan["lengths_sha256"]
        or hash_file(bundle / "selected-model.json") != plan["model_config_sha256"]
    ):
        raise ValueError("Frozen model/shape inputs differ")
    if (
        plan["max_pairs"] != 2
        or plan["max_physical_updates"] != 2964
        or plan["max_seconds"] != 3600
        or plan["max_artifact_bytes"] != 6 * 1024**3
        or plan["loss_tolerance"] != 1e-5
    ):
        raise ValueError("Unexpected diagnostic bounds")
    return plan, manifest


def profile_environment(base, profile, plan):
    env = dict(base)
    env["PYTHONHASHSEED"] = "0"
    workspace = plan["profile_settings"][profile]["CUBLAS_WORKSPACE_CONFIG"]
    if workspace is None:
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        env["CUBLAS_WORKSPACE_CONFIG"] = workspace
    return env


def controls(profile, plan):
    import torch

    settings = plan["profile_settings"][profile]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != settings["CUBLAS_WORKSPACE_CONFIG"]:
        raise ValueError("Workspace policy was not set before process startup")
    for key in (
        "CUBLASLT_WORKSPACE_SIZE",
        "CUDNN_ERRATA_JSON_FILE",
        "NVIDIA_TF32_OVERRIDE",
        "TORCH_CUBLAS_WORKSPACE_CACHE",
    ):
        if os.environ.get(key):
            raise ValueError(f"Undeclared numerical override: {key}")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = settings["cudnn_deterministic"]
    torch.use_deterministic_algorithms(settings["deterministic_algorithms"], warn_only=False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(True)
    return {
        **settings,
        "cpu_threads": 4,
        "matmul_precision": "highest",
        "TF32": False,
        "cudnn_benchmark": False,
        "SDPA_enabled": {
            name: getattr(torch.backends.cuda, name + "_sdp_enabled")()
            for name in ("flash", "mem_efficient", "math", "cudnn")
        },
        "bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def worker(args):
    # These imports occur only after the parent established the per-process CUDA environment.
    import mechanics as m
    import numpy as np
    import safetensors
    import torch

    import latos
    from latos.model import ModelConfig, create_model
    from latos.training.checkpoint import implementation_hash, load_checkpoint, save_checkpoint
    from latos.training.engine import Trainer
    from latos.training.probe import cuda_environment, inventory, memory, snapshot_source

    plan, manifest = verify_bundle(args.bundle)
    if implementation_hash() != plan["legacy_implementation_sha256"]:
        raise ValueError("Original training package source differs")
    if platform.python_version() != plan["python"] or str(torch.__version__) != plan["torch"]:
        raise ValueError("Required original Windows runtime unavailable")
    if np.__version__ != plan["numpy"] or safetensors.__version__ != plan["safetensors"]:
        raise ValueError("Pinned input/serialization dependencies differ")
    model_config = ModelConfig.load(args.bundle / "selected-model.json")
    if args.worker == "prepare":
        args.output.mkdir(parents=True, exist_ok=False)
        meta = m.make_cache(args.bundle / "lengths.json", args.output / "data", model_config, plan)
        if meta["files"] != plan["expected_synthetic_files"]:
            raise ValueError("Synthetic bytes differ from the frozen Mac construction")
        write_once(args.output / "synthetic-data.json", meta)
        return
    stage = args.output / args.profile / args.worker
    stage.mkdir(parents=True, exist_ok=False)
    try:
        snapshot_source(stage)
        applied = json.loads(json.dumps(controls(args.profile, plan), allow_nan=False))
        env = cuda_environment(
            args.output,
            min_free_bytes=6 * 1024**3
            if args.profile == "legacy" and args.worker == "reference"
            else 0,
        )
        if env["driver_report"] is None or plan["driver"] not in env["driver_report"]:
            raise ValueError("Required measured driver unavailable")
        # cuda_environment retains the declared deterministic controls; assert, do not assume.
        if torch.are_deterministic_algorithms_enabled() != applied["deterministic_algorithms"]:
            raise ValueError("Deterministic policy changed")
        random.seed(plan["model_seed"])
        np.random.seed(plan["model_seed"])
        torch.manual_seed(plan["model_seed"])
        torch.cuda.manual_seed_all(plan["model_seed"])
        if read(args.output / "data/dataset.json")["files"] != plan["expected_synthetic_files"]:
            raise ValueError("Synthetic cache differs from frozen byte identities")
        desc = m.descriptor(args.bundle / "lengths.json", model_config, plan)
        train = m.SyntheticDataset(args.output / "data", desc)
        dummy = m.SyntheticDataset(args.output / "data", desc, dummy=True)
        if (
            len(train.windows) != plan["expected_windows"]
            or train.identity["targets"] != plan["historical_shapes_targets"]
        ):
            raise ValueError("Synthetic exposure differs from the frozen shape schedule")
        config = m.training_config(plan)
        torch.cuda.reset_peak_memory_stats()
        if args.worker == "resumed":
            saved = args.output / args.profile / "reference/split"
            expected = read(args.output / args.profile / "reference/boundary.json")
            if expected["controls"] != applied:
                raise ValueError("Checkpoint numerical policy differs")
            trainer = load_checkpoint(
                saved,
                train,
                dummy,
                "cuda",
                expected_config=config,
                expected_precision="bfloat16",
                expected_single_pass=True,
            )
        else:
            trainer = Trainer(
                create_model(model_config, plan["model_seed"]),
                config,
                train,
                dummy,
                "cuda",
                precision="bfloat16",
                single_pass=True,
            )
        if trainer.model.parameter_count != plan["model_parameters"]:
            raise ValueError("Selected model parameter count differs")
        initial = m.snapshot(trainer)
        if args.worker == "resumed" and initial != expected["before"]:
            raise ValueError("New-process restore changed tensor/optimizer/sampler/RNG state")
        if args.worker == "reference" and args.profile == "deterministic":
            control_initial = read(args.output / "legacy/reference/start.json")["state"]
            if initial != control_initial:
                raise ValueError("Profiles do not share identical fresh initialization")
        write_once(
            stage / "start.json",
            {
                "controls": applied,
                "environment": env,
                "bundle_sha256": hash_file(args.bundle / "bundle.json"),
                "state": initial,
                "source_commit": manifest["source_commit"],
                "imported_package": str(Path(latos.__file__).resolve()),
                "initialization": {
                    "model_seed": plan["model_seed"],
                    "shuffle_seed": plan["shuffle_seed"],
                    "synthetic_seed": plan["synthetic_seed"],
                    "source": "fresh random"
                    if args.worker == "reference"
                    else "this profile's synthetic split",
                },
                "scope": "Synthetic mechanics; no learned model or held-out scoring",
            },
        )
        stop = plan["steps"]
        started = time.monotonic()
        physical = 0

        def guard():
            resources = memory()
            if (
                resources["host_peak_working_set_bytes"] is None
                or resources["host_peak_working_set_bytes"] > plan["max_host_bytes"]
                or resources["peak_reserved_bytes"]
                > plan["max_reserved_VRAM_fraction"] * env["total_bytes"]
            ):
                raise ValueError("Diagnostic memory bound exceeded")
            return resources

        def checkpoint(name):
            if (
                artifact_bytes(args.output, args.bundle, plan) + 512 * 1024**2
                > plan["max_artifact_bytes"]
            ):
                raise ValueError("Insufficient artifact budget for another checkpoint")
            before = m.snapshot(trainer)
            save_checkpoint(trainer, stage / name)
            after = m.snapshot(trainer)
            restored = load_checkpoint(
                stage / name,
                train,
                dummy,
                "cuda",
                expected_config=config,
                expected_precision="bfloat16",
                expected_single_pass=True,
            )
            loaded = m.snapshot(restored)
            if before != after or before != loaded:
                raise ValueError("Checkpoint readback changed full state")
            del restored
            gc.collect()
            # Preserve the uninterrupted worker's warm allocator/cache history.
            record = {
                "controls": applied,
                "before": before,
                "after": after,
                "loaded": loaded,
                "roundtrip_exact": True,
                "resources": guard(),
            }
            write_once(stage / ("boundary.json" if name == "split" else "final.json"), record)

        guard()
        with (stage / "trace.jsonl").open("x", encoding="utf-8") as stream:
            while trainer.step < stop:
                inputs = m.input_signature(trainer)
                metric = trainer.update()
                physical += 1
                state = m.snapshot(trainer)
                if state["global_rng"] != initial["global_rng"]:
                    raise ValueError("Unexpected global RNG consumption")
                record = {"metric": metric, "input": inputs, "state": state, "resources": guard()}
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                if trainer.step == plan["checkpoint_step"]:
                    checkpoint("split")
            checkpoint("final-checkpoint")
        if m.SyntheticDataset(args.output / "data", desc).identity != train.identity:
            raise ValueError("Synthetic input identity changed during execution")
        write_once(
            stage / "result.json",
            {
                "status": "complete",
                "executed_updates": physical,
                "final_step": trainer.step,
                "targets_seen": trainer.tokens_seen,
                "active_seconds": time.monotonic() - started,
                "resources": guard(),
                "diagnostic_only": True,
                "phase17_1_authorized": False,
            },
        )
    except BaseException as exc:
        write_once(stage / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        inventory(stage)


def job_specs(plan):
    jobs = [("prepare", "legacy", 0)]
    for profile in plan["profiles"]:
        jobs.extend(
            [
                ("reference", profile, plan["steps"]),
                ("resumed", profile, plan["replay_updates"]),
            ]
        )
    if sum(j[2] for j in jobs) != plan["max_physical_updates"] or len(plan["profiles"]) != 2:
        raise ValueError("Physical update accounting differs from frozen plan")
    return jobs


def read_trace(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def compare_all(output, plan):
    import mechanics as m

    pairs, initial_states = {}, []
    for profile in plan["profiles"]:
        root = output / profile
        reference = read_trace(root / "reference/trace.jsonl")
        resumed = read_trace(root / "resumed/trace.jsonl")
        initial_states.append(read(root / "reference/start.json")["state"]["state_sha256"])
        boundary = read(root / "reference/boundary.json")
        restored = read(root / "resumed/start.json")
        if (
            not boundary["roundtrip_exact"]
            or not boundary["before"]
            == boundary["after"]
            == boundary["loaded"]
            == restored["state"]
        ):
            raise ValueError("Missing exact roundtrip/new-process restoration of shared prefix")
        for mode in ("reference", "resumed"):
            if read(root / mode / "start.json")["controls"] != boundary["controls"]:
                raise ValueError("Within-pair numerical policies differ")
        pairs[profile] = m.compare_pair(
            reference, resumed, plan["checkpoint_step"], plan["loss_tolerance"]
        )
    if len(set(initial_states)) != 1:
        raise ValueError("Profiles did not start from the identical fresh random state")
    legacy, fixed = pairs["legacy"], pairs["deterministic"]
    reproduced = not legacy["loss_gate_passed"]
    corrected = fixed["loss_gate_passed"] and fixed["replay_states_identical"]
    return {
        "pairs": pairs,
        "legacy_loss_failure_reproduced": reproduced,
        "lossless_serialization_verified_within_each_trajectory": True,
        "prospective_correction_demonstrated": reproduced and corrected,
        "deterministic_pair_reproduced_exact_state": corrected,
        "interpretation": "Execution-policy sensitivity demonstrated in synthetic shared-state "
        "continuations; exact historical kernel cause not identified"
        if reproduced and corrected
        else "Correction not established under the frozen diagnostic decision rule; stop",
        "phase17_historical_failure_unchanged": True,
        "phase17_1_training_authorized": False,
        "phase18_authorized": False,
        "publication_authorized": False,
    }


def bounded_child(command, environment, log, event, started, limit, measure, cap):
    if time.monotonic() - started >= limit or measure() >= cap:
        raise TimeoutError("Budget exhausted before worker dispatch")
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    try:
        while child.poll() is None:
            elapsed, used = time.monotonic() - started, measure()
            event({"event": "running", "seconds": elapsed, "artifact_bytes": used})
            if elapsed >= limit or used >= cap:
                raise TimeoutError("Frozen time/artifact bound reached")
            time.sleep(0.5)
        if child.returncode != 0:
            raise RuntimeError("Diagnostic worker failed; no retry")
        if time.monotonic() - started >= limit or measure() >= cap:
            raise TimeoutError("Budget reached at worker completion")
    except BaseException:
        if child.poll() is None:
            child.kill()
        child.wait()
        raise


def supervise(args):
    plan, manifest = verify_bundle(args.bundle)
    if sys.platform != "win32":
        raise ValueError("Selected-scale diagnostics require the actual Windows RTX 4070 SUPER")
    preflight = read(args.bundle / "windows-preflight.json")
    if (
        preflight.get("passed") is not True
        or preflight.get("bundle_sha256") != hash_file(args.bundle / "bundle.json")
        or preflight.get("optimizer_updates") != 0
        or preflight.get("platform") != "win32"
    ):
        raise ValueError("Matching Windows zero-update preflight required")
    if hash_file(args.bundle / "windows-preflight.xml") != preflight["junit_sha256"]:
        raise ValueError("Preflight JUnit identity changed")
    if args.output.exists() or args.output.is_relative_to(args.bundle):
        raise ValueError("Use one new output root outside the bundle; no retry")
    # Preparation worker owns output creation. A separate exclusive lease reserves the one launch.
    lease = args.output.with_name(args.output.name + ".launch.json")
    lease.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        lease,
        {
            "source_commit": manifest["source_commit"],
            "reserved_updates": plan["max_physical_updates"],
        },
    )
    started = time.monotonic() - preflight["seconds"]
    jobs = []
    journal_path = args.output.with_name(args.output.name + ".events.jsonl")
    with journal_path.open("x", encoding="utf-8") as journal:

        def event(value):
            journal.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())

        event({"event": "start", "plan": plan, "pairs": 2, "no_retries": True})
        try:
            for mode, profile, reserved in job_specs(plan):
                command = [
                    sys.executable,
                    str(args.bundle / "run.py"),
                    "--bundle",
                    str(args.bundle),
                    "--output",
                    str(args.output),
                    "--worker",
                    mode,
                    "--profile",
                    profile,
                ]
                job = {
                    "mode": mode,
                    "profile": profile,
                    "reserved_updates": reserved,
                    "status": "running",
                }
                jobs.append(job)
                event({"event": "job-start", **job, "seconds": time.monotonic() - started})
                log_path = args.output.with_name(args.output.name + f".{profile}-{mode}.log")
                with log_path.open("x", encoding="utf-8") as log:
                    bounded_child(
                        command,
                        profile_environment(os.environ, profile, plan),
                        log,
                        event,
                        started,
                        plan["max_seconds"],
                        lambda: artifact_bytes(args.output, args.bundle, plan),
                        plan["max_artifact_bytes"],
                    )
                job["status"] = "complete"
                if mode != "prepare":
                    result = read(args.output / profile / mode / "result.json")
                    if result["executed_updates"] != reserved:
                        raise ValueError("Worker update budget mismatch")
                event({"event": "job-complete", **job, "seconds": time.monotonic() - started})
            result = compare_all(args.output, plan)
            elapsed = time.monotonic() - started
            if (
                elapsed >= plan["max_seconds"]
                or artifact_bytes(args.output, args.bundle, plan) >= plan["max_artifact_bytes"]
            ):
                raise TimeoutError("Budget exhausted during comparison")
            result.update(
                active_seconds=elapsed,
                physical_updates=sum(j["reserved_updates"] for j in jobs),
                jobs=jobs,
            )
            write_once(args.output / "summary.json", result)
            event({"event": "complete", "summary": result})
        except BaseException as exc:
            failure = {
                "status": "incomplete_or_failed",
                "type": type(exc).__name__,
                "error": str(exc),
                "seconds": time.monotonic() - started,
                "jobs": jobs,
                "next": "Preserve evidence and stop; no Phase 17.1 or automatic retry",
            }
            write_once(args.output.with_name(args.output.name + ".failure.json"), failure)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("prepare", "reference", "resumed"))
    parser.add_argument("--profile", choices=("legacy", "deterministic"), default="legacy")
    args = parser.parse_args()
    args.bundle, args.output = args.bundle.resolve(), args.output.resolve()
    worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    main()
