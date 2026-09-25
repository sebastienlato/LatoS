"""Ticketed child jobs; only run.py may dispatch these bounded Windows jobs."""

import argparse
import gc
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

from common import (
    ATTEMPT,
    artifact_bytes,
    canonical,
    equal_replay,
    hash_file,
    lock,
    read,
    require,
    safe_path,
    trace,
    verify_bundle,
    verify_transfer,
    write_once,
)


def schedule(config, path, fresh):
    value = {
        "config": config.to_dict(),
        "rates": [config.learning_rate_at(i) for i in range(1, config.max_steps + 1)],
    }
    if fresh:
        write_once(path, value)
    require(read(path) == value, "Same-Windows schedule table changed")
    return value["rates"]


def execute(request, bundle, output, transfer, jobdir):
    from runtime import (
        check_controls,
        fingerprints,
        fixture,
        load_bound,
        save_bound,
        seed_all,
        setup,
    )

    plan, manifest = verify_bundle(bundle)
    original = verify_transfer(transfer, bundle, plan)
    if request["kind"] == "integrity":
        expected = {
            name.removeprefix("source/src/latos/"): value
            for name, value in original["files"].items()
            if name.startswith("source/src/latos/") and name.endswith(".py")
        }
        from common import record

        wheel = transfer / "source/.wheel-env/Lib/site-packages/latos"
        actual = {p.relative_to(wheel).as_posix(): record(p) for p in wheel.rglob("*.py")}
        require(actual == expected, "Original installed-wheel source differs")
        return {
            "status": "complete",
            "kind": "integrity",
            "optimizer_updates": 0,
            "original_manifest_sha256": hash_file(transfer / "handoff.json"),
            "verified_original_files": len(original["files"]),
            "wheel_python_files": len(actual),
            "reserved_handling": "Opaque hash only, no parsing/scoring",
        }
    controls = setup(plan, output)
    import torch

    import latos
    from latos.model import ModelConfig, create_model
    from latos.training.base2 import evaluation_worker, prepare_data, training_config
    from latos.training.engine import Trainer
    from latos.training.probe import memory, snapshot_source

    actual = Path(latos.__file__).resolve()
    if request["runtime"] == "checkout":
        require(
            actual.is_relative_to(transfer / "source/src/latos"), "Not original checkout import"
        )
    else:
        require(
            actual.is_relative_to(transfer / "source/.wheel-env/Lib/site-packages/latos"),
            "Not original isolated wheel import",
        )
    snapshot_source(jobdir)
    m = fingerprints()
    write_once(
        jobdir / "environment.json",
        {
            **controls,
            "imported_package": str(actual),
            "controller_source_commit": manifest["source_commit"],
            "request": request,
        },
    )
    torch.cuda.reset_peak_memory_stats()

    def guard(reserve=0):
        check_controls(plan)
        current = memory()
        require(
            current["host_peak_working_set_bytes"] is not None
            and current["host_peak_working_set_bytes"]
            <= plan["budgets"]["max_process_host_RSS_bytes"]
            and current["peak_reserved_bytes"] <= controls["gpu_total_bytes"] * 0.85,
            "Host/GPU memory bound exceeded",
        )
        require(
            artifact_bytes(output, bundle) + reserve < plan["budgets"]["new_artifact_stop_bytes"],
            "Artifact space bound exceeded",
        )
        return current

    guard()
    kind = request["kind"]
    if kind in ("development", "final"):
        from runtime import verify_policy

        trained = read(output / "training-result.json")
        binding = read(output / "training-binding.json")
        require(
            binding["controls"] == controls
            and binding["bundle_sha256"] == hash_file(bundle / "bundle.json"),
            "Evaluation runtime/source differs from training",
        )
        candidate = output / "checkpoints/step-00007485"
        verify_policy(candidate, binding)
        require(
            trained["step"] == 7485
            and trained["tokens_seen"] == 37811418
            and trained["windows_seen"] == 119748
            and trained["candidate_weights"] == hash_file(candidate / "model/model.safetensors"),
            "Candidate identity/exposure differs",
        )
        evaluation_worker(
            SimpleNamespace(output=output, transfer=transfer, worker=kind),
            {"total_bytes": controls["gpu_total_bytes"]},
            transfer / "source",
            plan["training"],
        )
        return {"kind": kind, "optimizer_updates": 0, "resources": guard(), "status": "complete"}

    is_fixture = kind.startswith("fixture-")
    fresh = kind in ("fixture-reference", "train")
    if is_fixture:
        model_cfg, config, datasets, seed, fixture_sha = fixture(plan, request["fixture"])
        require(
            fixture_sha == read(bundle / "fixture-identities.json")[request["fixture"]],
            "Prespecified fixture bytes differ",
        )
        reference = output / "jobs" / request["reference_id"]
        schedule_path = reference / "schedule.json"
        checkpoints = jobdir / "checkpoints"
        checkpoint_steps = [3, 8]
        resume_path = reference / "checkpoints/step-00000003" if not fresh else None
    else:
        require(kind in ("train", "resume"), "Unknown optimizer job")
        datasets = prepare_data(transfer, output, transfer / "source")
        config = training_config(plan["training"])
        model_cfg = ModelConfig.load(transfer / "source/configs/training2/selected-model.json")
        seed, fixture_sha = 160, None
        schedule_path = output / "schedule.json"
        checkpoints = output / "checkpoints"
        checkpoint_steps = plan["training"]["checkpoint_steps"]
        resume_path = safe_path(output, request["resume_checkpoint"]) if not fresh else None
    rates = schedule(config, schedule_path, fresh)
    binding = {
        "attempt_id": ATTEMPT,
        "plan_sha256": hash_file(bundle / "plan.json"),
        "bundle_sha256": hash_file(bundle / "bundle.json"),
        "controls": controls,
        "schedule_sha256": hash_file(schedule_path),
        "model_config": model_cfg.to_dict(),
        "training_config": config.to_dict(),
        "datasets": {"train": datasets[0].identity, "validation": datasets[1].identity},
        "fixture_sha256": fixture_sha,
        "initial_seed": seed,
    }
    if not is_fixture:
        if fresh:
            write_once(output / "training-binding.json", binding)
        require(read(output / "training-binding.json") == binding, "Training binding changed")
    seed_all(seed)
    if fresh:
        trainer = Trainer(
            create_model(model_cfg, seed),
            config,
            *datasets,
            "cuda",
            precision="bfloat16",
            single_pass=True,
        )
        require(trainer.step == 0 and not trainer.optimizer.state, "Initialization is not fresh")
        if not is_fixture:
            require(trainer.model.parameter_count == 34087424, "Selected model scale changed")
    else:
        trainer = load_bound(resume_path, datasets, config, binding, m)
    initial = m.snapshot(trainer)
    write_once(
        jobdir / "start.json",
        {
            "state": initial,
            "binding": binding,
            "fresh_random": fresh,
            "resume_checkpoint": str(resume_path),
        },
    )
    overlap = {}
    if not fresh:
        if is_fixture:
            paths = [reference / "updates.jsonl"]
        else:
            paths = [safe_path(output, name) for name in request["overlap_traces"]]
        for path in paths:
            for old in trace(path, allow_torn=not is_fixture):
                step = old["metric"]["step"]
                if step > trainer.step:
                    if step in overlap:
                        equal_replay(overlap[step], old)
                    overlap[step] = old
        require(
            not overlap or sorted(overlap) == list(range(trainer.step + 1, max(overlap) + 1)),
            "Noncontiguous recorded overlap",
        )
    checkpoints.mkdir(exist_ok=True)
    if not is_fixture and fresh:
        write_once(jobdir / "initial-monitoring.json", trainer.validate())
        require(m.snapshot(trainer) == initial, "Initial monitoring changed training state")
        guard(512 * 1024**2)
        write_once(
            jobdir / "checkpoint-0.json",
            save_bound(trainer, checkpoints / "step-00000000", binding, m),
        )
    executed, repeated_targets = 0, 0
    with (jobdir / "updates.jsonl").open("xb") as log:
        while trainer.step < config.max_steps:
            guard()
            inputs = m.input_signature(trainer)
            metric = trainer.update()
            executed += 1
            require(executed <= request["reserved_updates"], "Job update budget exceeded")
            state = m.snapshot(trainer)
            require(
                state["global_rng"] == initial["global_rng"], "Unexpected global RNG consumption"
            )
            require(metric["learning_rate"] == rates[trainer.step - 1], "Schedule table mismatch")
            row = {"metric": metric, "input": inputs, "state": state, "resources": guard()}
            # Persist the actual mismatching update before failing; it remains evidence.
            log.write(canonical(row))
            log.flush()
            os.fsync(log.fileno())
            if trainer.step in overlap:
                equal_replay(overlap[trainer.step], row)
                repeated_targets += metric["targets"]
            if trainer.step in checkpoint_steps:
                if not is_fixture:
                    before = m.snapshot(trainer)
                    write_once(jobdir / f"monitoring-{trainer.step}.json", trainer.validate())
                    require(m.snapshot(trainer) == before, "Monitoring changed training state")
                guard(512 * 1024**2 if not is_fixture else 1024**2)
                saved = save_bound(trainer, checkpoints / f"step-{trainer.step:08d}", binding, m)
                write_once(jobdir / f"checkpoint-{trainer.step}.json", saved)
                guard()
    require(executed == request["reserved_updates"], "Job did not complete reserved trajectory")
    result = {
        "status": "complete",
        "kind": kind,
        "optimizer_updates": executed,
        "step": trainer.step,
        "tokens_seen": trainer.tokens_seen,
        "windows_seen": trainer.windows_seen,
        "recorded_overlap_updates": len(overlap),
        "replayed_logged_targets": repeated_targets,
        "resources": guard(),
    }
    if not is_fixture:
        require(
            trainer.tokens_seen == 37811418
            and trainer.windows_seen == 119748
            and trainer.stream.epoch == 0,
            "Full-data exposure differs",
        )
        result.update(
            candidate_weights=hash_file(checkpoints / "step-00007485/model/model.safetensors"),
            selection="new final planned update only",
            phase18_authorized=False,
        )
        write_once(output / "training-result.json", result)
    del trainer
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    token = os.environ.pop("LATOS_PHASE171_TICKET", "")
    expected_request = os.environ.pop("LATOS_PHASE171_REQUEST_SHA256", "")
    require(
        token and expected_request and hash_file(args.request) == expected_request,
        "Internal worker needs an unchanged request and live supervisor ticket",
    )
    request = read(args.request)
    require(
        token and hashlib.sha256(token.encode()).hexdigest() == request["ticket_sha256"],
        "Internal worker needs a live supervisor ticket",
    )
    bundle, output, transfer = (
        Path(request[k]).resolve() for k in ("bundle", "output", "transfer")
    )
    require(os.name == "nt", "No CUDA attempt on Mac or another platform")
    require(output == transfer / "phase17-1-results", "Attempt output root is fixed")
    jobdir = safe_path(output / "jobs", request["id"])
    with lock(output / "worker.lock"):
        jobdir.mkdir(exist_ok=False)
        try:
            result = execute(request, bundle, output, transfer, jobdir)
            write_once(jobdir / "result.json", result)
        except Exception as exc:
            write_once(jobdir / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
            raise
        finally:
            from common import record

            write_once(
                jobdir / "inventory.json",
                {
                    p.relative_to(jobdir).as_posix(): record(p)
                    for p in jobdir.rglob("*")
                    if p.is_file()
                },
            )


if __name__ == "__main__":
    main()
