"""Administrative recovery launcher. The original latos package is never changed.

Execute only this reviewed interruption, with the original Windows environment.
The failed ledger and all old files remain untouched. New status is append-only.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

SOURCE = "0fa94ad580110fd2dc7aaa2aa3560950abe73a02"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_once(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class Journal:
    """Keep one handle open; append and sync, never rename an active status file."""

    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf-8", newline="\n")

    def add(self, value):
        self.stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


@contextmanager
def lock(path):
    # Existing lock bytes must not change, even when acquiring the OS lease.
    with Path(path).open("r+b") as stream:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield stream
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def verify_records(root, records, held_files=None):
    for name, expected in records.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe reviewed path")
        path = root / relative
        if held_files and name in held_files:
            stream = held_files[name]
            stream.seek(0)
            actual_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        else:
            actual_digest = digest(path)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or actual_digest != expected["sha256"]
        ):
            raise ValueError(f"Preserved artifact mismatch: {name}")


def verify_bundle(bundle):
    manifest = read(bundle / "bundle.json")
    verify_records(bundle, manifest["files"])
    plan = read(bundle / "recovery-plan.json")
    if plan["training_source_commit"] != SOURCE or plan["recovery_segment"] != 1:
        raise ValueError("Wrong recovery plan")
    return plan


def verify_originals(transfer, attempt, plan, active_lease=None):
    if digest(transfer / "handoff.json") != plan["handoff_sha256"]:
        raise ValueError("Original handoff identity mismatch")
    if digest(transfer / "preflight.json") != plan["preflight_sha256"]:
        raise ValueError("Original preflight identity mismatch")
    verify_records(transfer, read(transfer / "handoff.json")["files"])
    verify_records(
        attempt,
        plan["original_attempt_files"],
        {"active.lock": active_lease} if active_lease is not None else None,
    )
    for name, expected in plan["frozen_files"].items():
        if digest(transfer / "source" / name) != expected:
            raise ValueError("Frozen experiment file changed")
    if read(attempt / "ledger.json.tmp")["jobs"][0]["status"] != "supervisor-failure":
        raise ValueError("Original failure must remain a supervisor failure")


def tail_metrics(attempt, plan):
    original = [
        json.loads(s) for s in (attempt / "segment-0/updates.jsonl").read_text().splitlines()
    ]
    replay = [json.loads(s) for s in (attempt / "segment-1/updates.jsonl").read_text().splitlines()]
    tail = original[plan["checkpoint_step"] :]
    if len(replay) != 485 or len(tail) != len(replay):
        raise ValueError("Recovery must replay exactly updates 7001 through 7485")
    for old, new in zip(tail, replay, strict=True):
        for key in ("step", "targets", "tokens_seen", "windows_seen", "epoch", "microbatches"):
            if old[key] != new[key]:
                raise ValueError(f"Recovered exposure/sampler mismatch: {key}")
        # Use the existing same-backend probe recovery loss tolerance, not a new
        # learned-quality threshold. No lost final tensor is claimed reconstructed bitwise.
        if not math.isfinite(new["loss"]) or abs(old["loss"] - new["loss"]) > 1e-5:
            raise ValueError("Replay differs beyond the existing recovery loss tolerance")
        if old["learning_rate"] != new["learning_rate"]:
            raise ValueError("Optimizer schedule changed")
        if not math.isfinite(new["grad_norm_before_clip"]):
            raise ValueError("Nonfinite replay gradient")
    if sum(row["targets"] for row in replay) != plan["replay_targets"]:
        raise ValueError("Replay accounting mismatch")
    start = read(attempt / "segment-1/start.json")
    if (
        start["start_step"] != 7000
        or start["tokens_seen"] != plan["retained_targets"]
        or start["windows_seen"] != 112000
        or start["previous_segment_logged_targets_rolled_back"] != plan["replay_targets"]
    ):
        raise ValueError("Recovery did not begin at the pinned complete checkpoint")
    return {
        "replayed_updates": len(replay),
        "replayed_targets": plan["replay_targets"],
        "logical_pass_targets": plan["final_targets"],
        "total_physical_target_executions": plan["final_targets"] + plan["replay_targets"],
        "max_replay_loss_delta": max(
            abs(a["loss"] - b["loss"]) for a, b in zip(tail, replay, strict=True)
        ),
    }


def verify_final(attempt, plan):
    replay = tail_metrics(attempt, plan)
    state_path = attempt / "checkpoints/step-00007485/state.json"
    state = read(state_path)
    result = read(attempt / "training-result.json")
    model_dir = state_path.parent / "model"
    metadata = read(model_dir / "metadata.json")
    if (
        state["step"] != 7485
        or state["runtime"] != plan["original_runtime"]
        or state["tokens_seen"] != plan["final_targets"]
        or state["windows_seen"] != plan["final_windows"]
        or state["epoch"] != 0
        or state["cursor"] != plan["final_windows"]
        or state["kind"] != "latos-training-single-pass-v3"
        or state["precision"] != "bfloat16"
        or result["status"] != "complete"
        or result["step"] != 7485
    ):
        raise ValueError("Recovered checkpoint/runtime identity mismatch")
    old = read(attempt / "checkpoints/step-00007000/state.json")
    if state["config"] != old["config"] or state["datasets"] != old["datasets"]:
        raise ValueError("Recovered configuration/data changed")
    if (
        digest(state_path.parent / "training.safetensors") != state["training_sha256"]
        or digest(model_dir / "metadata.json") != state["model_metadata_sha256"]
        or digest(model_dir / "config.json") != metadata["config_sha256"]
        or digest(model_dir / "model.safetensors") != metadata["weights_sha256"]
        or metadata["weights_sha256"] != result["candidate_weights"]
    ):
        raise ValueError("Recovered checkpoint checksum mismatch")
    # Original save_checkpoint already runs its strict CUDA load/readback before
    # exposing this directory. Rehashing here does not replace that validation.
    read(attempt / "segment-1/boundary-00007485.json")
    return {
        **replay,
        "final_weights_sha256": result["candidate_weights"],
        "same_source_runtime_config": True,
        "resumptions_used": 1,
        "phase17_complete": False,
        "fixed_quality_accepted": False,
    }


def size(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def artifact_bytes(transfer, attempt, sidecar, plan):
    return (
        2 * plan["original_transfer_bytes"]
        + size(transfer / "validation")
        + size(attempt)
        + size(sidecar)
        + plan["extra_artifact_reserve_bytes"]
    )


def run_child(command, cwd, log_path, journal, started, allowance, measure_bytes, cap):
    if time.monotonic() - started >= allowance:
        raise TimeoutError("Recovery preparation exhausted the remaining active budget")
    with log_path.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        try:
            journal.add(
                {
                    "event": "worker-start",
                    "pid": child.pid,
                    "command": command,
                    "seconds": time.monotonic() - started,
                }
            )
            while child.poll() is None:
                elapsed = time.monotonic() - started
                used = measure_bytes()
                journal.add({"event": "running", "seconds": elapsed, "artifact_bytes": used})
                if elapsed >= allowance or used > cap:
                    raise TimeoutError("Frozen time/artifact budget exhausted")
                time.sleep(0.5)
            if child.returncode != 0:
                raise RuntimeError(f"Original worker failed with exit code {child.returncode}")
            if time.monotonic() - started >= allowance or measure_bytes() > cap:
                raise TimeoutError("Frozen budget exhausted at worker completion")
            journal.add(
                {
                    "event": "worker-complete",
                    "returncode": child.returncode,
                    "seconds": time.monotonic() - started,
                }
            )
        except BaseException:
            if child.poll() is None:
                child.kill()
            child.wait()
            raise


def verified_receipt(sidecar, stage):
    if (sidecar / f"{stage}-failure.json").exists():
        raise ValueError("Prior stage has a failure record")
    result = read(sidecar / f"{stage}-result.json")
    events = [
        json.loads(line) for line in (sidecar / f"{stage}.events.jsonl").read_text().splitlines()
    ]
    if (
        not events
        or events[-1].get("event") != "complete"
        or events[-1].get("result") != result
        or result.get("status") != "complete"
    ):
        raise ValueError("Prior stage lacks a complete journal and receipt")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--stage", choices=("recover", "development"), required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if sys.platform != "win32":
        raise ValueError("Actual Windows RTX 4070 SUPER execution is required")
    bundle = Path(__file__).resolve().parent
    plan = verify_bundle(bundle)
    checks = read(bundle / "windows-checks.json")
    if checks.get("passed") is not True or checks.get("platform") != "win32":
        raise ValueError("Actual Windows administrative regression checks required")
    for name in ("recover.py", "test_recovery.py", "run_checks.py"):
        if checks["source_sha256"].get(name) != digest(bundle / name):
            raise ValueError("Administrative test source changed")
    transfer = args.transfer.resolve()
    source = transfer / "source"
    attempt = source / "outputs/phase17-cuda-attempt1"
    sidecar = transfer / "phase17-reviewed-recovery"
    # No other interpreter/environment or altered copy of the source is permitted.
    expected_python = source / ".venv/Scripts/python.exe"
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise ValueError("Use the original checkout interpreter, not a new environment")
    with lock(attempt / "active.lock") as active_lease:
        with lock(attempt / "worker.lock"):
            pass  # Refuse any live/orphaned old worker before inspecting state.
        if args.stage == "recover":
            sidecar.mkdir(exist_ok=False)  # One corrective launch; no automatic second replay.
            allowance = plan["training_limit_seconds"] - plan["prior_training_seconds_charged"]
        else:
            previous = verified_receipt(sidecar, "recover")
            if previous["status"] != "complete" or previous["resumptions_used"] != 1:
                raise ValueError("No complete reviewed recovery")
            allowance = plan["evaluation_limit_seconds"]
        journal = Journal(sidecar / f"{args.stage}.events.jsonl")
        try:
            journal.add(
                {
                    "event": "start",
                    "stage": args.stage,
                    "plan_sha256": digest(bundle / "recovery-plan.json"),
                    "prior_status_preserved": "supervisor-failure",
                    "allowance_seconds": allowance,
                    "prior_training_seconds_charged": plan["prior_training_seconds_charged"],
                }
            )
            verify_originals(transfer, attempt, plan, active_lease)
            if args.stage == "recover":
                if (
                    (attempt / "segment-1").exists()
                    or (attempt / "training-result.json").exists()
                    or sorted(p.name for p in (attempt / "checkpoints").iterdir())
                    != [f"step-{s:08d}" for s in range(0, 7001, 1000)]
                ):
                    raise ValueError("Unexpected continuation or checkpoint; stop for Mac review")
                worker = "train"
            else:
                verify_final(attempt, plan)
                if any(
                    (attempt / name).exists()
                    for name in (
                        "development",
                        "development-reference",
                        "development-candidate",
                        "final",
                    )
                ):
                    raise ValueError("Evaluation already attempted; no repeated selection")
                worker = "development"
            command = [
                str(expected_python),
                "-m",
                "latos.training.base2",
                "--transfer",
                str(transfer),
                "--output",
                str(attempt),
                "--worker",
                worker,
                "--segment",
                "1",
            ]
            journal.add({"event": "preflight-passed", "seconds": time.monotonic() - started})
            run_child(
                command,
                source,
                sidecar / f"{args.stage}.log",
                journal,
                started,
                allowance,
                lambda: artifact_bytes(transfer, attempt, sidecar, plan),
                plan["artifact_limit_bytes"],
            )
            verify_originals(transfer, attempt, plan, active_lease)
            if args.stage == "recover":
                details = verify_final(attempt, plan)
            else:
                report = read(attempt / "development-comparison.json")
                inference = read(attempt / "inference.json")
                details = {
                    "learned_gates_passed": report["passed"],
                    "resource_gates_passed": inference["passed"],
                    "phase17_complete": False,
                    "next": "Mac independent review; no automatic final scoring or Phase 18",
                }
            elapsed = time.monotonic() - started
            if elapsed >= allowance:
                raise TimeoutError("Post-run verification exhausted the remaining active budget")
            result = {
                "status": "complete",
                "stage": args.stage,
                "active_seconds": elapsed,
                **details,
                "training_seconds_charged": plan["prior_training_seconds_charged"] + elapsed
                if args.stage == "recover"
                else previous["training_seconds_charged"],
                "evaluation_seconds_charged": elapsed if args.stage == "development" else 0.0,
                "publication_authorized": False,
                "phase18_authorized": False,
            }
            write_once(sidecar / f"{args.stage}-result.json", result)
            journal.add({"event": "complete", "result": result})
        except BaseException as exc:
            # New records only. A second I/O failure is retained, not retried or reclassified.
            failure = {
                "status": "failed",
                "type": type(exc).__name__,
                "error": str(exc),
                "active_seconds": time.monotonic() - started,
                "stage": args.stage,
                "resumptions_used": 1 if args.stage == "recover" else None,
                "next": "Stop for Mac review; no automatic further execution",
            }
            try:
                write_once(sidecar / f"{args.stage}-failure.json", failure)
            finally:
                print(json.dumps(failure), file=sys.stderr, flush=True)
            raise
        finally:
            journal.close()


if __name__ == "__main__":
    main()
