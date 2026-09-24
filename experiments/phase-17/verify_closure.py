"""Read-only reconstruction of the failed corrective gate; never train or score."""

import argparse
import json
import math
import subprocess
import zipfile
from pathlib import Path

from latos.model.storage import file_hash, load_model

TRAINING_COMMIT = "0fa94ad580110fd2dc7aaa2aa3560950abe73a02"
ADMIN_COMMIT = "1436ec27a9ef42c6a72c7625ba9b4414cdf82daf"
ARCHIVE_SHA256 = "a0d521a0c9c4e7bf12d25b8bca19cc389b4ddd77928d73b3f2221cb4c0b0e0f9"
TOLERANCE = 1e-5


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def replay_statistics(original, recovered):
    if len(original) != 7485 or len(recovered) != 485:
        raise ValueError("Wrong original/replay update count")
    differences = []
    for expected_step, old, new in zip(range(7001, 7486), original[7000:], recovered, strict=True):
        if old["step"] != expected_step or new["step"] != expected_step:
            raise ValueError("Noncontiguous replay")
        for key in (
            "targets",
            "tokens_seen",
            "windows_seen",
            "microbatches",
            "epoch",
            "learning_rate",
        ):
            if old[key] != new[key]:
                raise ValueError("Replay counter or schedule mismatch")
        if not all(math.isfinite(row["loss"]) for row in (old, new)):
            raise ValueError("Nonfinite logged loss")
        differences.append(
            {
                "step": expected_step,
                "original_loss": old["loss"],
                "replay_loss": new["loss"],
                "absolute_delta": abs(old["loss"] - new["loss"]),
            }
        )
    failed = [row for row in differences if row["absolute_delta"] > TOLERANCE]
    return {
        "tolerance": TOLERANCE,
        "passed": not failed,
        "updates": len(recovered),
        "targets": sum(r["targets"] for r in recovered),
        "counter_schedule_mismatches": [],
        "updates_over_tolerance": len(failed),
        "first_over_tolerance": failed[0] if failed else None,
        "maximum_delta": max(differences, key=lambda r: r["absolute_delta"]),
        "logical_targets": recovered[-1]["tokens_seen"],
        "physical_targets": sum(r["targets"] for r in original + recovered),
        "final_microbatches": recovered[-1]["microbatches"],
    }


def verify(root, returned, archive, first_return, original_verification):
    if file_hash(archive) != ARCHIVE_SHA256:
        raise ValueError("Owner-specified return digest differs")
    with zipfile.ZipFile(archive) as packed:
        if packed.read("return.json") != (returned / "return.json").read_bytes():
            raise ValueError("Extracted manifest is not bound to the verified archive")
    manifest = read(returned / "return.json")
    actual_files = {p.relative_to(returned).as_posix() for p in returned.rglob("*") if p.is_file()}
    if actual_files != set(manifest["files"]) | {"return.json"}:
        raise ValueError("Extra or missing extracted evidence")
    all_records = {**manifest["files"], **manifest["omitted_retained_on_Windows"]}
    if set(manifest["files"]) & set(manifest["omitted_retained_on_Windows"]):
        raise ValueError("Ambiguous return inventory")
    for name, identity in manifest["files"].items():
        path = returned / name
        if path.stat().st_size != identity["bytes"] or file_hash(path) != identity["sha256"]:
            raise ValueError("Extracted evidence differs from inventory")
    first = read(first_return / "return.json")
    for name, identity in {**first["files"], **first["omitted_retained_on_Windows"]}.items():
        if all_records.get(name) != identity:
            raise ValueError(f"Original evidence was changed or omitted: {name}")
    administrative_archive = root / "outputs/phase17-recovery-handoff.zip"
    if (
        file_hash(administrative_archive)
        != "49bd4f5048365e67dc0ac5f8c0597c1970c7a5032c2c4a144333334cf77c978d"
    ):
        raise ValueError("Original administrative handoff digest changed")
    with zipfile.ZipFile(administrative_archive) as handoff:
        bundle = json.loads(handoff.read("bundle.json"))
        if bundle["administrative_commit"] != ADMIN_COMMIT:
            raise ValueError("Wrong administrative source")
        if handoff.read("bundle.json") != (returned / "recovery-bundle/bundle.json").read_bytes():
            raise ValueError("Recovery bundle changed")
        for name in bundle["files"]:
            source = (
                "tests/test_phase17_recovery.py"
                if name == "test_recovery.py"
                else "experiments/phase-17/recovery/" + name
            )
            authoritative = subprocess.check_output(
                ["git", "show", f"{ADMIN_COMMIT}:{source}"], cwd=root
            )
            if handoff.read(name) != authoritative:
                raise ValueError("Administrative handoff differs from reviewed Git source")
            if handoff.read(name) != (returned / "recovery-bundle" / name).read_bytes():
                raise ValueError("Executed administrative script/plan differs")
    checks = read(returned / "recovery-bundle/windows-checks.json")
    if checks["tests_passed"] != 9 or checks["skipped"] != 0 or checks["passed"] is not True:
        raise ValueError("Administrative Windows controls failed")
    for name, expected in checks["source_sha256"].items():
        if file_hash(returned / "recovery-bundle" / name) != expected:
            raise ValueError("Administrative test binding mismatch")
    log = (returned / "recovery-bundle/windows-checks.log").read_text()
    if log.count(" ... ok") != 9 or "Ran 9 tests" not in log or "\nOK\n" not in log:
        raise ValueError("Test log and receipt disagree")
    summary = replay_statistics(
        lines(returned / "attempt/segment-0/updates.jsonl"),
        lines(returned / "attempt/segment-1/updates.jsonl"),
    )
    if summary["passed"]:
        raise ValueError("Unexpected passing replay; this closure concerns the failed gate")
    failure = read(returned / "recovery/recover-failure.json")
    events = lines(returned / "recovery/recover.events.jsonl")
    if (
        failure["status"] != "failed"
        or failure["resumptions_used"] != 1
        or failure["error"] != "Replay differs beyond the existing recovery loss tolerance"
        or events[-1]["event"] != "worker-complete"
        or events[-1]["returncode"] != 0
        or (returned / "recovery/recover-result.json").exists()
        or any(e["event"] == "complete" for e in events)
    ):
        raise ValueError("Corrective failure/completion evidence inconsistent")
    forbidden = ("attempt/development", "attempt/final", "recovery/development")
    if any(name.startswith(forbidden) for name in all_records):
        raise ValueError("Unexpected fixed evaluation or further execution")
    segments = sorted(p.name for p in (returned / "attempt").glob("segment-*"))
    if segments != ["segment-0", "segment-1"]:
        raise ValueError("Unexpected additional resumption")
    old_state = read(returned / "attempt/checkpoints/step-00007000/state.json")
    final_dir = returned / "attempt/checkpoints/step-00007485"
    state = read(final_dir / "state.json")
    for key in ("config", "datasets", "runtime", "kind", "precision"):
        if state[key] != old_state[key]:
            raise ValueError("Recovered checkpoint contract changed")
    if (
        state["step"] != 7485
        or state["tokens_seen"] != 37811418
        or state["cursor"] != 119748
        or state["epoch"] != 0
        or state["windows_seen"] != 119748
    ):
        raise ValueError("Final worker counters differ")
    prefix = "attempt/checkpoints/step-00007485/"
    if all_records[prefix + "training.safetensors"] != {
        "bytes": state["training_bytes"],
        "sha256": state["training_sha256"],
    }:
        raise ValueError("Omitted final optimizer inventory disagrees with checkpoint")
    if file_hash(final_dir / "model/metadata.json") != state["model_metadata_sha256"]:
        raise ValueError("Recovered model metadata differs")
    # Safe CPU artifact inspection only: no forward pass, optimizer, sampling or scoring.
    model = load_model(
        final_dir / "model",
        expected_tokenizer_sha256=state["datasets"]["train"]["tokenizer_sha256"],
    )
    expected_model = json.loads(
        subprocess.check_output(
            ["git", "show", f"{TRAINING_COMMIT}:configs/training2/selected-model.json"], cwd=root
        )
    )
    if model.config.to_dict() != expected_model or model.parameter_count != 34087424:
        raise ValueError("Recovered model differs from frozen configuration")
    weight_hash = file_hash(final_dir / "model/model.safetensors")
    worker = read(returned / "attempt/training-result.json")
    if worker["candidate_weights"] != weight_hash or worker["status"] != "complete":
        raise ValueError("Worker/model identity mismatch")
    monitor = read(returned / "attempt/segment-1/boundary-00007485.json")
    peak = {
        key: max(
            row[key]
            for row in [
                *lines(returned / "attempt/segment-1/updates.jsonl"),
                monitor["resources"],
                worker["resources"],
            ]
        )
        for key in ("peak_allocated_bytes", "peak_reserved_bytes", "host_peak_working_set_bytes")
    }
    plan = read(returned / "recovery-bundle/recovery-plan.json")
    training_seconds = plan["prior_training_seconds_charged"] + failure["active_seconds"]
    artifact_peak = max(e.get("artifact_bytes", 0) for e in events)
    gpu_total = read(returned / "attempt/segment-1/environment.json")["total_bytes"]
    resource_pass = (
        training_seconds < 7200
        and peak["peak_reserved_bytes"] < 0.85 * gpu_total
        and peak["host_peak_working_set_bytes"] < 8 * 1024**3
        and artifact_peak < 20 * 1024**3
    )
    if not resource_pass:
        raise ValueError("Unexpected resource violation; must be reported separately")
    return {
        "return_sha256": ARCHIVE_SHA256,
        "verified_return_files": len(manifest["files"]),
        "omitted_files": len(manifest["omitted_retained_on_Windows"]),
        "original_return_records_unchanged": len(first["files"])
        + len(first["omitted_retained_on_Windows"]),
        "training_source_commit": TRAINING_COMMIT,
        "administrative_commit": ADMIN_COMMIT,
        "source_and_exposure_verification": read(original_verification),
        "Windows_administrative_tests": {"passed": 9, "skipped": 0},
        "replay": summary,
        "original_final_checkpoint": "never produced; not recreated as original evidence",
        "recovered_final_model": {
            "sha256": weight_hash,
            "bytes": (final_dir / "model/model.safetensors").stat().st_size,
            "parameters": model.parameter_count,
            "finite_float32_shapes_load_on_CPU": True,
            "status": "unaccepted experimental recovered model; retain, do not promote",
        },
        "optimizer_tensor_scope": "Final and historical optimizer tensors retained on Windows; "
        "only inventory identities verified on Mac",
        "training_monitoring_only": monitor["validation"],
        "fixed_acceptance_scores": None,
        "resource_limits_passed": resource_pass,
        "recovery_resource_peaks": peak,
        "peak_charged_artifact_bytes": artifact_peak,
        "training_seconds_charged": training_seconds,
        "evaluation_seconds_used": 0,
        "evaluation_reserve_seconds": 3600,
        "resumptions_consumed": 1,
        "further_retry_permitted": False,
        "disposition": "bounded Phase 17 experiment closed as interrupted/failed; "
        "recovery gate failed, learned-quality exit gate unmet and unmeasured",
        "phase17_exit_gate_passed": False,
        "phase18_authorized": False,
        "publication_authorized": False,
        "causation": "Replay divergence cause not established; original PowerShell involvement "
        "remains plausible, unproven",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returned", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--first-return", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Closure verification output exists")
    result = verify(Path.cwd(), args.returned, args.archive, args.first_return, args.verification)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
