"""Independently bind the first Windows interruption to the frozen experiment."""

import argparse
import hashlib
import json
import runpy
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from latos.model.storage import file_hash

SOURCE = "0fa94ad580110fd2dc7aaa2aa3560950abe73a02"
RETURN = "b4d959194b29f68ae02f7c3340e70f02d88d6aeb0d10e7b9ab4eebe7d0b5338e"


def review(root, returned, cache):
    def read(p):
        return json.loads(p.read_text())

    manifest = read(returned / "return.json")
    handoff = read(returned / "handoff.json")
    if handoff["source_commit"] != SOURCE:
        raise ValueError("Unexpected executed source")
    original = json.loads(
        subprocess.check_output(
            ["git", "show", f"{SOURCE}:configs/training2/phase17-plan.json"], cwd=root
        )
    )
    verification = runpy.run_path(str(root / "experiments/phase-17/verify_return.py"))["verify"](
        root, returned, cache
    )
    all_files = {**manifest["files"], **manifest["omitted_retained_on_Windows"]}
    if set(manifest["files"]) & set(manifest["omitted_retained_on_Windows"]):
        raise ValueError("Ambiguous return inventory")
    audit = read(returned / "validation/checkpoint-and-artifact-inventory.json")
    attempt_files = {
        n.removeprefix("attempt/"): r for n, r in all_files.items() if n.startswith("attempt/")
    }
    if attempt_files != audit["attempt_files"]:
        raise ValueError("Independent return and attempt inventories disagree")
    for name, record in manifest["files"].items():
        path = returned / name
        if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
            raise ValueError("Extracted evidence changed")
    # Reconstruct the implementation identity from Git, not a reported status.
    source_names = sorted(
        n.removeprefix("source/src/latos/")
        for n in handoff["files"]
        if n.startswith("source/src/latos/") and n.endswith(".py")
    )
    impl = hashlib.sha256()
    for name in source_names:
        data = subprocess.check_output(["git", "show", f"{SOURCE}:src/latos/{name}"], cwd=root)
        impl.update(name.encode())
        impl.update(data.replace(b"\r\n", b"\n"))
    checkpoints = []
    for step in range(0, 7001, 1000):
        directory = returned / "attempt/checkpoints" / f"step-{step:08d}"
        state = read(directory / "state.json")
        model = read(directory / "model/metadata.json")
        if (
            state["runtime"]["implementation_sha256"] != impl.hexdigest()
            or state["step"] != step
            or state["epoch"] != 0
            or state["precision"] != "bfloat16"
            or state["model_metadata_sha256"] != file_hash(directory / "model/metadata.json")
        ):
            raise ValueError("Checkpoint metadata/source binding mismatch")
        prefix = f"attempt/checkpoints/step-{step:08d}/"
        if all_files[prefix + "training.safetensors"] != {
            "bytes": state["training_bytes"],
            "sha256": state["training_sha256"],
        }:
            raise ValueError("Optimizer tensor inventory mismatch")
        tensor = all_files[prefix + "model/model.safetensors"]
        if tensor["sha256"] != model["weights_sha256"]:
            raise ValueError("Model tensor inventory mismatch")
        checkpoints.append(
            {
                "step": step,
                "targets": state["tokens_seen"],
                "windows": state["windows_seen"],
                "model_sha256": tensor["sha256"],
            }
        )
    for split in ("train", "development"):
        actual = read(returned / f"attempt/cache-{split}/cache.json")
        local = read(cache.parent / split / "cache.json")
        if actual != local:
            raise ValueError("Returned full-cache identities differ from local verified cache")
    receipt = read(returned / "preflight.json")
    tests = {}
    for kind in ("checkout", "wheel"):
        path = returned / f"validation/{kind}.xml"
        cases = list(ET.parse(path).getroot().iter("testcase"))
        if file_hash(path) != receipt[kind]["sha256"]:
            raise ValueError("Preflight test evidence differs")
        if any(c.find(k) is not None for c in cases for k in ("failure", "error")):
            raise ValueError("Failed tests")
        skips = [c.attrib["name"] for c in cases if c.find("skipped") is not None]
        required = {"test_cuda_bf16_numerical_control", "test_full_single_pass_cuda"}
        passed = {c.attrib["name"] for c in cases if c.find("skipped") is None}
        if not required <= passed:
            raise ValueError("Required CUDA controls missing")
        tests[kind] = {
            "passed": len(cases) - len(skips),
            "skipped": len(skips),
            "failed": 0,
            "CUDA_controls_passed": sorted(required),
        }
    wheel = next((returned / "validation/dist").glob("*.whl"))
    if file_hash(wheel) != receipt["wheel_sha256"]:
        raise ValueError("Tested wheel differs")
    with zipfile.ZipFile(wheel) as archive:
        for name in source_names:
            expected = handoff["files"]["source/src/latos/" + name]["sha256"]
            if hashlib.sha256(archive.read("latos/" + name)).hexdigest() != expected:
                raise ValueError("Wheel source differs from reviewed code")
    ledger = read(returned / "attempt/ledger.json")
    failed = read(returned / "attempt/ledger.json.tmp")
    if (
        ledger["jobs"][0]["status"] != "running"
        or failed["jobs"][0]["status"] != "supervisor-failure"
        or failed["segments"] != 1
        or failed["evaluation_seconds"] != 0
        or "WinError 5" not in failed["jobs"][0]["error"]
    ):
        raise ValueError("Unexpected interruption signature")
    logs = [
        json.loads(s)
        for s in (returned / "attempt/segment-0/updates.jsonl").read_text().splitlines()
    ]
    last = logs[-1]
    after = logs[7000:]
    if (
        len(logs) != 7485
        or last["microbatches"] != 2
        or last["tokens_seen"] != original["data"]["targets_with_EOS"]
        or any(n.startswith(("attempt/development", "attempt/final")) for n in all_files)
        or any("step-00007485" in n for n in attempt_files)
        or "training-result.json" in attempt_files
    ):
        raise ValueError("Unexpected run completion/evaluation evidence")
    return {
        "source_commit": SOURCE,
        "return_sha256": RETURN,
        "verification": verification,
        "implementation_sha256": impl.hexdigest(),
        "tests": tests,
        "checkpoint_records": checkpoints,
        "checkpoint_bytes_limit": "Inventory identities on Mac; tensor rehash/load still required "
        "on Windows",
        "original_status": "supervisor-failure; main ledger stale running",
        "cause": "PermissionError WinError 5 replacing ledger; PowerShell read contribution "
        "plausible, not proven",
        "active_training_seconds": failed["training_seconds"],
        "recovery_prior_seconds_conservative": 1562,
        "remaining_training_seconds": 7200 - 1562,
        "remaining_evaluation_seconds": 3600,
        "lost_completed_updates": len(after),
        "lost_completed_targets": sum(m["targets"] for m in after),
        "logical_pass_targets": last["tokens_seen"],
        "physical_target_executions_if_tail_replayed_once": last["tokens_seen"]
        + sum(m["targets"] for m in after),
        "logged_resource_peaks": {
            k: max(m[k] for m in logs)
            for k in ("peak_allocated_bytes", "peak_reserved_bytes", "host_peak_working_set_bytes")
        },
        "resource_limit_scope": "Logged peaks only; abrupt termination lacks final worker receipt",
        "serious_attempts": 1,
        "resumptions_used": 0,
        "final_checkpoint": False,
        "fixed_evaluation_executed": False,
        "phase17_complete": False,
        "phase18_authorized": False,
        "publication_authorized": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returned", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Review result already exists")
    result = review(Path.cwd(), args.returned, args.cache)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
