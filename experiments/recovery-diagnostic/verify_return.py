"""Independent Mac evidence reconstruction; never execute returned source or optimize."""

import argparse
import hashlib
import json
import math
import subprocess
import zipfile
from pathlib import Path

import mechanics as m
import run as runner

from latos.data.manifest import canonical_json
from latos.training.checkpoint import implementation_hash


def extract(archive_path, destination, expected_sha256):
    if destination.exists() or runner.hash_file(archive_path) != expected_sha256:
        raise ValueError("Return identity/destination mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read("return.json"))
        if len(set(names)) != len(names) or set(names) != set(manifest["files"]) | {"return.json"}:
            raise ValueError("Duplicate or unexpected archive entries")
        if sum(info.file_size for info in archive.infolist()) > 6 * 1024**3:
            raise ValueError("Return exceeds diagnostic artifact ceiling")
        destination.mkdir(parents=True)
        for name, record in manifest["files"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe return path")
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with archive.open(name) as source, output.open("xb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
                    h.update(chunk)
            if output.stat().st_size != record["bytes"] or h.hexdigest() != record["sha256"]:
                raise ValueError("Return file mismatch")
        (destination / "return.json").write_bytes(archive.read("return.json"))
    return manifest


def state_valid(state):
    core = {k: v for k, v in state.items() if k != "state_sha256"}
    if hashlib.sha256(canonical_json(core)).hexdigest() != state["state_sha256"]:
        raise ValueError("State digest does not reconstruct")


def verify(returned):
    bundle = returned / "bundle"
    plan, manifest = runner.verify_bundle(bundle)
    commit = manifest["source_commit"]
    for name, record in manifest["files"].items():
        raw = subprocess.check_output(["git", "show", f"{commit}:{record['source_path']}"])
        if raw != (bundle / name).read_bytes():
            raise ValueError("Returned diagnostic code differs from reviewed commit")
    if implementation_hash() != plan["legacy_implementation_sha256"]:
        raise ValueError(
            "Local core no longer matches the measured core; inspect exact source first"
        )
    expected = {
        p.relative_to(Path("src/latos")).as_posix(): runner.hash_file(p)
        for p in Path("src/latos").rglob("*.py")
    }
    snapshots = list((returned / "run").rglob("source/latos"))
    for snapshot in snapshots:
        actual = {
            p.relative_to(snapshot).as_posix(): runner.hash_file(p) for p in snapshot.rglob("*.py")
        }
        if actual != expected:
            raise ValueError("Executed core snapshot differs")
    record = m.read(returned / "return.json")
    all_records = {**record["files"], **record["omitted_retained_on_Windows"]}
    for inventory_path in (returned / "run").glob("*/*/inventory.json"):
        prefix = inventory_path.parent.relative_to(returned).as_posix() + "/"
        inventory = m.read(inventory_path)
        actual = {
            name.removeprefix(prefix): value
            for name, value in all_records.items()
            if name.startswith(prefix) and name != prefix + "inventory.json"
        }
        if inventory != actual:
            raise ValueError("Stage inventory incomplete or changed")
    shapes = m.read(bundle / "lengths.json")["lengths"]
    cumulative = [0]
    for length in shapes:
        cumulative.append(cumulative[-1] + length - 1)
    config = m.training_config(plan)
    total = 0
    for trace in sorted((returned / "run").glob("*/*/trace.jsonl")):
        records = runner.read_trace(trace)
        previous = plan["checkpoint_step"] if trace.parent.name == "resumed" else 0
        for row in records:
            state_valid(row["state"])
            metric = row["metric"]
            step = metric["step"]
            if step != previous + 1 or step > plan["steps"]:
                raise ValueError("Noncontiguous/out-of-budget update trace")
            start, end = (step - 1) * 16, min(step * 16, len(shapes))
            if (
                row["input"]["lengths"] != shapes[start:end]
                or metric["targets"] != cumulative[end] - cumulative[start]
                or metric["tokens_seen"] != cumulative[end]
                or metric["windows_seen"] != end
                or metric["epoch"] != 0
                or metric["microbatches"] != math.ceil((end - start) / 2)
            ):
                raise ValueError("Exposure/shape schedule does not reconstruct")
            lr = config.learning_rate_at(step)
            if abs(metric["learning_rate"] - lr) > math.ulp(lr):
                raise ValueError("Schedule mismatch beyond cross-host float64 ULP")
            previous = step
            total += 1
    if total > plan["max_physical_updates"]:
        raise ValueError("Physical diagnostic update cap exceeded")
    summary = returned / "run/summary.json"
    if not summary.exists():
        return {
            "status": "incomplete_or_failed",
            "verified_logged_updates": total,
            "prospective_correction_established": False,
            "phase17_1_authorized": False,
            "next": "Inspect retained failures; no automatic retry or Phase 17.1",
        }
    if len(snapshots) != 4:
        raise ValueError("Complete study requires four bound executed-source snapshots")
    for profile in plan["profiles"]:
        for mode in ("reference", "resumed"):
            stage = returned / "run" / profile / mode
            for filename in ("start.json", "final.json"):
                saved = m.read(stage / filename)
                if filename == "start.json":
                    state_valid(saved["state"])
                else:
                    for key in ("before", "after", "loaded"):
                        state_valid(saved[key])
            if not (stage / "inventory.json").is_file():
                raise ValueError("Missing stage inventory")
    rebuilt = runner.compare_all(returned / "run", plan)
    observed = m.read(summary)
    if any(observed[k] != v for k, v in rebuilt.items()) or total != plan["max_physical_updates"]:
        raise ValueError("Comparison summary does not reconstruct")
    if observed["active_seconds"] >= plan["max_seconds"]:
        raise ValueError("Time limit exceeded")
    return {
        **rebuilt,
        "verified_logged_updates": total,
        "diagnostic_source_commit": commit,
        "scope": "Trace/state-digest equality and source binding reconstructed; "
        "Windows tensor payloads not present or numerically replayed on Mac",
        "next": "Separate Mac review before any prospective Phase 17.1 handoff; "
        "no full training authorization",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Report already exists")
    extract(args.archive, args.destination, args.sha256)
    result = verify(args.destination)
    runner.write_once(args.report, result)
    print(json.dumps(result, indent=2))
