"""Mac-side independent return verification; never execute returned source."""

import argparse
import hashlib
import json
import math
import subprocess
import zipfile
from pathlib import Path

import torch

from latos.evaluation.compare import compare
from latos.model.storage import file_hash
from latos.training.base2 import frozen, training_config
from latos.training.full_data import FullDataset, expected_accounting
from latos.training.probe import TOKENIZER_SHA256


def unpack(archive_path, destination):
    if destination.exists():
        raise ValueError("Return destination exists")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive entries")
        manifest = json.loads(archive.read("return.json"))
        if set(names) != set(manifest["files"]) | {"return.json"}:
            raise ValueError("Unexpected return entries")
        if sum(i.file_size for i in archive.infolist()) > 20 * 1024**3:
            raise ValueError("Return exceeds artifact budget")
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe return path")
        destination.mkdir(parents=True)
        for name, record in manifest["files"].items():
            info = archive.getinfo(name)
            if info.file_size != record["bytes"]:
                raise ValueError("Return size mismatch")
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(name) as source, path.open("xb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
                    digest.update(block)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError("Return hash mismatch")
        (destination / "return.json").write_bytes(archive.read("return.json"))
    return manifest


def verify(root, returned, cache):
    record = json.loads((returned / "return.json").read_text())
    handoff = json.loads((returned / "handoff.json").read_text())
    commit = handoff["source_commit"]
    expected = {}
    for name, identity in handoff["files"].items():
        if name.startswith("source/"):
            relative = name.removeprefix("source/")
            data = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=root)
            if hashlib.sha256(data).hexdigest() != identity["sha256"]:
                raise ValueError("Returned handoff source differs from reviewed commit")
            if relative.startswith("src/latos/") and relative.endswith(".py"):
                expected[relative.removeprefix("src/latos/")] = identity["sha256"]
    snapshots = 0
    for package in (returned / "attempt").rglob("source/latos"):
        actual = {p.relative_to(package).as_posix(): file_hash(p) for p in package.rglob("*.py")}
        if actual != expected:
            raise ValueError("Executed runtime snapshot mismatch")
        snapshots += 1
    plan = frozen(root)
    layout = json.loads((root / "experiments/phase-16/full-corpus-layout.json").read_text())
    dataset = FullDataset(cache, expected_accounting(layout, "train"), TOKENIZER_SHA256, "train")
    config = training_config(plan)
    order = torch.randperm(
        len(dataset.windows), generator=torch.Generator().manual_seed(160)
    ).tolist()
    prefix = [0]
    for index in order:
        prefix.append(prefix[-1] + dataset.target_count(index))
    executions = 0
    observed_steps = set()
    for log in sorted((returned / "attempt").glob("segment-*/updates.jsonl")):
        start = json.loads((log.parent / "start.json").read_text())
        if start["datasets"]["train"] != dataset.identity or start["config"] != config.to_dict():
            raise ValueError("Training data/configuration identity mismatch")
        previous = start["start_step"]
        for line in log.read_text().splitlines():
            metric = json.loads(line)
            step = metric["step"]
            begin = min((step - 1) * 16, len(order))
            end = min(step * 16, len(order))
            if step != previous + 1 or not 1 <= step <= 7485:
                raise ValueError("Noncontiguous/out-of-budget update record")
            if (
                metric["targets"] != prefix[end] - prefix[begin]
                or metric["tokens_seen"] != prefix[end]
                or metric["windows_seen"] != end
                or metric["epoch"] != 0
                or metric["microbatches"] != math.ceil((end - begin) / 2)
            ):
                raise ValueError("Exposure cannot be independently reconstructed")
            if not math.isclose(
                metric["learning_rate"], config.learning_rate_at(step), rel_tol=1e-15, abs_tol=0.0
            ):
                raise ValueError("Learning rate differs from frozen schedule")
            if not all(
                math.isfinite(metric[k])
                for k in ("loss", "grad_norm_before_clip", "seconds", "targets_per_second")
            ):
                raise ValueError("Nonfinite completed update")
            previous = step
            executions += metric["targets"]
            observed_steps.add(step)
    comparisons = {}
    for stage in ("development", "final"):
        path = returned / "attempt" / f"{stage}-comparison.json"
        if path.exists():
            actual = compare(
                path.parent / f"{stage}-reference", path.parent / f"{stage}-candidate", "base"
            )
            if actual != json.loads(path.read_text()):
                raise ValueError("Comparison does not reconstruct")
            comparisons[stage] = actual
    return {
        "verified_return_files": len(record["files"]),
        "reviewed_source_commit": commit,
        "executed_source_snapshots": snapshots,
        "logged_target_executions_including_replay": executions,
        "distinct_logged_updates": len(observed_steps),
        "comparisons": comparisons,
        "omitted_tensors_scope": "Windows inventory identities only; absent bytes not rehashed",
        "phase17_complete": False,
        "next": "Separate source, failure, resource and learned-result review; "
        "not automatic acceptance",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Report exists")
    unpack(args.archive, args.destination)
    result = verify(Path.cwd(), args.destination, args.cache)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
