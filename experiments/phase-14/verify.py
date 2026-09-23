"""Check complete retrospective runs, repeatability and preserved protocol identities."""

import argparse
import math
from pathlib import Path

from latos.evaluation.compare import compare, verified_run
from latos.evaluation.inputs import read_json, write_json
from latos.model.storage import file_hash


def equivalent(left, right, location="root"):
    if type(left) is not type(right):
        raise ValueError(f"Repeat type mismatch: {location}")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError(f"Repeat field mismatch: {location}")
        for key in left:
            if key not in ("prefill_ms", "first_token_ms", "subsequent_token_ms", "resources"):
                equivalent(left[key], right[key], location + "." + key)
    elif isinstance(left, list):
        if len(left) != len(right):
            raise ValueError(f"Repeat case count mismatch: {location}")
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            equivalent(a, b, location + f"[{i}]")
    elif isinstance(left, float):
        if not math.isclose(left, right, abs_tol=1e-5, rel_tol=1e-6):
            raise ValueError(f"Repeat numerical mismatch: {location}")
    elif left != right:
        raise ValueError(f"Repeat value mismatch: {location}")


def verify(directory):
    names = ["base", "sft", "lora-merged", "full-control", "dpo", "base-repeat"]
    summaries, identities = {}, {}
    for name in names:
        manifest, summaries[name] = verified_run(directory / name)
        identities[name] = {
            "weights_sha256": manifest["model"]["weights_sha256"],
            "inventory_sha256": file_hash(directory / name / "inventory.json"),
            "source_sha256": manifest["source"]["snapshot_sha256"],
            "device": manifest["environment"]["device"],
        }
    if len({row["source_sha256"] for row in identities.values()}) != 1:
        raise ValueError("Retrospective panel did not use one implementation")
    for name in (
        "summary.json",
        "lm-legacy.json",
        "lm-matched.json",
        "ARC-Easy.json",
        "ARC-Challenge.json",
        "instructions.json",
        "generation.json",
    ):
        equivalent(
            read_json(directory / "base" / name), read_json(directory / "base-repeat" / name)
        )
    comparisons = {
        name: compare(directory / "base", directory / name, "assistant", directory / "base")
        for name in names[1:-1]
    }
    return {
        "schema_version": 1,
        "kind": "Phase 14 retrospective measurements, not original experiment scores",
        "runs": identities,
        "summaries": summaries,
        "comparisons": comparisons,
        "repeatability": {
            "passed": True,
            "scope": (
                "base, same host/backend, two separate processes; "
                "raw rows and summaries excluding timings"
            ),
            "nll_atol": 1e-5,
            "nll_rtol": 1e-6,
            "generated_ids_and_predictions": "exact",
        },
        "reserved_model_evaluations": 0,
        "training_updates": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Verification output already exists")
    write_json(args.output, verify(args.directory))
