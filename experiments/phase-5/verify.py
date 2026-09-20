"""Independently re-read completed pilot evidence and exercise saved MPS recovery."""

import argparse
import json
import math
from pathlib import Path

import torch

from latos.data.manifest import canonical_json
from latos.model import ModelConfig, create_model
from latos.model.storage import bind_tokenizer, file_hash
from latos.training.checkpoint import load_checkpoint
from latos.training.data import prepare_dataset


def verify(directory: Path, output: Path):
    if output.exists():
        raise ValueError("Verification report already exists")
    plan = json.loads((directory / "plan.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    baseline = json.loads((directory / "baseline.json").read_text())
    inventory = json.loads((directory / "artifacts.json").read_text())
    metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    for name, entry in inventory.items():
        path = directory / name
        if file_hash(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Artifact mismatch: {name}")
    if file_hash(directory / "runner.py") != plan["runner_sha256"]:
        raise ValueError("Runner provenance mismatch")
    if [m["step"] for m in metrics] != list(range(1, summary["steps"] + 1)):
        raise ValueError("Metrics have missing or repeated updates")
    if sum(m["targets"] for m in metrics) != summary["tokens_seen"]:
        raise ValueError("Exposure does not equal the update totals")
    if summary["steps"] != plan["config"]["max_steps"]:
        raise ValueError("Run stopped before its predetermined final checkpoint")
    torch.set_num_threads(plan["runtime"]["cpu_threads"])
    config = ModelConfig(**plan["model_config"])
    codec = bind_tokenizer(config, Path("artifacts/tokenizers/english-bpe-v1"))
    datasets = [
        prepare_dataset(
            Path("data/manifests/english-books-v1.json"),
            Path("data/processed/english-books-v1"),
            codec,
            config.tokenizer_sha256,
            plan["config"]["sequence_length"],
            split,
        )
        for split in ("train", "validation")
    ]
    if {d.split: d.identity for d in datasets} != plan["datasets"]:
        raise ValueError("Dataset identities changed")
    device = plan["runtime"]["device"]
    initial = load_checkpoint(directory / "step-00000000", *datasets, device)
    random_model = create_model(config, plan["config"]["seed"])
    if any(
        not torch.equal(v.cpu(), random_model.state_dict()[k])
        for k, v in initial.model.state_dict().items()
    ):
        raise ValueError("Initial checkpoint is not the declared seeded random model")
    initial_validation = initial.validate()
    if not math.isclose(initial_validation["loss"], baseline["validation"]["loss"], abs_tol=1e-6):
        raise ValueError("Recomputed baseline differs")
    del initial, random_model
    final = load_checkpoint(directory / summary["selected_checkpoint"], *datasets, device)
    final_validation = final.validate()
    if not math.isclose(
        final_validation["loss"], summary["final_validation"]["loss"], abs_tol=1e-6
    ):
        raise ValueError("Recomputed final validation differs")
    if final.tokens_seen != summary["tokens_seen"]:
        raise ValueError("Final checkpoint exposure mismatch")
    del final
    # Resume one real update from the penultimate checkpoint, in this new process.
    checkpoints = sorted(directory.glob("step-*"))
    recovery = checkpoints[-2]
    restored = load_checkpoint(recovery, *datasets, device)
    expected = metrics[restored.step]
    actual = restored.update()
    if actual["tokens_seen"] != expected["tokens_seen"]:
        raise ValueError("Recovery changed shuffle/exposure")
    if not math.isclose(actual["loss"], expected["loss"], rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("Recovery loss exceeded declared MPS tolerance")
    report = {
        "status": "ok",
        "artifact_files_verified": len(inventory),
        "inventory_sha256": file_hash(directory / "artifacts.json"),
        "all_updates_accounted_for": len(metrics),
        "initial_random_tensors_equal": True,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "recovery_checkpoint": recovery.name,
        "replayed_step": actual["step"],
        "replayed_loss_abs_difference": abs(actual["loss"] - expected["loss"]),
        "replayed_loss_tolerance": {"atol": 1e-6, "rtol": 1e-5},
        "replayed_target_exposure_equal": True,
        "scope": "same-host/runtime scalar recovery check, not general bitwise determinism",
        "test_split": "reserved; not opened",
    }
    output.write_bytes(canonical_json(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run_dir, args.output)
