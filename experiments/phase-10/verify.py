"""Read back DPO artifacts, replay exposures and check validation with a scalar oracle."""

import argparse
import json
import math
from pathlib import Path

import torch

from latos.chat import format_chat
from latos.data.manifest import canonical_json
from latos.model.storage import bind_tokenizer, file_hash, load_model
from latos.preferences import DPOConfig, prepare_preferences
from latos.training.data import ShuffleStream


@torch.no_grad()
def verify(args):
    root = args.run_dir
    plan = json.loads((root / "plan.json").read_text())
    result = json.loads((root / "summary.json").read_text())
    baseline = json.loads((root / "baseline.json").read_text())
    records = json.loads((root / "preferences.json").read_text())
    inventory = json.loads((root / "artifacts.json").read_text())
    for name, entry in inventory.items():
        path = root / name
        if path.stat().st_size != entry["bytes"] or file_hash(path) != entry["sha256"]:
            raise ValueError(f"Artifact changed: {name}")
    required = {
        "plan.json",
        "baseline.json",
        "summary.json",
        "metrics.jsonl",
        "preferences.json",
        "config.json",
        "uv.lock",
        "model/model.safetensors",
        "model/config.json",
        "model/metadata.json",
    } | set(plan["experiment_files"])
    if not required <= set(inventory):
        raise ValueError("Incomplete artifact inventory")
    for name, digest in plan["experiment_files"].items():
        if file_hash(root / name) != digest:
            raise ValueError("Experiment file identity mismatch")
    config = DPOConfig(**plan["config"])
    if file_hash(args.base / "model.safetensors") != plan["base_weights_sha256"]:
        raise ValueError("Wrong fixed SFT baseline")
    reference = load_model(args.base).to(args.device)
    policy = load_model(root / "model").to(args.device)
    codec = bind_tokenizer(policy.config, args.tokenizer_dir)
    train = prepare_preferences(
        records["train"], codec, plan["tokenizer_sha256"], config.sequence_length, "train"
    )
    metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    if len(metrics) != config.steps or result["steps"] != config.steps:
        raise ValueError("Incomplete DPO run")
    stream = ShuffleStream(len(records["train"]), config.seed)
    targets, pairs, visited = 0, 0, set()
    for step, row in enumerate(metrics, 1):
        indices = stream.take(config.batch_size)
        count = sum(train.target_count(2 * i) + train.target_count(2 * i + 1) for i in indices)
        targets += count
        pairs += len(indices)
        visited.update(indices)
        expected = {
            "step": step,
            "indices": indices,
            "epoch": stream.epoch,
            "targets": count,
            "targets_seen": targets,
            "pairs_seen": pairs,
        }
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError("Exposure replay mismatch")
    if result["targets_seen"] != targets or result["pairs_seen"] != pairs:
        raise ValueError("Summary exposure mismatch")
    maxima = []
    # Independent scoring: ordinary per-token probabilities with explicit final-body boundary.
    # No sequence_logps, pair_logps, dpo_loss or training target masks are used here.
    for record_index, record in enumerate(records["validation"]):
        prefix, _ = format_chat(
            codec, record["prompt"], max_length=config.sequence_length, generation_prompt=True
        )
        scores = []
        for model in (policy, reference):
            pair = []
            for side in ("chosen", "rejected"):
                ids, _ = format_chat(
                    codec,
                    record["prompt"] + [{"role": "assistant", "content": record[side]}],
                    max_length=config.sequence_length,
                )
                logits = model(torch.tensor([ids], device=args.device)).logits[0].cpu().double()
                total = 0.0
                for position in range(len(prefix), len(ids)):
                    values = logits[position - 1]
                    maximum = values.max().item()
                    denominator = sum(math.exp(v - maximum) for v in values.tolist())
                    total += values[ids[position]].item() - maximum - math.log(denominator)
                pair.append(total)
            scores.append(pair)
        saved = result["preferences"]["validation"]
        for computed, expected in zip(
            scores,
            (saved["sequence_logps"][record_index], saved["reference_logps"][record_index]),
            strict=True,
        ):
            error = max(abs(a - b) for a, b in zip(computed, expected, strict=True))
            if error > 1e-4:
                raise ValueError("Independent response score mismatch")
            maxima.append(error)
    p = result["preferences"]["validation"]
    margins = [
        config.beta * ((a[0] - a[1]) - (b[0] - b[1]))
        for a, b in zip(p["sequence_logps"], p["reference_logps"], strict=True)
    ]
    scalar_loss = sum(max(0, -m) + math.log1p(math.exp(-abs(m))) for m in margins) / len(margins)
    if abs(scalar_loss - p["loss"]) > 1e-6:
        raise ValueError("Independent DPO objective mismatch")
    for split in ("train", "validation"):
        b = baseline["preferences"][split]
        if b["relative_ties"] != b["pairs"] or abs(b["loss"] - math.log(2)) > 1e-6:
            raise ValueError("Initial reference baseline not tied")
    return {
        "inventory_files_verified": len(inventory),
        "complete_updates": config.steps,
        "pairs_accounted": pairs,
        "response_targets_accounted": targets,
        "distinct_pairs_seen": len(visited),
        "distinct_response_targets_seen": sum(
            train.target_count(2 * i) + train.target_count(2 * i + 1) for i in visited
        ),
        "oracle_device": args.device,
        "response_logp_absolute_tolerance": 1e-4,
        "maximum_oracle_logp_error": max(maxima),
        "scalar_dpo_loss": scalar_loss,
        "scalar_dpo_loss_absolute_tolerance": 1e-6,
        "baseline_ties_verified": True,
        "test_payloads_read": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("outputs/phase-6-instruction-reviewed/step-00000200/model"),
    )
    parser.add_argument(
        "--tokenizer-dir", type=Path, default=Path("artifacts/tokenizers/english-bpe-v1")
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = verify(args)
    with args.output.open("xb") as stream:
        stream.write(canonical_json(result))
    print(json.dumps(result, indent=2))
