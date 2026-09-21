"""Read-back integrity and full-artifact numerical checks, without test-set access."""

import argparse
import json
from pathlib import Path

import torch

from latos.adaptation import agreement
from latos.data.manifest import canonical_json
from latos.inference.cache import KVCache
from latos.instruction import prepare_conversations
from latos.lora import load_adapter, merge_adapter, state_identity
from latos.model.storage import bind_tokenizer, file_hash, load_model
from latos.training.data import ShuffleStream


def verify(root, base_dir, tokenizer_dir, conversations, device):
    torch.set_num_threads(1)
    files = 0
    for method in ("lora", "full"):
        directory = root / method
        inventory = json.loads((directory / "artifacts.json").read_text())
        for name, expected in inventory.items():
            path = directory / name
            if path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
                raise ValueError(f"Artifact integrity failed: {method}/{name}")
            files += 1
    plan = json.loads((root / "lora/plan.json").read_text())
    if file_hash(base_dir / "model.safetensors") != plan["base_weights_sha256"]:
        raise ValueError("Base artifact identity mismatch")
    base = load_model(base_dir)
    before = state_identity(base.config, base.state_dict())
    codec = bind_tokenizer(base.config, tokenizer_dir)
    validation = prepare_conversations(
        conversations,
        codec,
        base.config.tokenizer_sha256,
        plan["config"]["sequence_length"],
        "validation",
    )
    train = prepare_conversations(
        conversations,
        codec,
        base.config.tokenizer_sha256,
        plan["config"]["sequence_length"],
        "train",
    )
    stream = ShuffleStream(len(train.windows), plan["config"]["seed"])
    exposure = 0
    for metric in [
        json.loads(line) for line in (root / "lora/metrics.jsonl").read_text().splitlines()
    ]:
        indices = [
            index
            for _ in range(plan["config"]["accumulation_steps"])
            for index in stream.take(plan["config"]["batch_size"])
        ]
        count = sum(train.target_count(i) for i in indices)
        exposure += count
        if metric["targets"] != count or metric["tokens_seen"] != exposure:
            raise ValueError("Independent exposure replay failed")
    adapted = load_adapter(base, root / "lora/adapter").to(device)
    adapted.validate_adapter()
    merged = load_model(root / "lora/merged").to(device)
    repeated_merge = merge_adapter(adapted)
    # Merge arithmetic is defined on CPU; serialized merged weights must match exactly.
    for name, tensor in repeated_merge.state_dict().items():
        if not torch.equal(tensor, merged.state_dict()[name].cpu()):
            raise ValueError("Merged artifact differs from independently repeated merge")
    tolerance = 1e-4
    cache_tolerance = 1e-4 if device == "mps" else 1e-5
    validation_result = agreement(adapted, merged, validation, device, tolerance=tolerance)
    ids = (
        torch.arange(2 * base.config.context_length).reshape(2, -1) * 37 % base.config.vocab_size
    ).to(device)
    with torch.inference_mode():
        a, b = adapted(ids).logits, merged(ids).logits
    torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
    capacity = {
        "batch": 2,
        "positions": ids.shape[1],
        "cache_atol": cache_tolerance,
        "cache_rtol": cache_tolerance,
        "atol": tolerance,
        "rtol": tolerance,
        "merge_max_absolute_difference": (a - b).abs().max().item(),
    }
    for name, model in (("adapter", adapted), ("merged", merged)):
        cache = KVCache(model)
        cut = ids.shape[1] // 2 + 1
        cached = torch.cat([cache.append(ids[:, :cut]), cache.append(ids[:, cut:])], dim=1)
        with torch.inference_mode():
            expected = model(ids).logits
        torch.testing.assert_close(cached, expected, atol=cache_tolerance, rtol=cache_tolerance)
        capacity[f"{name}_cache_max_absolute_difference"] = (cached - expected).abs().max().item()
    if state_identity(base.config, base.state_dict()) != before:
        raise ValueError("Caller base changed")
    return {
        "device": device,
        "artifact_files_verified": files,
        "frozen_base_and_caller_base_unchanged": True,
        "independent_target_exposures": exposure,
        "repeated_merge_weights_exact": True,
        "validation_merge": validation_result,
        "synthetic_capacity": capacity,
        "test_payloads_opened": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--base", type=Path, default=Path("outputs/phase-5-english-pilot/step-00003000/model")
    )
    parser.add_argument(
        "--tokenizer-dir", type=Path, default=Path("artifacts/tokenizers/english-bpe-v1")
    )
    parser.add_argument(
        "--conversations", type=Path, default=Path("data/processed/english-instructions-v1")
    )
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Verification output already exists")
    result = verify(args.run_dir, args.base, args.tokenizer_dir, args.conversations, args.device)
    args.output.write_bytes(canonical_json(result))
    print(json.dumps(result, indent=2))
