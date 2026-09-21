"""Independent high-precision merge equation on the fixed first validation example."""

import argparse
import json
from pathlib import Path

import torch

from latos.data.manifest import canonical_json
from latos.instruction import prepare_conversations
from latos.lora import LoRALinear, load_adapter
from latos.model import create_model
from latos.model.storage import bind_tokenizer, load_model
from latos.training.data import collate


def reference(run_dir):
    torch.set_num_threads(1)
    base = load_model(Path("outputs/phase-5-english-pilot/step-00003000/model"))
    codec = bind_tokenizer(base.config, Path("artifacts/tokenizers/english-bpe-v1"))
    dataset = prepare_conversations(
        Path("data/processed/english-instructions-v1"),
        codec,
        base.config.tokenizer_sha256,
        256,
        "validation",
    )
    ids, _, _ = collate(dataset, [0], "cpu")
    adapter32 = load_adapter(base, run_dir / "lora/adapter")
    merged32 = load_model(run_dir / "lora/merged")
    adapter64 = load_adapter(base, run_dir / "lora/adapter").double()
    merged64 = create_model(base.config).double()
    state = {name: tensor.clone() for name, tensor in adapter64.base_state().items()}
    for name, layer in adapter64.named_modules():
        if isinstance(layer, LoRALinear):
            state[f"{name}.weight"] += (layer.b @ layer.a) * layer.scale
    merged64.load_state_dict(state)
    with torch.inference_mode():
        reference = adapter64(ids).logits
        dense = merged64(ids).logits
        torch.testing.assert_close(reference, dense, atol=1e-10, rtol=1e-10)
        a, b = adapter32(ids).logits.double(), merged32(ids).logits.double()
        torch.testing.assert_close(a, reference, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(b, reference, atol=1e-4, rtol=1e-4)
    return {
        "scope": "first fixed instruction validation window; CPU diagnostic only",
        "logits": reference.numel(),
        "float64_atol": 1e-10,
        "float64_rtol": 1e-10,
        "float64_merge_max_absolute_difference": (reference - dense).abs().max().item(),
        "adapter_float32_reference_max_absolute_difference": (a - reference).abs().max().item(),
        "merged_float32_reference_max_absolute_difference": (b - reference).abs().max().item(),
        "interpretation": "float64 equation agrees; float32 operation ordering introduces rounding",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Reference output already exists")
    result = reference(args.run_dir)
    args.output.write_bytes(canonical_json(result))
    print(json.dumps(result, indent=2))
