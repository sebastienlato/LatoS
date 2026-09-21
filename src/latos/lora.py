"""Original, float32 attention LoRA with identity-bound adapter-only snapshots."""

import copy
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from latos.data.manifest import canonical_json
from latos.model.network import LatoModel, create_model
from latos.model.storage import file_hash


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    seed: int = 91

    def __post_init__(self):
        if type(self.rank) is not int or not 1 <= self.rank <= 64:
            raise ValueError("LoRA rank must be an integer in [1, 64]")
        if (
            type(self.alpha) not in (int, float)
            or not math.isfinite(self.alpha)
            or not (0 < self.alpha <= 256)
        ):
            raise ValueError("LoRA alpha must be finite in (0, 256]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("LoRA seed must be an integer in [0, 2**63)")


def state_identity(config, tensors: dict[str, torch.Tensor]) -> str:
    """Canonical config, sorted names/shapes and little-endian float32 tensor bytes."""
    digest = hashlib.sha256(canonical_json(config.to_dict()))
    for name, tensor in sorted(tensors.items()):
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item():
            raise ValueError("LoRA requires finite float32 weights")
        digest.update(canonical_json([name, list(tensor.shape)]))
        digest.update(tensor.detach().cpu().numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, config: LoRAConfig, generator: torch.Generator):
        super().__init__()
        self.base = base
        self.scale = config.alpha / config.rank
        self.a = nn.Parameter(
            (
                torch.randn(
                    config.rank,
                    base.in_features,
                    generator=generator,
                    device="cpu",
                    dtype=torch.float32,
                )
                * 0.02
            ).to(device=base.weight.device, dtype=torch.float32)
        )
        self.b = nn.Parameter(
            torch.zeros(
                base.out_features, config.rank, device=base.weight.device, dtype=torch.float32
            )
        )

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


class LoRAModel(LatoModel):
    """Owns a separate frozen base copy; fused QKV and attention output only.

    No dropout, bias, embeddings, norms or feed-forward adapters. Q/K/V share A.
    The ordinary model forward and KV-cache paths both call the adapted projections.
    """

    def __init__(self, base: LatoModel, config: LoRAConfig | None = None):
        config = config or LoRAConfig()
        if type(base) is not LatoModel or config.rank > base.config.d_model:
            raise ValueError("LoRA needs a dense base and rank no larger than its width")
        nn.Module.__init__(self)
        self.config = base.config
        self.lora_config = config
        self.base_identity = state_identity(base.config, base.state_dict())
        cloned = copy.deepcopy(base)
        self.embedding, self.blocks, self.final_norm = (
            cloned.embedding,
            cloned.blocks,
            cloned.final_norm,
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        for block in self.blocks:
            for name in ("qkv", "output"):
                setattr(
                    block.attention,
                    name,
                    LoRALinear(getattr(block.attention, name), config, generator),
                )
        self.eval()

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {name: p for name, p in self.named_parameters() if name.endswith((".a", ".b"))}

    def base_state(self) -> dict[str, torch.Tensor]:
        return {
            name.replace(".base.weight", ".weight"): tensor
            for name, tensor in self.state_dict().items()
            if not name.endswith((".a", ".b"))
        }

    def validate_adapter(self):
        expected = set()
        for index, block in enumerate(self.blocks):
            for name, width in (("qkv", 3 * self.config.d_model), ("output", self.config.d_model)):
                layer = getattr(block.attention, name)
                prefix = f"blocks.{index}.attention.{name}"
                expected.update((f"{prefix}.a", f"{prefix}.b"))
                if not isinstance(layer, LoRALinear) or (
                    layer.a.shape != (self.lora_config.rank, self.config.d_model)
                    or layer.b.shape != (width, self.lora_config.rank)
                    or layer.scale != self.lora_config.alpha / self.lora_config.rank
                ):
                    raise ValueError("Invalid LoRA projection structure")
        if set(self.adapter_state()) != expected:
            raise ValueError("Invalid adapter parameter names")
        for name, p in self.named_parameters():
            if p.requires_grad != (name in expected) or (
                name not in expected and p.grad is not None
            ):
                raise ValueError("LoRA base must be frozen with no gradients; only adapters train")
            if p.dtype != torch.float32 or not torch.isfinite(p).all().item():
                raise ValueError("LoRA requires finite float32 parameters")
        if state_identity(self.config, self.base_state()) != self.base_identity:
            raise ValueError("Frozen base identity changed")


def merge_adapter(model: LoRAModel) -> LatoModel:
    """Return an independent dense model; never edit or unmerge the source in place."""
    model.validate_adapter()
    tensors = {name: tensor.detach().cpu().clone() for name, tensor in model.base_state().items()}
    for name, layer in model.named_modules():
        if isinstance(layer, LoRALinear):
            tensors[f"{name}.weight"] += (
                layer.b.detach().cpu() @ layer.a.detach().cpu()
            ) * layer.scale
    state_identity(model.config, tensors)  # Reject overflow before constructing a dense model.
    merged = create_model(model.config)
    merged.load_state_dict(tensors, strict=True)
    return merged.eval()


def save_adapter(model: LoRAModel, directory: Path) -> dict:
    if directory.exists():
        raise ValueError("Adapter output already exists; choose a new destination")
    model.validate_adapter()
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.adapter_state().items()}
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".latos-adapter-", dir=directory.parent))
    try:
        path = staging / "adapter.safetensors"
        save_file(tensors, str(path))
        metadata = {
            "schema_version": 1,
            "kind": "latos-attention-lora-v1",
            "config": model.config.to_dict(),
            "lora": asdict(model.lora_config),
            "base_state_sha256": model.base_identity,
            "weights_sha256": file_hash(path),
            "weights_bytes": path.stat().st_size,
            "training_state": "adapter only; fresh optimizer required; no resume state",
        }
        (staging / "metadata.json").write_bytes(canonical_json(metadata))
        restored = load_file(str(path))
        if set(restored) != set(tensors) or any(
            not torch.equal(restored[k], v) for k, v in tensors.items()
        ):
            raise ValueError("Adapter save round trip failed")
        staging.rename(directory)
        return metadata
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_adapter(base: LatoModel, directory: Path) -> LoRAModel:
    meta_path = directory / "metadata.json"
    if meta_path.stat().st_size > 16_384:
        raise ValueError("Adapter metadata exceeds budget")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["schema_version"] != 1 or meta["kind"] != "latos-attention-lora-v1":
        raise ValueError("Unsupported adapter snapshot")
    if meta["config"] != base.config.to_dict() or meta["base_state_sha256"] != state_identity(
        base.config, base.state_dict()
    ):
        raise ValueError("Adapter base or tokenizer identity mismatch")
    config = LoRAConfig(**meta["lora"])
    # Derive file budget before loading tensors or allocating adapter factors.
    parameters = base.config.n_layers * config.rank * base.config.d_model * 6
    path = directory / "adapter.safetensors"
    if (
        path.stat().st_size != meta["weights_bytes"]
        or path.stat().st_size > parameters * 4 + 100_000
        or file_hash(path) != meta["weights_sha256"]
    ):
        raise ValueError("Adapter tensor size or checksum mismatch")
    model = LoRAModel(base, config)
    tensors = load_file(str(path), device="cpu")
    expected = model.adapter_state()
    if set(tensors) != set(expected):
        raise ValueError("Invalid adapter tensor names")
    for name, tensor in tensors.items():
        if (
            tensor.shape != expected[name].shape
            or tensor.dtype != torch.float32
            or not (torch.isfinite(tensor).all().item())
        ):
            raise ValueError(f"Invalid adapter tensor: {name}")
    with torch.no_grad():
        for name, tensor in tensors.items():
            expected[name].copy_(tensor)
    model.validate_adapter()
    return model.eval()
