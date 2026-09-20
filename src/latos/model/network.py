"""Dense causal decoder expressed directly with PyTorch tensor operations."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from latos.model.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work = x if x.dtype == torch.float64 else x.float()
        normalized = work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def rotary(
    q: torch.Tensor, k: torch.Tensor, theta: float, offset: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate adjacent feature pairs at absolute positions starting at offset."""
    if type(offset) is not int or offset < 0:
        raise ValueError("Rotary offset must be a nonnegative integer")
    if q.shape != k.shape or q.ndim != 4 or q.shape[-1] % 2:
        raise ValueError("Rotary inputs need matching [batch, heads, time, even head_dim] shapes")
    width = q.shape[-1]
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    frequencies = theta ** (-torch.arange(0, width, 2, device=q.device, dtype=dtype) / width)
    positions = torch.arange(offset, offset + q.shape[-2], device=q.device, dtype=dtype)
    angles = positions[:, None] * frequencies[None, :]
    cosine, sine = angles.cos().to(q.dtype), angles.sin().to(q.dtype)

    def rotate(x):
        first, second = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (first * cosine - second * sine, first * sine + second * cosine), dim=-1
        ).flatten(-2)

    return rotate(q), rotate(k)


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.theta = config.rope_theta
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False, dtype=torch.float32)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        projected = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = projected.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = rotary(q, k, self.theta)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return self.output(attended.transpose(1, 2).reshape(batch, length, width))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.ffn_dim, bias=False, dtype=torch.float32)
        self.up = nn.Linear(config.d_model, config.ffn_dim, bias=False, dtype=torch.float32)
        self.down = nn.Linear(config.ffn_dim, config.d_model, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


def causal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross entropy; labels are unshifted, -100 masks target positions."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or logits.shape[1] < 2:
        raise ValueError("Loss needs [batch, time>=2, vocab] logits and matching unshifted labels")
    if labels.dtype != torch.long or labels.device != logits.device:
        raise ValueError("Labels must be int64 on the logits device")
    invalid = (labels != -100) & ((labels < 0) | (labels >= logits.shape[-1]))
    if invalid.any().item():
        raise ValueError("Labels contain an invalid target ID")
    targets = labels[:, 1:]
    if not (targets != -100).any().item():
        raise ValueError("At least one next-token target must be unmasked")
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
    )


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


class LatoModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, dtype=torch.float32)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.output.weight, std=residual_std)
            nn.init.normal_(block.ffn.down.weight, std=residual_std)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> ModelOutput:
        if (
            input_ids.ndim != 2
            or input_ids.shape[0] < 1
            or not 1 <= input_ids.shape[1] <= self.config.context_length
        ):
            raise ValueError("Input needs a nonempty [batch, time] shape within the context limit")
        if input_ids.dtype != torch.long or input_ids.device != self.embedding.weight.device:
            raise ValueError("Input IDs must be int64 on the model device")
        if ((input_ids < 0) | (input_ids >= self.config.vocab_size)).any().item():
            raise ValueError("Input contains an invalid token ID")
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        # Functional projection ties output weights to the one embedding parameter.
        logits = F.linear(self.final_norm(x), self.embedding.weight)
        return ModelOutput(logits, None if labels is None else causal_loss(logits, labels))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def create_model(config: ModelConfig, seed: int = 0) -> LatoModel:
    """Initialize on CPU reproducibly without consuming the caller's CPU RNG state."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Seed must be an integer in [0, 2**63)")
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        return LatoModel(config)
