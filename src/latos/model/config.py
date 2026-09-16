"""Versioned, bounded model dimensions and a required tokenizer identity."""

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    schema_version: int
    vocab_size: int
    context_length: int
    d_model: int
    n_heads: int
    n_layers: int
    ffn_dim: int
    rope_theta: float
    norm_eps: float
    tokenizer_sha256: str

    def __post_init__(self):
        bounds = {
            "schema_version": (1, 1),
            "vocab_size": (260, 32768),
            "context_length": (2, 4096),
            "d_model": (8, 1024),
            "n_heads": (1, 32),
            "n_layers": (1, 24),
            "ffn_dim": (8, 4096),
        }
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Model {key} must be an integer in [{low}, {high}]")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("Model width must divide into even-sized attention heads")
        for key, low, high in (("rope_theta", 1.0, 1e8), ("norm_eps", 1e-12, 1e-2)):
            value = getattr(self, key)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"Invalid model {key}")
        if not isinstance(self.tokenizer_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.tokenizer_sha256
        ):
            raise ValueError("A tokenizer SHA-256 identity is required")
        if self.parameter_count > 100_000_000:
            raise ValueError("Model exceeds the 100-million-parameter configuration budget")

    @property
    def parameter_count(self) -> int:
        d, f, layers = self.d_model, self.ffn_dim, self.n_layers
        return self.vocab_size * d + layers * (4 * d * d + 3 * d * f + 2 * d) + d

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: Path) -> ModelConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))
