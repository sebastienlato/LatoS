"""Bounded float32 training configuration; schedule is indexed by optimizer update."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    schema_version: int = 1
    sequence_length: int = 64
    batch_size: int = 2
    accumulation_steps: int = 2
    max_steps: int = 100
    warmup_steps: int = 5
    learning_rate: float = 0.003
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    seed: int = 17

    def __post_init__(self):
        for key, low, high in (
            ("schema_version", 1, 1),
            ("sequence_length", 2, 4096),
            ("batch_size", 1, 256),
            ("accumulation_steps", 1, 256),
            ("max_steps", 1, 1_000_000),
            ("warmup_steps", 0, self.max_steps - 1),
            ("seed", 0, 2**63 - 1),
        ):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Training {key} must be an integer in [{low}, {high}]")
        for key, low, high in (
            ("learning_rate", 1e-10, 1.0),
            ("min_lr_ratio", 0.0, 1.0),
            ("weight_decay", 0.0, 1.0),
            ("beta1", 0.0, 0.9999),
            ("beta2", 0.0, 0.9999),
            ("eps", 1e-12, 1e-2),
            ("max_grad_norm", 1e-8, 1e6),
        ):
            value = getattr(self, key)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"Invalid training {key}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: Path) -> TrainingConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def learning_rate_at(self, step: int) -> float:
        """One-based update: linear warmup, then cosine to the final minimum."""
        if type(step) is not int or not 1 <= step <= self.max_steps:
            raise ValueError("Schedule step outside the configured run")
        if step <= self.warmup_steps:
            return self.learning_rate * step / self.warmup_steps
        # With no warmup, update 1 starts at peak; a one-update run uses peak.
        start = max(1, self.warmup_steps)
        progress = (step - start) / max(1, self.max_steps - start)
        factor = (
            self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        )
        return self.learning_rate * factor
