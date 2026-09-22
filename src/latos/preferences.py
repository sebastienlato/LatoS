"""Bounded DPO over deterministic preferences; shared chat serialization stays unchanged."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from latos.chat import CHAT_CONTRACT, format_chat
from latos.data.manifest import canonical_json, sha256
from latos.instruction import load_conversations
from latos.training.data import TokenDataset, collate


@dataclass(frozen=True)
class DPOConfig:
    steps: int = 100
    batch_size: int = 4
    sequence_length: int = 256
    seed: int = 101
    beta: float = 0.1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    clip_norm: float = 1.0

    def __post_init__(self):
        for name, low, high in (
            ("steps", 1, 1000),
            ("batch_size", 1, 32),
            ("sequence_length", 2, 4096),
            ("seed", 0, 2**63 - 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid DPO {name}")
        for name in ("beta", "learning_rate", "weight_decay", "clip_norm"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Invalid DPO {name}")
            if value < 0 or (name != "weight_decay" and value == 0):
                raise ValueError(f"Invalid DPO {name}")

    @classmethod
    def load(cls, path: Path):
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self):
        return asdict(self)


def preference_records(directory: Path, split: str) -> list[dict]:
    """Use only one allowed split; rotate to the next different same-family answer."""
    source = load_conversations(directory, split)
    records = []
    for index, record in enumerate(source):
        chosen = record["messages"][-1]["content"]
        candidates = source[index + 1 :] + source[:index]
        other = next(
            (
                r
                for r in candidates
                if r["family"] == record["family"] and r["messages"][-1]["content"] != chosen
            ),
            None,
        )
        if other is None:
            raise ValueError("Preference family needs two distinct answers")
        records.append(
            {
                "id": record["id"],
                "source_group": record["source_group"],
                "family": record["family"],
                "prompt": record["messages"][:-1],
                "chosen": chosen,
                "rejected": other["messages"][-1]["content"],
                "rejected_source_id": other["id"],
            }
        )
    return records


def prepare_preferences(records, codec, token_hash, length, split):
    """Only the final response and EOS are targets; prior assistant turns are context."""
    if split not in ("train", "validation"):
        raise ValueError("Preference test data is reserved")
    windows, masks = [], []
    for record in records:
        prompt, _ = format_chat(codec, record["prompt"], max_length=length, generation_prompt=True)
        if record["chosen"] == record["rejected"]:
            raise ValueError("Preference responses must differ")
        pair = []
        for side in ("chosen", "rejected"):
            ids, _ = format_chat(
                codec,
                record["prompt"] + [{"role": "assistant", "content": record[side]}],
                max_length=length,
            )
            if ids[: len(prompt)] != prompt:
                raise ValueError("Preference prompt prefix mismatch")
            mask = (False,) * len(prompt) + (True,) * (len(ids) - len(prompt))
            windows.append(ids)
            masks.append(mask)
            pair.append(ids)
        if pair[0] == pair[1]:
            raise ValueError("Preference responses tokenize identically")
    return TokenDataset(
        tuple(windows),
        length,
        codec.vocab_size,
        token_hash,
        split,
        sha256(canonical_json({"records": records, "chat": CHAT_CONTRACT})),
        tuple(masks),
    )


def sequence_logps(logits, labels):
    """Sum (not mean) next-token log probabilities over each response, including EOS."""
    if logits.ndim != 3 or logits.shape[1] < 2 or labels.shape != logits.shape[:2]:
        raise ValueError("Invalid response logits/labels shape")
    if labels.dtype != torch.long or labels.device != logits.device:
        raise ValueError("Labels must be int64 on the logits device")
    if not torch.isfinite(logits).all().item():
        raise ValueError("Nonfinite logits")
    targets = labels[:, 1:]
    mask = targets != -100
    if not mask.any(dim=1).all().item():
        raise ValueError("Every response needs a target")
    if ((targets[mask] < 0) | (targets[mask] >= logits.shape[-1])).any().item():
        raise ValueError("Invalid response target")
    safe = targets.masked_fill(~mask, 0)
    values = F.log_softmax(logits[:, :-1], dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return values.masked_fill(~mask, 0).sum(dim=-1)


def dpo_loss(policy, reference, beta):
    """Rows are [chosen, rejected]; reference never participates in autograd."""
    if type(beta) not in (int, float) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("DPO beta must be finite and positive")
    if policy.ndim != 2 or policy.shape[0] < 1 or policy.shape[1] != 2:
        raise ValueError("DPO needs nonempty chosen/rejected pairs")
    if reference.shape != policy.shape or reference.device != policy.device:
        raise ValueError("DPO reference shape/device mismatch")
    if not torch.isfinite(policy).all().item() or not torch.isfinite(reference).all().item():
        raise ValueError("Nonfinite preference log probabilities")
    relative = policy - reference.detach()
    margin = beta * (relative[:, 0] - relative[:, 1])
    if not torch.isfinite(margin).all().item():
        raise ValueError("Nonfinite DPO margin")
    return F.softplus(-margin).mean(), margin


def pair_logps(model, dataset, indices, device):
    rows = [row for index in indices for row in (2 * index, 2 * index + 1)]
    ids, labels, targets = collate(dataset, rows, device)
    return sequence_logps(model(ids).logits, labels).reshape(-1, 2), targets


@torch.no_grad()
def score_preferences(policy, reference, dataset, config, device):
    policy_mode, reference_mode = policy.training, reference.training
    policy.eval()
    reference.eval()
    scores, refs = [], []
    try:
        for start in range(0, len(dataset.windows) // 2, config.batch_size):
            indices = list(range(start, min(start + config.batch_size, len(dataset.windows) // 2)))
            scores.append(pair_logps(policy, dataset, indices, device)[0])
            refs.append(pair_logps(reference, dataset, indices, device)[0])
        p, r = torch.cat(scores), torch.cat(refs)
        loss, margin = dpo_loss(p, r, config.beta)
        raw = p[:, 0] - p[:, 1]
        return {
            "pairs": len(p),
            "loss": loss.item(),
            "mean_relative_margin": margin.mean().item(),
            "relative_wins": (margin > 0).sum().item(),
            "relative_ties": (margin == 0).sum().item(),
            "raw_chosen_wins": (raw > 0).sum().item(),
            "raw_ties": (raw == 0).sum().item(),
            "mean_chosen_logp": p[:, 0].mean().item(),
            "mean_rejected_logp": p[:, 1].mean().item(),
            "sequence_logps": p.cpu().tolist(),
            "reference_logps": r.cpu().tolist(),
        }
    finally:
        policy.train(policy_mode)
        reference.train(reference_mode)
