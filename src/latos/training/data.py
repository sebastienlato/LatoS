"""Isolated record windows, right padding, and a resumable CPU shuffle stream."""

from dataclasses import dataclass
from pathlib import Path

import torch

from latos.data.manifest import canonical_json, sha256
from latos.tokenization import LatoTokenizer
from latos.tokenization.corpus import read_split


@dataclass(frozen=True)
class TokenDataset:
    windows: tuple[tuple[int, ...], ...]
    sequence_length: int
    vocab_size: int
    tokenizer_sha256: str
    split: str
    source_sha256: str
    target_masks: tuple[tuple[bool, ...], ...] | None = None

    def __post_init__(self):
        if self.split not in ("train", "validation"):
            raise ValueError("Training engine accepts only train or validation; test is reserved")
        if type(self.sequence_length) is not int or not 2 <= self.sequence_length <= 4096:
            raise ValueError("Invalid dataset sequence length")
        if type(self.vocab_size) is not int or not 260 <= self.vocab_size <= 32768:
            raise ValueError("Invalid dataset vocabulary size")
        for digest in (self.tokenizer_sha256, self.source_sha256):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Dataset identities must be SHA-256 digests")
        if not isinstance(self.windows, tuple) or not self.windows:
            raise ValueError("Dataset needs immutable nonempty windows")
        for window in self.windows:
            if not isinstance(window, tuple) or not 2 <= len(window) <= self.sequence_length:
                raise ValueError("Invalid training window length")
            if any(type(t) is not int or not 1 <= t < self.vocab_size or t == 3 for t in window):
                raise ValueError("Invalid training token or reserved padding/unknown ID")

        if self.target_masks is not None:
            if not isinstance(self.target_masks, tuple) or len(self.target_masks) != len(
                self.windows
            ):
                raise ValueError("Target masks must match windows")
            for row, mask in zip(self.windows, self.target_masks, strict=True):
                if (
                    not isinstance(mask, tuple)
                    or len(mask) != len(row)
                    or any(type(value) is not bool for value in mask)
                    or mask[0]
                    or not any(mask[1:])
                ):
                    raise ValueError("Invalid target mask or no usable next-token targets")

    def target_count(self, index: int) -> int:
        return (
            sum(self.target_masks[index])
            if self.target_masks is not None
            else len(self.windows[index]) - 1
        )

    @property
    def identity(self) -> dict:
        identity = {
            "schema_version": 1,
            "boundary_policy": "isolated-record-overlap-one-v1",
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "tokenizer_sha256": self.tokenizer_sha256,
            "split": self.split,
            "source_sha256": self.source_sha256,
            "windows_sha256": sha256(canonical_json(self.windows)),
            "windows": len(self.windows),
            "targets": sum(self.target_count(i) for i in range(len(self.windows))),
        }
        if self.target_masks is not None:
            identity["boundary_policy"] = "isolated-explicit-target-windows-v1"
            identity["objective"] = "explicit-assistant-targets-v1"
            identity["target_masks_sha256"] = sha256(canonical_json(self.target_masks))
        return identity


def prepare_dataset(
    manifest: Path,
    corpus: Path,
    codec: LatoTokenizer,
    tokenizer_sha256: str,
    sequence_length: int,
    split: str,
) -> TokenDataset:
    if split not in ("train", "validation"):
        raise ValueError("Training engine does not read the reserved test split")
    if type(sequence_length) is not int or not 2 <= sequence_length <= 4096:
        raise ValueError("Invalid dataset sequence length")
    texts, source = read_split(manifest, corpus, split)
    windows = []
    for text in texts:
        ids = codec.encode(text, add_bos=True, add_eos=True)
        # Each transition occurs once; overlap preserves the target at a chunk seam.
        for start in range(0, len(ids) - 1, sequence_length - 1):
            windows.append(tuple(ids[start : start + sequence_length]))
    return TokenDataset(
        tuple(windows),
        sequence_length,
        codec.vocab_size,
        tokenizer_sha256,
        split,
        sha256(canonical_json(source)),
    )


def collate(
    dataset: TokenDataset, indices: list[int], device: str
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not indices:
        raise ValueError("Cannot collate an empty batch")
    rows = [dataset.windows[i] for i in indices]
    inputs = torch.zeros((len(rows), max(map(len, rows))), dtype=torch.long)
    labels = torch.full_like(inputs, -100)
    for i, row in enumerate(rows):
        inputs[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        if dataset.target_masks is None:
            labels[i, 1 : len(row)] = inputs[i, 1 : len(row)]
        else:
            mask = torch.tensor(dataset.target_masks[indices[i]], dtype=torch.bool)
            labels[i, : len(row)][mask] = inputs[i, : len(row)][mask]
    return inputs.to(device), labels.to(device), sum(dataset.target_count(i) for i in indices)


class ShuffleStream:
    def __init__(self, size: int, seed: int):
        self.size = size
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.epoch = 0
        self.order = torch.randperm(size, generator=self.generator)
        self.cursor = 0

    def take(self, batch_size: int) -> list[int]:
        if self.cursor == self.size:
            self.epoch += 1
            self.order = torch.randperm(self.size, generator=self.generator)
            self.cursor = 0
        end = min(self.size, self.cursor + batch_size)
        indices = self.order[self.cursor : end].tolist()
        self.cursor = end
        return indices
