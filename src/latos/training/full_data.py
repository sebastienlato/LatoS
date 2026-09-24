"""Compact, verified one-document-at-a-time Data 2.0 windows for a full pass."""

import hashlib
import json
from itertools import groupby
from pathlib import Path

import numpy as np

from latos.data.manifest import canonical_json, sha256
from latos.data2.tokenizer import read_records
from latos.model.storage import file_hash
from latos.tokenization import LatoTokenizer
from latos.training.data2 import POLICY, coalesce_document


class CompactWindows:
    """Read-only little-endian disk arrays; materialize only requested windows."""

    def __init__(self, directory: Path):
        self.tokens = np.memmap(directory / "tokens.u32", dtype="<u4", mode="r")
        self.offsets = np.memmap(directory / "offsets.u64", dtype="<u8", mode="r")

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        begin, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return tuple(self.tokens[begin:end].tolist())


class FullDataset:
    """Training dataset interface with verified cached identities and O(1) counts."""

    target_masks = None

    def __init__(self, directory: Path, expected: dict, tokenizer_hash: str, split: str):
        if split not in ("train", "development"):
            raise ValueError("Full data cache accepts no reserved split")
        meta = json.loads((directory / "cache.json").read_text())
        if meta["split"] != split or meta["tokenizer_sha256"] != tokenizer_hash:
            raise ValueError("Cache split/tokenizer mismatch")
        if meta["policy"] != POLICY or meta["sequence_length"] != 512:
            raise ValueError("Cache layout policy mismatch")
        for name in ("tokens.u32", "offsets.u64"):
            if file_hash(directory / name) != meta["files"][name]:
                raise ValueError("Cache payload checksum mismatch")
        for key, value in expected.items():
            if meta["accounting"].get(key) != value:
                raise ValueError(f"Full-corpus accounting mismatch: {key}")
        self.windows = CompactWindows(directory)
        offsets = self.windows.offsets
        lengths = np.diff(offsets.astype(np.int64))
        if (
            int(offsets[0]) != 0
            or int(offsets[-1]) != len(self.windows.tokens)
            or not np.all((lengths >= 2) & (lengths <= 512))
            or len(lengths) != expected["windows"]
            or int((lengths - 1).sum()) != expected["targets_with_EOS"]
        ):
            raise ValueError("Cache offsets/targets invalid")
        self.sequence_length = 512
        self.vocab_size = meta["vocab_size"]
        self.tokenizer_sha256 = tokenizer_hash
        self.split = "train" if split == "train" else "validation"
        self.source_sha256 = expected["source_file_sha256"]
        # Verify the semantic sequence, not just a self-reported binary checksum.
        digest = hashlib.sha256()
        for window in self.windows:
            if any(t < 1 or t >= self.vocab_size or t == 3 for t in window):
                raise ValueError("Invalid cache token")
            digest.update(canonical_json(window))
        if digest.hexdigest() != expected["window_sequence_sha256"]:
            raise ValueError("Cache window sequence mismatch")
        self.identity = {
            "schema_version": 1,
            "boundary_policy": POLICY,
            "sequence_length": 512,
            "vocab_size": self.vocab_size,
            "tokenizer_sha256": tokenizer_hash,
            "split": self.split,
            "source_sha256": self.source_sha256,
            "windows_sha256": digest.hexdigest(),
            "windows": len(self.windows),
            "targets": expected["targets_with_EOS"],
        }

    def target_count(self, index):
        return int(self.windows.offsets[index + 1]) - int(self.windows.offsets[index]) - 1


def build_cache(corpus: Path, tokenizer: Path, split: str, output: Path, expected: dict):
    """Create once; incomplete construction remains evidence, never a usable cache."""
    if split not in ("train", "development"):
        raise ValueError("Reserved data is forbidden")
    output.mkdir(parents=True, exist_ok=False)
    codec = LatoTokenizer.load(tokenizer)
    seen, groups, identities = set(), set(), []
    content = byte_count = record_count = targets = windows = offset = 0
    digest = hashlib.sha256()
    with (
        (output / "tokens.u32").open("xb") as tokens,
        (output / "offsets.u64").open("xb") as offsets,
    ):
        offsets.write(np.asarray([0], dtype="<u8").tobytes())
        for doc, iterable in groupby(
            read_records(corpus, "pretrain", split), lambda r: r["document_id"]
        ):
            if doc in seen:
                raise ValueError("Noncontiguous document")
            seen.add(doc)
            rows = list(iterable)
            record_count += len(rows)
            groups.add(rows[0]["group_id"])
            text = coalesce_document(rows)
            ids = codec.encode(text, add_bos=True, add_eos=True)
            if len(text.encode()) > 262144 or len(ids) > 65536:
                raise ValueError("Document exceeds the accepted bounds; no omission permitted")
            byte_count += len(text.encode())
            content += len(ids) - 2
            identities.append({"id": doc, "text_sha256": sha256(text.encode())})
            for start in range(0, len(ids) - 1, 511):
                window = ids[start : start + 512]
                tokens.write(np.asarray(window, dtype="<u4").tobytes())
                offset += len(window)
                offsets.write(np.asarray([offset], dtype="<u8").tobytes())
                digest.update(canonical_json(window))
                targets += len(window) - 1
                windows += 1
    accounting = {
        "documents": len(seen),
        "records": record_count,
        "groups": len(groups),
        "content_bytes": byte_count,
        "content_token_positions": content,
        "targets_with_EOS": targets,
        "windows": windows,
        "window_sequence_sha256": digest.hexdigest(),
        "document_identity_sha256": sha256(canonical_json(identities)),
        "source_file_sha256": file_hash(corpus / f"pretrain-{split}.jsonl"),
    }
    for key, value in expected.items():
        if accounting.get(key) != value:
            raise ValueError(f"Full-corpus accounting mismatch: {key}")
    meta = {
        "policy": POLICY,
        "split": split,
        "sequence_length": 512,
        "vocab_size": codec.vocab_size,
        "tokenizer_sha256": file_hash(tokenizer / "tokenizer.json"),
        "accounting": accounting,
        "groups": sorted(groups),
        "files": {name: file_hash(output / name) for name in ("tokens.u32", "offsets.u64")},
    }
    (output / "cache.json").write_bytes(canonical_json(meta))
    return FullDataset(output, expected, meta["tokenizer_sha256"], split)


def expected_accounting(layout: dict, split: str) -> dict:
    keys = (
        "documents",
        "records",
        "groups",
        "content_bytes",
        "content_token_positions",
        "targets_with_EOS",
        "windows",
        "window_sequence_sha256",
        "document_identity_sha256",
        "source_file_sha256",
    )
    return {key: layout["splits"][split][key] for key in keys}
