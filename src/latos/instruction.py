"""Verified conversation splits; reserved test payloads are never opened."""

import json
from pathlib import Path

from latos.chat import CHAT_CONTRACT, format_chat
from latos.data.manifest import canonical_json, sha256
from latos.tokenization import LatoTokenizer
from latos.training.data import TokenDataset


def load_conversations(directory: Path, split: str) -> list[dict]:
    if split not in ("train", "validation"):
        raise ValueError("Instruction test split is reserved")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or set(manifest["splits"]) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("Invalid instruction manifest")
    groups, ids = set(), set()
    for entry in manifest["splits"].values():
        if (
            not entry["source_groups"]
            or len(set(entry["source_groups"])) != len(entry["source_groups"])
            or len(set(entry["ids"])) != len(entry["ids"])
            or groups.intersection(entry["source_groups"])
            or ids.intersection(entry["ids"])
        ):
            raise ValueError("Conversation/source identities overlap across splits")
        groups.update(entry["source_groups"])
        ids.update(entry["ids"])
    entry = manifest["splits"][split]
    path = directory / f"{split}.jsonl"
    if path.stat().st_size > 10_000_000:
        raise ValueError("Instruction split exceeds size budget")
    data = path.read_bytes()
    if len(data) != entry["bytes"] or sha256(data) != entry["sha256"]:
        raise ValueError("Instruction split checksum mismatch")
    records = [json.loads(line) for line in data.splitlines()]
    if [r["id"] for r in records] != entry["ids"] or any(
        r["source_group"] not in entry["source_groups"] for r in records
    ):
        raise ValueError("Instruction records differ from manifest")
    if not records:
        raise ValueError("Empty instruction split")
    return records


def prepare_conversations(
    directory: Path,
    codec: LatoTokenizer,
    tokenizer_sha256: str,
    sequence_length: int,
    split: str,
) -> TokenDataset:
    records = load_conversations(directory, split)
    formatted = [format_chat(codec, r["messages"], max_length=sequence_length) for r in records]
    return TokenDataset(
        tuple(row[0] for row in formatted),
        sequence_length,
        codec.vocab_size,
        tokenizer_sha256,
        split,
        sha256(canonical_json({"records": records, "chat_contract": CHAT_CONTRACT})),
        tuple(row[1] for row in formatted),
    )
