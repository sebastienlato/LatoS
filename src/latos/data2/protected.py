"""Controlled contamination-only access: no examples exported and no model scoring."""

import json
import re
from collections import Counter
from pathlib import Path

from latos.data2.acquire import file_hash, read_manifest, verify
from latos.data2.lexical import LexicalIndex, digest, grams, normalized


class Guard:
    def __init__(self):
        self.exact = set()
        self.thirteen = set()
        self.near = LexicalIndex(0.8)
        self.counts = Counter()
        self.inputs = {}

    def add(self, text: str, kind: str):
        if not isinstance(text, str) or not text.strip():
            return
        self.counts[kind] += 1
        value = normalized(text)
        if not value or digest(value) in self.exact:
            return
        self.exact.add(digest(value))
        self.thirteen.update(grams(text, 13))
        self.near.add(grams(text))

    def match(self, text: str):
        if digest(normalized(text)) in self.exact:
            return "protected_exact"
        if self.thirteen.intersection(grams(text, 13)):
            return "protected_13word"
        if self.near.matches(grams(text)):
            return "protected_near"
        return None

    def summary(self):
        return {
            "controlled_audit_only": True,
            "model_scoring_or_final_override": False,
            "input_sha256": self.inputs,
            "texts_by_kind": dict(self.counts),
            "unique_exact_fingerprints": len(self.exact),
            "unique_13word_fingerprints": len(self.thirteen),
            "policy": (
                "docs/EVALUATION.md contamination policy; fingerprint comparison "
                "only; no example export"
            ),
        }


def build_guard(root: Path) -> Guard:
    import pyarrow.parquet as pq

    guard = Guard()

    def load(path):
        guard.inputs[path.relative_to(root).as_posix()] = file_hash(path)
        return json.loads(path.read_text(encoding="utf-8"))

    protocol = load(root / "configs/evaluation/protocol-v1.json")
    paths = [
        (
            "data/processed/english-books-v1/validation.jsonl",
            "lm_validation_sha256",
            "historical_lm_development",
        ),
        ("data/processed/english-books-v1/test.jsonl", "lm_test_sha256", "historical_lm_reserved"),
    ]
    for relative, key, kind in paths:
        path = root / relative
        if file_hash(path) != protocol[key]:
            raise ValueError("Protected LM identity mismatch")
        guard.inputs[relative] = file_hash(path)
        with path.open(encoding="utf-8") as f:
            for line in f:
                guard.add(json.loads(line)["text"], kind)
    for split in ("train", "validation", "test"):
        path = root / f"data/processed/english-instructions-v1/{split}.jsonl"
        guard.inputs[path.relative_to(root).as_posix()] = file_hash(path)
        if (
            split == "test"
            and file_hash(path)
            != "5e0fe6050dcd1df7e85ce46ebcd830ee6cec8b4841b9c1f8f270b110889395a7"
        ):
            raise ValueError("Protected instruction identity mismatch")
        with path.open(encoding="utf-8") as f:
            for line in f:
                for message in json.loads(line)["messages"]:
                    guard.add(message["content"], "historical_instructions_" + split)
    for filename, key in (
        ("suite-v1.json", "suite_sha256"),
        ("final-suite-v1.json", "final_suite_sha256"),
    ):
        path = root / "configs/evaluation" / filename
        if file_hash(path) != protocol[key]:
            raise ValueError("Protected suite identity mismatch")
        suite = load(path)
        for row in suite.get("instructions", []) + suite.get("generation", []):
            guard.add(row["prompt"], filename)
            if "expected" in row:
                value = row["expected"]
                guard.add(
                    value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                    filename,
                )
    manifest_path = root / "configs/data2/protected-arc-v1.json"
    guard.inputs[manifest_path.relative_to(root).as_posix()] = file_hash(manifest_path)
    for source in read_manifest(manifest_path)["sources"]:
        for item in source["files"]:
            path = root / "data/cache/data2-protected-arc-v1" / item["file"]
            verify(path, item)
            guard.inputs[path.relative_to(root).as_posix()] = item["sha256"]
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    guard.add(row["question"], "ARC_all_splits_question")
                    for choice in row["choices"]["text"]:
                        guard.add(choice, "ARC_all_splits_answer_choice")
    # Both retained retrospective panels, all actual generated text, no token scoring.
    for directory in ("phase-14-evaluation", "phase-14-evaluation-reviewed"):
        paths = sorted((root / "outputs" / directory).glob("*/*.json"))
        paths = [p for p in paths if p.name in ("generation.json", "instructions.json")]
        if len(paths) != 12:
            raise ValueError("Missing historical generated-output protection")
        for path in paths:
            for row in load(path):
                guard.add(row.get("text", ""), "historical_generated_output")
    return guard


def fragments(record: dict):
    if record["kind"] == "instruction":
        for message in record["messages"]:
            yield message["content"]
    else:
        yield record["text"]
        yield from (x.strip() for x in re.split(r"\n+", record["text"]) if x.strip())
