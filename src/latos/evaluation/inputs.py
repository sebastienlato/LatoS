"""Strict, pinned evaluation inputs; acquisition is separate from model evaluation."""

import json
import urllib.request
from pathlib import Path

from latos.data.manifest import canonical_json, sha256
from latos.model.storage import file_hash


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid
    )


def write_json(path, value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
    Path(path).write_text(data + "\n", encoding="utf-8")


def acquire(manifest_path, destination):
    """Only the two pinned development files; no train/test corpus download."""
    manifest = read_json(manifest_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for entry in manifest["datasets"]:
        path = destination / entry["file"]
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError("Unsafe dataset filename")
        if path.exists():
            if file_hash(path) != entry["sha256"]:
                raise ValueError("Existing evaluation data checksum mismatch")
            continue
        if not entry["url"].startswith("https://huggingface.co/datasets/allenai/ai2_arc/resolve/"):
            raise ValueError("Unexpected evaluation source")
        with urllib.request.urlopen(entry["url"], timeout=60) as response:
            if not response.url.startswith("https://"):
                raise ValueError("Evaluation download downgraded HTTPS")
            data = response.read(2_000_001)
        if len(data) != entry["bytes"] or sha256(data) != entry["sha256"]:
            raise ValueError("Evaluation download size/checksum mismatch")
        with path.open("xb") as stream:
            stream.write(data)
    return manifest


def external_cases(manifest_path, directory):
    # Optional dependency needed only for reading the evaluation-only Parquet files.
    import pyarrow.parquet as pq

    manifest = read_json(manifest_path)
    if manifest["schema_version"] != 1 or not manifest["provenance"] or not manifest["license"]:
        raise ValueError("Missing evaluation provenance")
    result = {}
    for entry in manifest["datasets"]:
        if (
            entry["split"] != "validation"
            or len(entry["revision"]) != 40
            or Path(entry["file"]).name != entry["file"]
        ):
            raise ValueError("Only pinned validation files are supported")
        path = Path(directory) / entry["file"]
        if path.stat().st_size != entry["bytes"] or file_hash(path) != entry["sha256"]:
            raise ValueError("Evaluation input checksum mismatch")
        records = pq.read_table(path).to_pylist()
        if len(records) != entry["records"]:
            raise ValueError("Evaluation record count mismatch")
        cases, ids = [], set()
        for row in records:
            choices, labels = row["choices"]["text"], row["choices"]["label"]
            if (
                not isinstance(row["id"], str)
                or not row["id"]
                or row["id"] in ids
                or not isinstance(row["question"], str)
                or not row["question"].strip()
                or not 2 <= len(choices) <= 5
                or len(labels) != len(choices)
                or len(set(labels)) != len(labels)
                or row["answerKey"] not in labels
                or any(not isinstance(c, str) or not c.strip() for c in choices)
            ):
                raise ValueError("Malformed external question")
            ids.add(row["id"])
            cases.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "choices": choices,
                    "gold": labels.index(row["answerKey"]),
                    "content_sha256": sha256(canonical_json(row)),
                }
            )
        result[entry["name"]] = cases
    return result


def validate_suite(suite):
    if (
        suite.get("schema_version") != 1
        or not suite.get("provenance")
        or not suite.get("instructions")
        or not suite.get("generation")
    ):
        raise ValueError("Invalid suite schema or missing provenance/cases")
    seen = set()
    for name in ("instructions", "generation"):
        for case in suite[name]:
            for key in ("id", "category", "prompt"):
                if not isinstance(case.get(key), str) or not case[key].strip():
                    raise ValueError("Missing suite case identity, category or prompt")
            if case["id"] in seen:
                raise ValueError("Duplicate suite case ID")
            seen.add(case["id"])
            if name == "instructions":
                if case.get("scorer") not in ("exact", "json") or not case.get("family"):
                    raise ValueError("Unknown instruction scorer or family")
                if "expected" not in case or (
                    case["scorer"] == "exact" and not isinstance(case["expected"], str)
                ):
                    raise ValueError("Invalid instruction gold answer")
