"""Read historical training inputs only; audit fixed development evaluation overlap."""

import argparse
import re
from collections import defaultdict
from pathlib import Path

from latos.evaluation.inputs import external_cases, read_json, write_json
from latos.model.storage import file_hash
from latos.tokenization.corpus import read_split


def shingles(text, width=13):
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[i : i + width]) for i in range(len(words) - width + 1)}


def audit(root):
    texts, identity = read_split(
        root / "data/manifests/english-books-v1.json",
        root / "data/processed/english-books-v1",
        "train",
    )
    instructions = root / "data/processed/english-instructions-v1/train.jsonl"
    import json

    records = [json.loads(line) for line in instructions.read_text().splitlines()]
    for record in records:
        texts.extend(m["content"] for m in record["messages"])
    index = defaultdict(set)
    for i, text in enumerate(texts):
        for part in shingles(text):
            index[part].add(i)
    suite = read_json(root / "configs/evaluation/suite-v1.json")
    queries = [(c["id"], c["prompt"]) for c in suite["instructions"] + suite["generation"]]
    for name, cases in external_cases(
        root / "configs/evaluation/external-v1.json", root / "data/cache/evaluation-v1"
    ).items():
        queries.extend(
            (name + ":" + c["id"], c["question"] + " " + " ".join(c["choices"])) for c in cases
        )
    findings = []
    exact = {" ".join(re.findall(r"\w+", text.casefold())) for text in texts}
    for case_id, text in queries:
        matches = set().union(*(index.get(s, set()) for s in shingles(text)))
        normalized = " ".join(re.findall(r"\w+", text.casefold()))
        if matches or normalized in exact:
            findings.append(
                {
                    "id": case_id,
                    "matching_training_records": sorted(matches),
                    "exact": normalized in exact,
                }
            )
    return {
        "schema_version": 1,
        "scope": (
            "historical book train and instruction train "
            "vs fixed development prompts/questions/choices"
        ),
        "algorithm": "casefold word tokens; exact full text and any contiguous 13-word overlap",
        "training_identity": identity,
        "instruction_train_sha256": file_hash(instructions),
        "queries": len(queries),
        "short_queries_without_13gram": sum(not shingles(t) for _, t in queries),
        "findings": findings,
        "reserved_payload_access": False,
        "limitations": (
            "No semantic absence claim; short common answers not evidence of contamination. "
            "Public benchmark exposure in future corpora needs a fresh source/near-duplicate "
            "audit. Final suites not parsed here."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Audit output already exists")
    write_json(args.output, audit(Path(__file__).resolve().parents[2]))
