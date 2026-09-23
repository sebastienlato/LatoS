"""Construction acceptance only: these gates say nothing about learned capability."""

import argparse
import json
from pathlib import Path

from latos.data2.acquire import file_hash, write_json


def check(corpus: Path, audit_path: Path, tokenizer_dir: Path):
    report = json.loads((corpus / "report.json").read_text())
    audit = json.loads(audit_path.read_text())
    selection = json.loads((tokenizer_dir / "selection.json").read_text())
    accounting = json.loads((tokenizer_dir / "training-accounting.json").read_text())
    for name, item in report["files"].items():
        if file_hash(corpus / name) != item["sha256"]:
            raise ValueError("Prepared input changed since construction")
    categories = (
        "brainstorming",
        "classification",
        "closed_qa",
        "creative_writing",
        "general_qa",
        "information_extraction",
        "open_qa",
        "summarization",
    )
    train = report["splits"]["instruction/train"]
    gates = {
        "pretraining_at_least_100MB": sum(
            report["splits"]["pretrain/" + s]["utf8_bytes"]
            for s in ("train", "development", "reserved")
        )
        >= 100_000_000,
        "training_conversations_at_least_10000": train["documents"] >= 10_000,
        "two_instruction_origins": train.get("source/dolly", 0) > 0
        and train.get("source/oasst", 0) > 0,
        "all_eight_observed_Dolly_categories_in_training": all(
            train.get("category/" + c, 0) > 0 for c in categories
        ),
        "actual_multi_turn": train.get("multi_turn", 0) > 0,
        "complete_output_audit": audit["passed"]
        and audit["report_sha256"] == file_hash(corpus / "report.json"),
        "no_remaining_output_duplicate_pairs": not any(audit["output_near_pairs"].values()),
        "tokenizer_contracts": all(
            v["contract_checks"]["passed"] for v in selection["development"].values()
        ),
        "positive_training_token_accounting": accounting["stats"]["pretrain"]["content_tokens"] > 0
        and accounting["stats"]["instruction"]["assistant_targets"] > 0,
        "no_reserved_tokenizer_analysis": all(
            v["reserved_text_access"] is False
            for v in [accounting, *selection["development"].values()]
        ),
    }
    return {
        "schema_version": 1,
        "passed": all(gates.values()),
        "gates": gates,
        "corpus_report_sha256": file_hash(corpus / "report.json"),
        "audit_sha256": file_hash(audit_path),
        "tokenizer_selection_sha256": file_hash(tokenizer_dir / "selection.json"),
        "scope": (
            "Construction minima only; quality inspection, rights/review "
            "and fixed future learned-quality gates remain separate."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--tokenizers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Acceptance output already exists")
    result = check(args.corpus, args.audit, args.tokenizers)
    write_json(args.output, result)
    if not result["passed"]:
        raise ValueError("Construction gate failed")


if __name__ == "__main__":
    main()
