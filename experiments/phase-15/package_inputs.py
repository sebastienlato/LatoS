"""Assemble a local training-input directory from explicit reviewed artifact paths."""

import argparse
import json
import shutil
from pathlib import Path

from latos.data2.acquire import file_hash, read_manifest, verify, write_json
from latos.tokenization import LatoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizers", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.corpus / "report.json").read_text())
    selection = json.loads((args.tokenizers / "selection.json").read_text())
    audit = json.loads(args.audit.read_text())
    acceptance = json.loads(args.acceptance.read_text())
    if not acceptance["passed"] or not audit["passed"]:
        raise ValueError("Unaccepted inputs cannot be packaged")
    if acceptance["corpus_report_sha256"] != file_hash(args.corpus / "report.json") or acceptance[
        "audit_sha256"
    ] != file_hash(args.audit):
        raise ValueError("Acceptance identities differ")
    if acceptance["tokenizer_selection_sha256"] != file_hash(args.tokenizers / "selection.json"):
        raise ValueError("Tokenizer selection changed")
    codec_path = args.tokenizers / selection["selected"]
    if selection["selected"] == "historical8192":
        codec_path = Path("artifacts/tokenizers/english-bpe-v1")
    LatoTokenizer.load(
        codec_path,
        expected_sha256=selection["development"][selection["selected"]]["tokenizer_sha256"],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}

    def copy(source, relative):
        path = args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        if file_hash(source) != file_hash(path):
            raise ValueError("Package copy differs")
        files[relative] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}

    for kind in ("pretrain", "instruction"):
        for split in ("train", "development"):
            name = f"{kind}-{split}.jsonl"
            source = args.corpus / name
            if file_hash(source) != report["files"][name]["sha256"]:
                raise ValueError("Data changed after audit")
            copy(source, "corpus/" + name)
    copy(args.corpus / "report.json", "corpus/report.json")
    for name in ("tokenizer.json", "metadata.json"):
        copy(codec_path / name, "tokenizer/" + name)
    copy(args.audit, "evidence/audit.json")
    copy(args.acceptance, "evidence/acceptance.json")
    copy(args.tokenizers / "training-accounting.json", "evidence/training-accounting.json")
    copy(args.tokenizers / "selection.json", "evidence/tokenizer-selection.json")
    manifest_path = Path("configs/data2/sources-v1.json")
    copy(manifest_path, "notices/sources-v1.json")
    for source in read_manifest(manifest_path)["sources"]:
        for item in source["files"]:
            if item["purpose"] == "notice":
                path = Path("data/raw/data2-v1") / item["file"]
                verify(path, item)
                copy(path, "notices/" + item["file"])
    instructions = (
        "LatoS Phase 16 local input package\n\n"
        "Construction inputs only; no learned model or CUDA feasibility claim.\n"
        "Prepared text is selected, normalized, filtered, deduplicated and split from "
        "the source revisions and notices in notices/. Text is not repository MIT.\n"
        "Use corpus/*-train.jsonl for fitting and *-development.jsonl for selection.\n"
        "Reserved payloads and protected evaluation examples are intentionally absent. "
        "They remain in their controlled local locations. The original report lists "
        "additional evidence/reservations that are not bundled.\n"
        "Verify bundle.json before use. Tokenizer IDs differ from old learned weights. "
        "Use the LatoS loader and chat contract 1. Existing historical trainers do not "
        "automatically support this schema; Phase 16 owns integration/feasibility.\n"
        "No permission for redistribution or source-license relicensing is implied.\n"
    )
    (args.output / "README.txt").write_text(instructions)
    files["README.txt"] = {
        "bytes": (args.output / "README.txt").stat().st_size,
        "sha256": file_hash(args.output / "README.txt"),
    }
    write_json(
        args.output / "bundle.json",
        {
            "schema_version": 1,
            "files": files,
            "reserved_payloads_bundled": False,
            "weights_bundled": False,
            "parent_corpus_report_sha256": file_hash(args.corpus / "report.json"),
        },
    )


if __name__ == "__main__":
    main()
