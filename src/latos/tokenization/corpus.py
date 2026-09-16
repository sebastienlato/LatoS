"""Read only one verified prepared split; fitting has no held-out file dependency."""

import json
from pathlib import Path

from latos.data.manifest import SPLITS, load_manifest, sha256


def read_split(manifest_path: Path, corpus_dir: Path, split: str) -> tuple[list[str], dict]:
    if split not in SPLITS:
        raise ValueError("Unknown corpus split")
    manifest = load_manifest(manifest_path)
    report_data = (corpus_dir / "report.json").read_bytes()
    report = json.loads(report_data)
    manifest_hash = sha256(manifest_path.read_bytes())
    if report["schema_version"] != 1 or report["manifest_sha256"] != manifest_hash:
        raise ValueError("Corpus report does not match the supplied source manifest")
    summary = report["splits"][split]
    if type(summary["bytes"]) is not int or not 0 < summary["bytes"] <= 64_000_000:
        raise ValueError("Prepared split exceeds the tokenizer input byte budget")
    with (corpus_dir / f"{split}.jsonl").open("rb") as stream:
        data = stream.read(summary["bytes"] + 1)
    if len(data) != summary["bytes"] or sha256(data) != summary["sha256"]:
        raise ValueError(f"Prepared {split} split checksum mismatch")
    sources = {s["id"]: s for s in manifest["sources"] if s["split"] == split}
    texts, documents, source_ids, ids = [], set(), set(), set()
    for line in data.splitlines():
        record = json.loads(line)
        source = sources.get(record["source_id"])
        if source is None or any(
            record[key] != source[key] for key in ("split", "document_id", "group_id")
        ):
            raise ValueError("Record is not assigned to the requested split")
        text = record["text"]
        digest = sha256(text.encode("utf-8"))
        if (
            digest != record["text_sha256"]
            or record["id"] != f"{source['id']}:{record['paragraph']}:{digest}"
        ):
            raise ValueError("Prepared record text or identity mismatch")
        if record["id"] in ids or not text or len(text) > 8000:
            raise ValueError("Repeated, empty, or oversized prepared record")
        ids.add(record["id"])
        documents.add(record["document_id"])
        source_ids.add(record["source_id"])
        texts.append(text)
        if len(texts) > 50_000:
            raise ValueError("Tokenizer input exceeds the record budget")
    observed = {
        "records": len(texts),
        "documents": len(documents),
        "characters": sum(map(len, texts)),
        "words": sum(len(t.split()) for t in texts),
    }
    if not texts or any(summary[key] != value for key, value in observed.items()):
        raise ValueError("Prepared split statistics mismatch")
    return texts, {
        "manifest_id": manifest["id"],
        "manifest_sha256": manifest_hash,
        "corpus_report_sha256": sha256(report_data),
        "split": split,
        "split_sha256": sha256(data),
        "source_ids": sorted(source_ids),
        "document_ids": sorted(documents),
        **observed,
    }
