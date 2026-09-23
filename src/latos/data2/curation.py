"""Monotonic final curation: only remove records, never rewrite text or reassign splits."""

import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from latos.data2.acquire import file_hash, write_json
from latos.data2.lexical import LexicalIndex, digest, grams, normalized, words
from latos.data2.prepare import sample_records, text_of
from latos.data2.resources import peak_rss_bytes


def finalize(corpus: Path, quarantine: Path, output: Path):
    start = time.monotonic()
    report = json.loads((corpus / "report.json").read_text())
    for name, item in report["files"].items():
        if Path(name).name != name or file_hash(corpus / name) != item["sha256"]:
            raise ValueError("Parent corpus identity failure")
    policy = json.loads(quarantine.read_text())
    if policy["schema_version"] != 1 or not isinstance(policy["records"], dict):
        raise ValueError("Invalid output quarantine")
    docs = {}
    with (corpus / "documents.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs[row["id"]] = row
    if not set(policy["records"]).issubset(docs):
        raise ValueError("Quarantine ID is absent from the parent corpus")
    output.mkdir(parents=True, exist_ok=False)
    index, owners, row_counts = LexicalIndex(), [], Counter()
    exact = {}
    stats, removed = defaultdict(Counter), []
    for kind in ("instruction", "pretrain"):
        for split in ("reserved", "development", "train"):
            name = f"{kind}-{split}.jsonl"
            with (
                (corpus / name).open(encoding="utf-8") as source,
                (output / name).open("x", encoding="utf-8") as dest,
            ):
                for line in source:
                    row = json.loads(line)
                    parent = row.get("document_id", row["id"])
                    if parent in policy["records"]:
                        continue
                    signature = grams(text_of(row))
                    text_key = digest(normalized(text_of(row)))
                    if kind == "pretrain":
                        matches = index.matches(signature)
                        if text_key in exact:
                            matches = sorted(set(matches + [exact[text_key]]))
                        if matches:
                            removed.append(
                                {
                                    "removed_id": row["id"],
                                    "kept_id": owners[matches[0]],
                                    "kind": "final_chunk_exact_or_jaccard80",
                                    "match_type": "exact" if text_key in exact else "near",
                                }
                            )
                            continue
                    exact.setdefault(text_key, len(owners))
                    index.add(signature)
                    owners.append(row["id"])
                    dest.write(line)  # Preserve exact existing bytes/content/IDs/split.
                    row_counts[parent] += 1
                    key = kind + "/" + split
                    stats[key]["records"] += 1
                    stats[key]["utf8_bytes"] += len(text_of(row).encode())
                    stats[key]["words"] += len(words(text_of(row)))
    kept_docs = []
    for identity, n in sorted(row_counts.items()):
        doc = {**docs[identity], "rows": n}
        kept_docs.append(doc)
        key = doc["kind"] + "/" + doc["split"]
        stats[key]["documents"] += 1
        stats[key]["source/" + doc["source"]] += 1
        stats[key]["category/" + doc["category"]] += 1
        if doc["category"] == "multi_turn":
            stats[key]["multi_turn"] += 1
    with (output / "documents.jsonl").open("x", encoding="utf-8") as f:
        for row in kept_docs:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    for name in ("quality-before.json", "duplicates.jsonl", "rejections.jsonl"):
        shutil.copyfile(corpus / name, output / name)
    with (output / "duplicates.jsonl").open("a", encoding="utf-8") as f:
        for row in removed:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with (output / "rejections.jsonl").open("a", encoding="utf-8") as f:
        for identity in sorted(policy["records"]):
            f.write(
                json.dumps(
                    {"id": identity, "reason": "post_filter_quality_quarantine"}, sort_keys=True
                )
                + "\n"
            )
    # Reproduce the fixed sample rule over remaining train/development documents.
    selected = sample_records([d for d in kept_docs if d["split"] != "reserved"])
    selected_ids = {d["id"] for rows in selected.values() for d in rows}
    samples, pieces = {}, defaultdict(list)
    for kind in ("instruction", "pretrain"):
        for split in ("train", "development"):
            with (output / f"{kind}-{split}.jsonl").open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    identity = row.get("document_id", row["id"])
                    if identity in selected_ids:
                        samples[identity] = {
                            **docs[identity],
                            **({"messages": row["messages"]} if kind == "instruction" else {}),
                        }
                        if kind == "pretrain":
                            pieces[identity].append(row["text"])
    for identity, texts in pieces.items():
        samples[identity]["text"] = "\n\n".join(texts)
    write_json(
        output / "quality-after.json",
        {k: [samples[d["id"]] for d in rows] for k, rows in selected.items()},
    )
    report["splits"] = dict(stats)
    report["rejections"]["post_filter_quality_quarantine"] = len(policy["records"])
    report["duplicates"]["final_chunk_exact_or_jaccard80"] = len(removed)
    report["retained_groups"] = len({d["group_id"] for d in kept_docs})
    report["finalization"] = {
        "parent_report_sha256": file_hash(corpus / "report.json"),
        "quarantine_sha256": file_hash(quarantine),
        "implementation_sha256": file_hash(Path(__file__)),
        "policy": "Removal only; all retained text/IDs and component split assignments unchanged",
        "removed_records": len(removed),
        "quarantined_documents": len(policy["records"]),
    }
    report["files"] = {
        p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
        for p in sorted(output.iterdir())
        if p.is_file()
    }
    write_json(output / "report.json", report)
    write_json(
        output / "resources.json",
        {"seconds": time.monotonic() - start, "peak_rss_bytes": peak_rss_bytes()},
    )
    return report
