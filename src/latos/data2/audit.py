"""Read-only full-output integrity, group, duplication and contamination verification."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from latos.data2.acquire import file_hash
from latos.data2.filtering import validate_messages
from latos.data2.lexical import LexicalIndex, digest, grams
from latos.data2.prepare import split_for, text_of


def audit(corpus: Path, guard):
    report = json.loads((corpus / "report.json").read_text())
    for name, identity in report["files"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe corpus inventory name")
        p = corpus / name
        if p.stat().st_size != identity["bytes"] or file_hash(p) != identity["sha256"]:
            raise ValueError("Corpus output integrity failure")
    docs = {}
    with (corpus / "documents.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["id"] in docs or split_for(row["group_id"]) != row["split"]:
                raise ValueError("Duplicate document or incorrect split")
            docs[row["id"]] = row
    group_splits, ids, totals = defaultdict(set), set(), defaultdict(Counter)
    index, owners, exact, overlaps = LexicalIndex(), [], {}, Counter()
    affected, doc_rows = [], Counter()
    for kind in ("instruction", "pretrain"):
        for split in ("reserved", "development", "train"):
            with (corpus / f"{kind}-{split}.jsonl").open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    parent = row.get("document_id", row["id"])
                    doc = docs[parent]
                    if row["id"] in ids or row["kind"] != kind or row["split"] != split:
                        raise ValueError("Duplicate or misassigned record")
                    ids.add(row["id"])
                    doc_rows[parent] += 1
                    if doc["group_id"] != row["group_id"] or doc["split"] != split:
                        raise ValueError("Document/group leakage")
                    group_splits[row["group_id"]].add(split)
                    text = text_of(row)
                    if digest(text) != row["content_sha256"]:
                        raise ValueError("Content hash mismatch")
                    if kind == "instruction":
                        validate_messages(row["messages"])
                        parts = [m["content"] for m in row["messages"]]
                    else:
                        parts = [text]
                    for part in parts:
                        match = guard.match(part)
                        if match:
                            affected.append({"id": row["id"], "reason": match})
                    # Full prepared records: records may span chunk boundaries differently
                    # from the document preparation join; verify the actual output again.
                    value = digest(" ".join(text.casefold().split()))
                    signature = grams(text)
                    matches = index.matches(signature)
                    if value in exact:
                        matches = sorted(set(matches + [exact[value]]))
                    for match in matches:
                        other = owners[match]
                        relation = "within_split" if other[1] == split else "cross_split"
                        overlaps[kind + "/" + relation] += 1
                        if relation == "cross_split":
                            affected.append(
                                {
                                    "id": row["id"],
                                    "reason": "cross_split_lexical",
                                    "other_id": other[0],
                                }
                            )
                    exact.setdefault(value, len(owners))
                    index.add(signature)
                    owners.append((row["id"], split))
                    totals[kind + "/" + split]["records"] += 1
                    totals[kind + "/" + split]["utf8_bytes"] += len(text.encode())
    if any(len(v) != 1 for v in group_splits.values()):
        raise ValueError("Group crosses splits")
    if any(doc_rows[k] != v["rows"] for k, v in docs.items()):
        raise ValueError("Document record counts differ")
    for key, counts in totals.items():
        for field, value in counts.items():
            if report["splits"][key][field] != value:
                raise ValueError("Reported totals differ")
    return {
        "schema_version": 1,
        "report_sha256": file_hash(corpus / "report.json"),
        "records": len(ids),
        "documents": len(docs),
        "groups": len(group_splits),
        "group_cross_split": 0,
        "output_near_pairs": dict(overlaps),
        "affected": affected,
        "passed": not affected,
        "protected": guard.summary(),
        "scope": (
            "full output, all splits, no model scoring; exact and Jaccard80 full "
            "records; protected exact/13word/near"
        ),
        "limits": (
            "Lexical and source exclusions cannot certify semantic or unknown-"
            "origin absence; short common answers match only whole segments. No "
            "exposure to reserved examples is printed."
        ),
    }
