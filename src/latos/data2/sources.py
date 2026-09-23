"""Parse pinned data as data, never execute remote dataset code."""

import gzip
import heapq
import json
from collections import Counter
from pathlib import Path

from latos.data2.acquire import read_manifest, verify
from latos.data2.filtering import clean, validate_messages
from latos.data2.lexical import digest


def node_allowed(node):
    labels = node.get("labels") or {}
    if node.get("lang") != "en" or node.get("review_result") is not True or node.get("deleted"):
        return False
    if node.get("synthetic") is not False or node.get("model_name"):
        return False
    if any(
        labels.get(k, {}).get("value", 0) > 0
        for k in ("spam", "pii", "not_appropriate", "hate_speech", "sexual_content")
    ):
        return False
    return labels.get("quality", {}).get("value", 0) >= 0.5


def best_path(tree):
    """One deterministic reviewed path/tree, avoiding inflated shared-prefix exposures."""
    root = tree["prompt"]
    if tree["tree_state"] != "ready_for_export" or not node_allowed(root):
        return []
    result = [root]
    while len(result) < 12:
        expected = "assistant" if result[-1]["role"] == "prompter" else "prompter"
        options = [
            n
            for n in result[-1].get("replies", [])
            if node_allowed(n) and n.get("role") == expected
        ]
        options.sort(
            key=lambda n: (
                n.get("rank") if n.get("rank") is not None else 1_000_000,
                -n.get("labels", {}).get("quality", {}).get("value", 0),
                n["message_id"],
            )
        )
        if not options:
            break
        chosen = options[0]
        if chosen.get("parent_id", result[-1]["message_id"]) != result[-1]["message_id"]:
            raise ValueError("Broken conversation parent link")
        if chosen["message_id"] in {n["message_id"] for n in result}:
            raise ValueError("Cyclic conversation")
        result.append(chosen)
    if result[-1]["role"] != "assistant":
        result.pop()
    return result


def load_sources(manifest_path: Path, raw: Path):
    import pyarrow.parquet as pq

    manifest = read_manifest(manifest_path)
    records, stats, rejected = [], Counter(), []
    for source in manifest["sources"]:
        for item in source["files"]:
            verify(raw / item["file"], item)
    wiki_heap = []
    for source in manifest["sources"]:
        for item in source["files"]:
            if item["purpose"] != "payload":
                continue
            path = raw / item["file"]
            if source["id"] == "wiki":
                for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
                    for row in batch.to_pylist():
                        stats["wiki_raw_documents"] += 1
                        text = row["text"]
                        if not all(
                            isinstance(row.get(k), str) for k in ("id", "title", "url", "text")
                        ):
                            raise ValueError("Malformed Wikipedia schema")
                        identity = "wiki:" + row["id"]
                        if len(text.encode()) > 100_000:
                            rejected.append(
                                {"id": identity, "reason": "oversized_before_selection"}
                            )
                            continue
                        record = {
                            "id": identity,
                            "source": "wiki",
                            "kind": "pretrain",
                            "category": "encyclopedia",
                            "title": row["title"],
                            "url": row["url"],
                            "text": text,
                            "raw_file": item["file"],
                        }
                        rank = int(digest(identity), 16)
                        entry = (-rank, identity, record)
                        if len(wiki_heap) < 180_000:
                            heapq.heappush(wiki_heap, entry)
                        elif rank < -wiki_heap[0][0]:
                            heapq.heapreplace(wiki_heap, entry)
            elif source["id"] == "dolly":
                with path.open(encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        stats["dolly_raw_documents"] += 1
                        row = json.loads(line)
                        if not all(
                            isinstance(row.get(k), str)
                            for k in ("instruction", "context", "response", "category")
                        ):
                            raise ValueError("Malformed Dolly schema")
                        messages = [
                            {
                                "role": "user",
                                "content": clean(
                                    row["instruction"]
                                    + ("\n\nContext:\n" + row["context"] if row["context"] else "")
                                ),
                            },
                            {"role": "assistant", "content": clean(row["response"])},
                        ]
                        record = {
                            "id": f"dolly:{i:05}",
                            "source": "dolly",
                            "kind": "instruction",
                            "category": row["category"],
                            "messages": messages,
                            "context": clean(row["context"]),
                            "prompt": clean(row["instruction"]),
                            "raw_file": item["file"],
                            "raw_row": i,
                        }
                        records.append(record)
            elif source["id"] == "oasst":
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    for line in f:
                        stats["oasst_raw_trees"] += 1
                        tree = json.loads(line)
                        nodes = best_path(tree)
                        if not nodes:
                            rejected.append(
                                {
                                    "id": "oasst:" + tree["message_tree_id"],
                                    "reason": "unreviewed_nonenglish_synthetic_or_low_quality_tree",
                                }
                            )
                            continue
                        messages = [
                            {
                                "role": "user" if n["role"] == "prompter" else "assistant",
                                "content": clean(n["text"]),
                            }
                            for n in nodes
                        ]
                        validate_messages(messages)
                        records.append(
                            {
                                "id": "oasst:" + tree["message_tree_id"],
                                "source": "oasst",
                                "kind": "instruction",
                                "category": "multi_turn" if len(nodes) > 2 else "single_turn",
                                "messages": messages,
                                "prompt": messages[0]["content"],
                                "raw_file": item["file"],
                                "message_ids": [n["message_id"] for n in nodes],
                                "message_quality": [n["labels"]["quality"]["value"] for n in nodes],
                            }
                        )
            else:
                raise ValueError("Unknown source adapter")
    chosen = sorted((x[2] for x in wiki_heap), key=lambda r: digest(r["id"]))
    size = 0
    for record in chosen:
        n = len(record["text"].encode())
        if size + n > 600_000_000:
            rejected.append({"id": record["id"], "reason": "selected_text_byte_budget"})
            continue
        size += n
        records.append(record)
    stats["wiki_selected_raw_bytes"] = size
    stats["wiki_selected_documents"] = sum(r["source"] == "wiki" for r in records)
    stats["wiki_not_selected"] = stats["wiki_raw_documents"] - stats["wiki_selected_documents"]
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate upstream source ID")
    return sorted(records, key=lambda r: r["id"]), dict(stats), rejected
