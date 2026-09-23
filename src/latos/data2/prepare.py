"""Document/group isolation, complete lexical joins and reproducible new corpora."""

import gc
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from latos.data2.acquire import file_hash, write_json
from latos.data2.filtering import (
    SOURCE_BLOCK,
    clean,
    is_english,
    make_detector,
    quality_reason,
    response_reason,
    validate_messages,
)
from latos.data2.lexical import Groups, LexicalIndex, digest, grams, normalized, words
from latos.data2.protected import fragments
from latos.data2.sources import load_sources


def text_of(record):
    return (
        record["text"]
        if record["kind"] == "pretrain"
        else "\n".join(m["content"] for m in record["messages"])
    )


def split_for(identity):
    value = int(digest("data2-split-v1:" + identity)[:16], 16) % 100
    return "reserved" if value < 2 else "development" if value < 4 else "train"


def family_keys(record):
    if record["kind"] == "pretrain":
        title = re.sub(r"\([^)]*\)", "", record["title"])
        yield "title:" + normalized(re.sub(r"\b\d+\b", "NUMBER", title))
    else:
        # Groups recurring prompt stems even when named entities/numbers differ.
        prompt = re.sub(r'"[^"\n]*"|\b\d+\b', " SLOT ", record["prompt"])
        yield "prompt:" + normalized(prompt)
        tokens = words(prompt)
        if len(tokens) >= 5:
            yield "stem:" + " ".join(tokens[:5])
        if record.get("context"):
            yield "context:" + digest(normalized(record["context"]))


def sample_records(records, count=10):
    groups = defaultdict(list)
    for r in records:
        groups[r["source"]].append(r)
        if r["source"] == "dolly":
            groups["dolly/" + r["category"]].append(r)
    selected = {}
    for key, rows in groups.items():
        selected[key] = sorted(rows, key=lambda r: digest("data2-quality-v1:" + r["id"]))[
            : 3 if "/" in key else count
        ]
    return selected


def paragraph_units(record):
    if record["kind"] == "pretrain":
        return [p.strip() for p in re.split(r"\n+", record["text"]) if len(words(p)) >= 40]
    return (
        [record["context"]] if record.get("context") and len(words(record["context"])) >= 40 else []
    )


def prepare(manifest: Path, raw: Path, output: Path, guard, *, detector=None):
    output.mkdir(parents=True, exist_ok=False)
    records, source_stats, rejections = load_sources(manifest, raw)
    exclusions_path = manifest.parent / "quality-exclusions-v1.json"
    exclusions = (
        json.loads(exclusions_path.read_text())["records"] if exclusions_path.exists() else {}
    )
    write_json(output / "quality-before.json", sample_records(records))
    if detector is None:
        detector = make_detector()
    kept = []
    for number, record in enumerate(records):
        if number and number % 10000 == 0:
            print(f"Quality scan {number}/{len(records)}", flush=True)
        if record["id"] in exclusions:
            rejections.append({"id": record["id"], "reason": "quality_sample_quarantine"})
            continue
        if record["kind"] == "pretrain":
            record["text"] = clean(record["text"], encyclopedia=True)
        else:
            try:
                validate_messages(record["messages"])
            except ValueError, UnicodeError:
                rejections.append({"id": record["id"], "reason": "malformed_conversation"})
                continue
        text = text_of(record)
        reason = (
            "excluded_source_family"
            if SOURCE_BLOCK.search(record.get("title", ""))
            else quality_reason(text, instruction=record["kind"] == "instruction")
        )
        if not reason and record["kind"] == "instruction":
            reason = response_reason(record)
        if not reason and not is_english(detector, text):
            reason = "language_not_confident_english"
        if not reason:
            for part in fragments(record):
                if record["kind"] == "pretrain" and len(words(part)) < 5:
                    continue  # Short headings/common answers are not informative overlap.
                reason = guard.match(part)
                if reason:
                    break
        if reason:
            rejections.append({"id": record["id"], "reason": reason})
        else:
            kept.append(record)
    del records, detector
    gc.collect()
    print(f"Quality/contamination retained {len(kept)} documents", flush=True)
    groups, families = Groups(len(kept)), {}
    exact, index, duplicates, removed = {}, LexicalIndex(), [], set()
    prompts, prompt_owners = LexicalIndex(0.6), []
    for i, record in enumerate(kept):
        for family in family_keys(record):
            if family in families:
                groups.union(i, families[family])
            families.setdefault(family, i)
        text = text_of(record)
        signature = digest(normalized(text))
        values = grams(text)
        matches = index.matches(values)
        if signature in exact:
            matches = sorted(set(matches + [exact[signature]]))
        for other in matches:
            groups.union(i, other)
        if matches:
            removed.add(i)
            duplicates.append(
                {
                    "removed_id": record["id"],
                    "kept_id": kept[matches[0]]["id"],
                    "kind": "document_exact_or_jaccard80",
                }
            )
        exact.setdefault(signature, i)
        index.add(values)
        if record["kind"] == "instruction":
            prompt_values = grams(record["prompt"], 3)
            for other in prompts.matches(prompt_values):
                groups.union(i, prompt_owners[other])
            prompts.add(prompt_values)
            prompt_owners.append(i)
        if (i + 1) % 10000 == 0:
            print(f"Document join {i + 1}/{len(kept)}", flush=True)
    comparisons = index.comparisons
    del index, exact, prompts, prompt_owners
    gc.collect()
    # Identify embedded near copies/context reuse before split assignment. Instruction
    # context is never removed. Earlier instruction records take retention priority.
    paragraph_index, owners, dropped_parts = LexicalIndex(), [], defaultdict(set)
    for i, record in enumerate(kept):
        if i in removed:
            continue
        for position, paragraph in enumerate(paragraph_units(record)):
            values = grams(paragraph)
            matches = paragraph_index.matches(values)
            for match in matches:
                groups.union(i, owners[match][0])
            if matches and record["kind"] == "pretrain":
                dropped_parts[i].add(position)
                duplicates.append(
                    {
                        "removed_id": record["id"] + f":p{position}",
                        "kept_id": kept[owners[matches[0]][0]]["id"],
                        "kind": "paragraph_jaccard80",
                    }
                )
            paragraph_index.add(values)
            owners.append((i, position))
        if (i + 1) % 10000 == 0:
            print(f"Paragraph join {i + 1}/{len(kept)}", flush=True)
    paragraph_comparisons = paragraph_index.comparisons
    del paragraph_index, owners
    gc.collect()
    roots = defaultdict(list)
    for i, r in enumerate(kept):
        roots[groups.find(i)].append(r["id"])
    group_ids = {root: digest("\n".join(sorted(ids))) for root, ids in roots.items()}
    stats, documents, after = defaultdict(Counter), [], []
    with (output / "duplicates.jsonl").open("x", encoding="utf-8") as f:
        for row in duplicates:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    handles = {
        (kind, split): (output / f"{kind}-{split}.jsonl").open("x", encoding="utf-8")
        for kind in ("pretrain", "instruction")
        for split in ("train", "development", "reserved")
    }
    try:
        for i, record in enumerate(kept):
            if i in removed:
                continue
            group = group_ids[groups.find(i)]
            split = split_for(group)
            base = {
                k: v
                for k, v in record.items()
                if k not in ("text", "messages", "context", "prompt")
            }
            base.update(group_id=group, split=split)
            kind = record["kind"]
            rows = []
            if kind == "instruction":
                rows = [{**base, "messages": record["messages"]}]
            else:
                for j, paragraph in enumerate(paragraph_units(record)):
                    if j in dropped_parts[i]:
                        continue
                    # Chunks only now, after all document/group assignments are fixed.
                    for offset in range(0, len(paragraph), 8000):
                        text = paragraph[offset : offset + 8000]
                        rows.append(
                            {
                                **base,
                                "id": record["id"] + f":p{j}:c{offset // 8000}",
                                "document_id": record["id"],
                                "text": text,
                            }
                        )
            if not rows:
                rejections.append({"id": record["id"], "reason": "no_retained_paragraph"})
                continue
            documents.append({**base, "rows": len(rows)})
            after.append(
                {**record, "text": "\n\n".join(r["text"] for r in rows)}
                if kind == "pretrain"
                else record
            )
            key = kind + "/" + split
            stats[key]["documents"] += 1
            stats[key]["source/" + record["source"]] += 1
            stats[key]["category/" + record["category"]] += 1
            if kind == "instruction" and len(record["messages"]) > 2:
                stats[key]["multi_turn"] += 1
            for row in rows:
                text = text_of(row)
                row["content_sha256"] = digest(text)
                handles[kind, split].write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
                stats[key]["records"] += 1
                stats[key]["utf8_bytes"] += len(text.encode())
                stats[key]["words"] += len(words(text))
    finally:
        for handle in handles.values():
            handle.close()
    with (output / "documents.jsonl").open("x", encoding="utf-8") as f:
        for row in documents:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    with (output / "rejections.jsonl").open("x", encoding="utf-8") as f:
        for row in rejections:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    # Reserve new reserved payloads too: samples for review use train/development only.
    eligible = {r["id"] for r in documents if r["split"] != "reserved"}
    write_json(
        output / "quality-after.json", sample_records([r for r in after if r["id"] in eligible])
    )
    report = {
        "schema_version": 1,
        "manifest_sha256": file_hash(manifest),
        "quality_exclusions_sha256": file_hash(exclusions_path)
        if exclusions_path.exists()
        else None,
        "source_stats": source_stats,
        "splits": dict(stats),
        "rejections": dict(Counter(r["reason"] for r in rejections)),
        "duplicates": dict(Counter(r["kind"] for r in duplicates)),
        "groups": len(roots),
        "largest_groups": sorted([len(ids) for ids in roots.values()], reverse=True)[:10],
        "document_jaccard_comparisons": comparisons,
        "paragraph_jaccard_comparisons": paragraph_comparisons,
        "dedup_coverage": (
            "all selected quality-passing documents and >=40-word "
            "paragraphs/context; exhaustive prefix candidate join, no sampled "
            "estimate"
        ),
        "guard": guard.summary(),
        "files": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file():
            report["files"][path.name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    write_json(output / "report.json", report)
    return report
