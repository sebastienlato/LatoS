"""Bounded Data 2.0 probes: verified splits and isolated coalesced documents."""

import re
from collections import defaultdict
from itertools import groupby
from pathlib import Path

from latos.data.manifest import canonical_json, sha256
from latos.data2.tokenizer import read_records
from latos.model.storage import file_hash
from latos.tokenization import LatoTokenizer
from latos.training.data import TokenDataset

POLICY = "data2-retained-document-overlap-one-v1"


def coalesce_document(rows: list[dict]) -> str:
    """Rejoin adjacent chunks verbatim; separate paragraphs and missing chunks."""
    if not rows:
        raise ValueError("Cannot coalesce an empty document")
    document = rows[0]["document_id"]
    group, split = rows[0]["group_id"], rows[0]["split"]
    parts = {}
    for row in rows:
        match = re.fullmatch(re.escape(document) + r":p(\d+):c(\d+)", row["id"])
        if (
            not match
            or row["document_id"] != document
            or row["group_id"] != group
            or row["split"] != split
        ):
            raise ValueError("Inconsistent document/chunk identity")
        key = tuple(map(int, match.groups()))
        if key in parts:
            raise ValueError("Duplicate paragraph/chunk position")
        parts[key] = row["text"]
    result, previous = [], None
    for key, text in sorted(parts.items()):
        if previous is not None and key != (previous[0], previous[1] + 1):
            result.append("\n\n")
        result.append(text)
        previous = key
    return "".join(result)


def prepare_probe_dataset(
    corpus: Path,
    tokenizer: Path,
    split: str,
    sequence_length: int,
    *,
    max_documents: int,
) -> tuple[TokenDataset, dict]:
    """Read no reservations; hash-select whole documents before any tokenization.

    This bounded in-memory probe adapter is not the full Phase 17 data engine.
    Tokenization is streamed one selected document at a time. Only a bounded
    subset of token windows is retained; no cross-document attention is allowed.
    """
    if split not in ("train", "development"):
        raise ValueError("Probe inputs accept only train/development; reservations stay closed")
    if type(max_documents) is not int or not 1 <= max_documents <= 4096:
        raise ValueError("Probe document bound must be in [1, 4096]")
    if type(sequence_length) is not int or not 2 <= sequence_length <= 4096:
        raise ValueError("Invalid probe sequence length")
    codec = LatoTokenizer.load(tokenizer)
    corpus_hash = file_hash(corpus / "report.json")
    identities = {r["document_id"] for r in read_records(corpus, "pretrain", split)}
    selected = set(
        sorted(identities, key=lambda x: (sha256((POLICY + ":" + x).encode()), x))[:max_documents]
    )
    windows, included, groups = [], [], set()
    seen, content_tokens, content_bytes = set(), 0, 0
    isolated_targets, isolated_windows = 0, 0
    omissions = defaultdict(int)
    records = read_records(corpus, "pretrain", split)
    for document, iterable in groupby(records, key=lambda r: r["document_id"]):
        if document in seen:
            raise ValueError("Corpus document rows are not contiguous")
        seen.add(document)
        if document not in selected:
            for _ in iterable:
                pass
            continue
        rows = list(iterable)
        text = coalesce_document(rows)
        if len(text.encode()) > 262144:
            omissions["document_byte_limit"] += 1
            continue
        ids = codec.encode(text, add_bos=True, add_eos=True)
        if len(ids) > 65536 or content_tokens + len(ids) - 2 > 4_000_000:
            omissions["token_budget"] += 1
            continue
        included.append({"id": document, "text_sha256": sha256(text.encode())})
        groups.add(rows[0]["group_id"])
        content_bytes += len(text.encode())
        content_tokens += len(ids) - 2
        for row in rows:
            targets = len(codec.encode(row["text"])) + 1
            isolated_targets += targets
            isolated_windows += (targets + sequence_length - 2) // (sequence_length - 1)
        windows.extend(
            tuple(ids[start : start + sequence_length])
            for start in range(0, len(ids) - 1, sequence_length - 1)
        )
    selection = {
        "policy": POLICY,
        "corpus_report_sha256": corpus_hash,
        "file_sha256": file_hash(corpus / f"pretrain-{split}.jsonl"),
        "max_documents": max_documents,
        "selected_documents": len(selected),
        "included": included,
        "groups": sorted(groups),
        "omissions": dict(omissions),
    }
    dataset = TokenDataset(
        tuple(windows),
        sequence_length,
        codec.vocab_size,
        file_hash(tokenizer / "tokenizer.json"),
        "train" if split == "train" else "validation",
        sha256(canonical_json(selection)),
    )
    targets = sum(dataset.target_count(i) for i in range(len(windows)))
    return dataset, {
        **selection,
        "content_tokens": content_tokens,
        "content_bytes": content_bytes,
        "targets": targets,
        "windows": len(windows),
        "fixed_slot_fraction": targets / (len(windows) * (sequence_length - 1)),
        "isolated_record_fixed_slot_fraction": isolated_targets
        / (isolated_windows * (sequence_length - 1)),
        "note": "Within-document seams overlap one; no cross-document transition/attention. "
        "Missing material is not reconstructed. Newline joins change content token accounting. "
        "Fixed-slot utilization is arithmetic, not measured CUDA throughput.",
    }
