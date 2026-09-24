"""Full accepted train/development layout accounting; no model or optimizer is created."""

import argparse
import hashlib
import json
import time
from array import array
from itertools import groupby
from pathlib import Path

import torch

from latos.data.manifest import canonical_json, sha256
from latos.data2.resources import peak_rss_bytes
from latos.data2.tokenizer import read_records
from latos.model.storage import file_hash
from latos.tokenization import LatoTokenizer
from latos.training.data2 import POLICY, coalesce_document
from latos.training.probe import verify_inputs


def account(package, output):
    verify_inputs(package)
    codec = LatoTokenizer.load(package / "tokenizer")
    result = {
        "scope": (
            "Complete train/development layout accounting only; no model/training/reserved read"
        ),
        "policy": POLICY,
        "context_length": 512,
        "seed": 160,
        "splits": {},
    }
    group_sets = {}
    for split in ["train", "development"]:
        start = time.perf_counter()
        lengths = array("I")
        seen = set()
        groups = set()
        identities = []
        content = byte_count = record_count = max_bytes = max_ids = 0
        exceeded = []
        digest = hashlib.sha256()
        for doc, rows in groupby(
            read_records(package / "corpus", "pretrain", split), lambda r: r["document_id"]
        ):
            if doc in seen:
                raise ValueError("Noncontiguous document")
            seen.add(doc)
            rows = list(rows)
            record_count += len(rows)
            groups.add(rows[0]["group_id"])
            text = coalesce_document(rows)
            nbytes = len(text.encode())
            ids = codec.encode(text, add_bos=True, add_eos=True)
            if nbytes > 262144 or len(ids) > 65536:
                exceeded.append(doc)
            max_bytes = max(max_bytes, nbytes)
            max_ids = max(max_ids, len(ids))
            content += len(ids) - 2
            byte_count += nbytes
            identities.append({"id": doc, "text_sha256": sha256(text.encode())})
            for i in range(0, len(ids) - 1, 511):
                window = ids[i : i + 512]
                lengths.append(len(window))
                digest.update(canonical_json(window))
        targets = sum(n - 1 for n in lengths)
        order = torch.randperm(len(lengths), generator=torch.Generator().manual_seed(160)).tolist()
        batches = [order[i : i + 2] for i in range(0, len(order), 2)]
        padded_slots = sum(len(b) * (max(lengths[j] for j in b) - 1) for b in batches)
        final_batches = len(batches) % 8 or 8
        result["splits"][split] = {
            "documents": len(seen),
            "records": record_count,
            "groups": len(groups),
            "content_bytes": byte_count,
            "content_token_positions": content,
            "targets_with_EOS": targets,
            "windows": len(lengths),
            "maximum_document_bytes": max_bytes,
            "maximum_document_ids": max_ids,
            "probe_per_document_bounds_exceeded": len(exceeded),
            "window_sequence_sha256": digest.hexdigest(),
            "document_identity_sha256": sha256(canonical_json(identities)),
            "source_file_sha256": file_hash(package / "corpus" / f"pretrain-{split}.jsonl"),
            "fixed_slot_fraction": targets / (len(lengths) * 511),
            "shuffled_batch2_padded_target_slots": padded_slots,
            "shuffled_batch2_useful_fraction": targets / padded_slots,
            "batch2_microbatches": len(batches),
            "updates_at_accumulation8": (len(batches) + 7) // 8,
            "last_update_microbatches": final_batches,
            "last_microbatch_windows": len(batches[-1]),
            "seconds": time.perf_counter() - start,
            "Mac_process_peak_RSS_bytes": peak_rss_bytes(),
            "compact_token_uint32_upper_bytes": 4 * sum(lengths),
            "compact_window_offsets_uint64_bytes": 8 * (len(lengths) + 1),
        }
        group_sets[split] = groups
    if group_sets["train"] & group_sets["development"]:
        raise ValueError("Cross-split group")
    result["cross_split_groups"] = 0
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Output exists")
    print(json.dumps(account(a.inputs, a.output), indent=2))


if __name__ == "__main__":
    main()
