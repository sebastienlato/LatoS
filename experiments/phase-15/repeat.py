"""Bounded real-source replay; explicitly not a second full-corpus preparation."""

import argparse
import copy
import importlib
import time
from pathlib import Path

from latos.data2.acquire import file_hash, write_json
from latos.data2.lexical import digest
from latos.data2.protected import build_guard
from latos.data2.resources import peak_rss_bytes
from latos.data2.sources import load_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    start = time.monotonic()
    records, _, _ = load_sources(root / "configs/data2/sources-v1.json", root / "data/raw/data2-v1")
    sample = []
    for source, limit in (("wiki", 3000), ("dolly", 500), ("oasst", 500)):
        rows = sorted(
            (r for r in records if r["source"] == source),
            key=lambda r: digest("data2-replay-v1:" + r["id"]),
        )
        sample.extend(rows[:limit])
    del records
    sample.sort(key=lambda r: r["id"])
    module = importlib.import_module("latos.data2.prepare")
    original_loader = module.load_sources
    # The only substitution is a declared subset of the actual pinned source adapter.
    # Both passes execute the real filters, guard, complete joins, splitting and writes.
    module.load_sources = lambda *_: (
        copy.deepcopy(sample),
        {"bounded_replay_candidates": len(sample)},
        [],
    )
    try:
        reports = [
            module.prepare(
                root / "configs/data2/sources-v1.json",
                root / "data/raw/data2-v1",
                args.output / name,
                build_guard(root),
            )
            for name in ("a", "b")
        ]
    finally:
        module.load_sources = original_loader
    identical = reports[0] == reports[1] and all(
        file_hash(args.output / "a" / name) == file_hash(args.output / "b" / name)
        for name in reports[0]["files"]
    )
    write_json(
        args.output / "replay.json",
        {
            "identical": identical,
            "candidates": len(sample),
            "sample_ids_sha256": digest("\n".join(r["id"] for r in sample)),
            "report_sha256": file_hash(args.output / "a/report.json"),
            "scope": "4000 deterministic real-source candidates; not a full-corpus repeat",
            "seconds": time.monotonic() - start,
            "peak_rss_bytes": peak_rss_bytes(),
        },
    )
    if not identical:
        raise ValueError("Real-source replay differs")


if __name__ == "__main__":
    main()
