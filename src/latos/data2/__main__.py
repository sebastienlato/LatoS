"""Explicit offline preparation/audit commands and separate pinned acquisition."""

import argparse
from pathlib import Path

from latos.data2.acquire import acquire, write_json


def main():
    parser = argparse.ArgumentParser(description="LatoS Data 2.0 (no model training)")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("acquire")
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--raw", type=Path, required=True)
    check = sub.add_parser("audit")
    check.add_argument("--root", type=Path, default=Path.cwd())
    check.add_argument("--corpus", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(args.manifest, args.raw)
    else:
        from latos.data2.audit import audit
        from latos.data2.protected import build_guard

        if args.output.exists():
            raise ValueError("Audit output already exists")
        result = audit(args.corpus, build_guard(args.root.resolve()))
        write_json(args.output, result)
        if not result["passed"]:
            raise ValueError("Audit detected affected records; see the retained report")
        print("Full corpus audit passed")


if __name__ == "__main__":
    main()
