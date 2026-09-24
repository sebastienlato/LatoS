"""Standard-library verification before installing or running the Windows handoff."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Extracted folder containing handoff.json")
    args = parser.parse_args()
    manifest = json.loads((args.root / "handoff.json").read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
        path = args.root / relative
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != record["bytes"] or digest != record["sha256"]:
            raise ValueError(f"Handoff identity mismatch: {name}")
    print(
        json.dumps(
            {
                "verified_files": len(manifest["files"]),
                "source_commit": manifest["source_commit"],
                "phase16_complete": False,
            }
        )
    )


if __name__ == "__main__":
    main()
