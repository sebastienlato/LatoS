"""Standard-library transfer verification, before installation or execution."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root / "handoff.json").read_text(encoding="utf-8"))
    if manifest.get("phase") != 17 or manifest.get("execution_authorized") is not True:
        raise ValueError("Wrong handoff")
    for name, record in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Unsafe transfer path")
        path = root / relative
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != record["bytes"] or digest != record["sha256"]:
            raise ValueError(f"Transfer identity mismatch: {name}")
    return {
        "verified_files": len(manifest["files"]),
        "source_commit": manifest["source_commit"],
        "phase17_complete": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
