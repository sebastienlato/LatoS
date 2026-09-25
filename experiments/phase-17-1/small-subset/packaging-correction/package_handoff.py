"""Export only the reviewed corrective script; original Windows helpers stay untouched."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def package(output):
    root = Path(__file__).resolve().parents[4]
    if (
        output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip()
    ):
        raise ValueError("New destination and clean reviewed commit required")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    source = Path(__file__).with_name("correct.py").relative_to(root).as_posix()
    raw = subprocess.check_output(["git", "show", commit + ":" + source], cwd=root)
    with output.open("xb") as f:
        f.write(raw)
    assert output.read_bytes() == raw
    receipt = {
        "source_commit": commit,
        "source_path": source,
        "script": {
            "name": output.name,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "reported_original_seconds_floor": 1273,
        "cumulative_total_limit_seconds": 2700,
        "cumulative_packaging_limit_seconds": 300,
        "additional_optimizer_updates": 0,
        "scope": "One conditional packaging-only correction; original study unchanged",
        "publication_authorized": False,
        "phase18_authorized": False,
    }
    with output.with_suffix(".receipt.json").open("x", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    print(json.dumps(package(p.parse_args().output), indent=2))
