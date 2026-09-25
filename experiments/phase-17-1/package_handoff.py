"""Allowlisted reviewed controller transfer; reuse original Windows source/data in place."""

import argparse
import ast
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from common import ATTEMPT, ORIGINAL, hash_file, require, write_once

FILES = {
    name: "experiments/phase-17-1/" + name
    for name in (
        "common.py",
        "runtime.py",
        "worker.py",
        "run.py",
        "package_return.py",
        "verify_handoff.py",
        "plan.json",
        "authorization.json",
        "fixture-identities.json",
        "PROPOSAL.md",
        "WINDOWS.md",
    )
}
FILES.update(
    {
        "fingerprints.py": "experiments/recovery-diagnostic/mechanics.py",
        "test_controller.py": "tests/test_phase171.py",
        "LICENSE": "LICENSE",
    }
)


def package(output, original):
    require(
        not output.exists()
        and not subprocess.check_output(["git", "status", "--porcelain"]).strip(),
        "A new archive and clean reviewed commit are required",
    )
    require(
        hash_file(original) == "be2af53a7f19a62c010e2735b097794d2a79b426bf4c489cb78c797a0d958e41",
        "Original transfer archive changed",
    )
    with zipfile.ZipFile(original) as z:
        previous = z.read("handoff.json")
    require(hashlib.sha256(previous).hexdigest() == ORIGINAL, "Original manifest mismatch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "schema_version": 1,
        "attempt_id": ATTEMPT,
        "source_commit": commit,
        "publication_authorized": False,
        "phase18_authorized": False,
        "files": {},
    }
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in FILES.items():
            raw = subprocess.check_output(["git", "show", f"{commit}:{path}"])
            archive.writestr(name, raw)
            manifest["files"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            manifest.setdefault("source_paths", {})[name] = path
            if name == "test_controller.py":
                manifest["cpu_test_count"] = sum(
                    isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
                    for n in ast.parse(raw).body
                )
        archive.writestr("original-transfer.json", previous)
        manifest["files"]["original-transfer.json"] = {"bytes": len(previous), "sha256": ORIGINAL}
        archive.writestr("bundle.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        require(
            set(archive.namelist()) == set(manifest["files"]) | {"bundle.json"}, "Member mismatch"
        )
        for name, expected in manifest["files"].items():
            raw = archive.read(name)
            require(
                len(raw) == expected["bytes"]
                and hashlib.sha256(raw).hexdigest() == expected["sha256"],
                "Readback mismatch",
            )
    receipt = {
        "source_commit": commit,
        "sha256": hash_file(output),
        "bytes": output.stat().st_size,
        "files": len(manifest["files"]) + 1,
        "scope": "Standalone controller/metadata; no corpus, tokenizer, model or private payload",
    }
    write_once(output.with_suffix(".receipt.json"), receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.output, args.original), indent=2))
