"""Package only diagnostic code, synthetic shape metadata and the selected config."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

FILES = {
    name: "experiments/recovery-diagnostic/" + name
    for name in (
        "run.py",
        "mechanics.py",
        "plan.json",
        "lengths.json",
        "run_checks.py",
        "package_return.py",
        "PLAN.md",
        "WINDOWS.md",
    )
}
FILES.update(
    {
        "selected-model.json": "configs/training2/selected-model.json",
        "test_diagnostic.py": "tests/test_recovery_diagnostic.py",
        "LICENSE": "LICENSE",
    }
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    ):
        raise ValueError("New archive and clean reviewed local commit required")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "schema_version": 1,
        "source_commit": commit,
        "files": {},
        "scope": "Recovery diagnostic only; no corpus, tokenizer, historical model, "
        "held-out or final payloads",
        "full_Phase17_1_authorized": False,
        "publication_authorized": False,
    }
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in FILES.items():
            data = subprocess.check_output(["git", "show", f"{commit}:{source}"])
            archive.writestr(name, data)
            manifest["files"][name] = {
                "source_path": source,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        archive.writestr("bundle.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    with zipfile.ZipFile(args.output) as archive:
        assert set(archive.namelist()) == set(FILES) | {"bundle.json"}
        for name, record in manifest["files"].items():
            data = archive.read(name)
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("Diagnostic archive readback mismatch")
    with args.output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(
        json.dumps(
            {
                "source_commit": commit,
                "sha256": digest,
                "bytes": args.output.stat().st_size,
                "files": len(FILES) + 1,
                "actual_CUDA_diagnostic_executed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
