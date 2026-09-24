"""Create a local transfer archive from a reviewed clean commit and pinned inputs."""

import argparse
import io
import json
import subprocess
import zipfile
from pathlib import Path

from latos.data.manifest import canonical_json, sha256
from latos.model.storage import file_hash
from latos.training.probe import BUNDLE_SHA256, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Handoff archive already exists")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Handoff requires a reviewed clean local source commit")
    verify_inputs(args.inputs)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source = subprocess.check_output(["git", "archive", "--format=zip", commit])
    manifest = {
        "schema_version": 1,
        "source_commit": commit,
        "phase16_complete": False,
        "phase17_authorized": False,
        "input_bundle_sha256": BUNDLE_SHA256,
        "files": {},
    }
    bundle = json.loads((args.inputs / "bundle.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        zipfile.ZipFile(io.BytesIO(source)) as reviewed,
        zipfile.ZipFile(
            args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive,
    ):
        for item in reviewed.infolist():
            if item.is_dir():
                continue
            path = Path(item.filename)
            if path.is_absolute() or any(p in ("..", ".private", ".venv") for p in path.parts):
                raise ValueError("Unsafe source archive member")
            data = reviewed.read(item)
            name = "source/" + item.filename
            archive.writestr(name, data)
            manifest["files"][name] = {"bytes": len(data), "sha256": sha256(data)}
        for relative in sorted([*bundle["files"], "bundle.json"]):
            path = args.inputs / relative
            name = "inputs/" + relative
            archive.write(path, name)
            manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        archive.writestr("handoff.json", canonical_json(manifest))
    with zipfile.ZipFile(args.output) as archive:
        if set(archive.namelist()) != set(manifest["files"]) | {"handoff.json"}:
            raise ValueError("Unexpected archive members")
        for name, record in manifest["files"].items():
            data = archive.read(name)
            if len(data) != record["bytes"] or sha256(data) != record["sha256"]:
                raise ValueError("Archive readback mismatch")
    print(
        json.dumps(
            {
                "source_commit": commit,
                "files": len(manifest["files"]),
                "bytes": args.output.stat().st_size,
                "sha256": file_hash(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
