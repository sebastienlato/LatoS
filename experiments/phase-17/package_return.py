"""Return compact evidence plus the final learned model; preserve all originals."""

import argparse
import json
import zipfile
from pathlib import Path

from latos.data.manifest import canonical_json
from latos.model.storage import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Return already exists")
    records, included, omitted = {}, {}, {}
    for prefix, root in [("attempt", args.attempt), ("validation", args.validation)]:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.is_symlink():
                raise ValueError("Symlink in return evidence")
            relative = path.relative_to(root).as_posix()
            name = prefix + "/" + relative
            record = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
            final_model = relative == "checkpoints/step-00007485/model/model.safetensors"
            if path.suffix in (".safetensors", ".u32", ".u64") and not final_model:
                omitted[name] = record
            else:
                included[name] = path
                records[name] = record
    for name in ("handoff.json", "preflight.json"):
        path = args.transfer / name
        if path.exists():
            included[name] = path
            records[name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    manifest = {
        "schema_version": 1,
        "scope": "Evidence subset plus final model if produced; NOT a complete checkpoint backup",
        "files": records,
        "omitted_retained_on_Windows": omitted,
        "phase17_complete": False,
    }
    with zipfile.ZipFile(
        args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, path in included.items():
            archive.write(path, name)
        archive.writestr("return.json", canonical_json(manifest))
    with zipfile.ZipFile(args.output) as archive:
        for name, record in records.items():
            import hashlib

            with archive.open(name) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != record["sha256"]:
                raise ValueError("Return archive readback mismatch")
    print(
        json.dumps(
            {
                "files": len(records),
                "omitted": len(omitted),
                "bytes": args.output.stat().st_size,
                "sha256": file_hash(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
