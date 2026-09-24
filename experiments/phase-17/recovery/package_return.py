"""Pack reviewed recovery evidence and final model; never discard Windows originals."""

import argparse
import json
import zipfile
from pathlib import Path

from recover import digest, read, write_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    transfer = args.transfer.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Return exists; use a new name")
    roots = {
        "attempt": transfer / "source/outputs/phase17-cuda-attempt1",
        "validation": transfer / "validation",
        "recovery": transfer / "phase17-reviewed-recovery",
        "recovery-bundle": Path(__file__).resolve().parent,
    }
    included, omitted, paths = {}, {}, {}
    for prefix, root in roots.items():
        if output.is_relative_to(root):
            raise ValueError("Place return outside the collected roots")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                raise ValueError("Symlink in evidence")
            name = prefix + "/" + path.relative_to(root).as_posix()
            record = {"bytes": path.stat().st_size, "sha256": digest(path)}
            if (
                path.suffix in (".safetensors", ".u32", ".u64")
                and name != "attempt/checkpoints/step-00007485/model/model.safetensors"
            ):
                omitted[name] = record
            else:
                included[name], paths[name] = record, path
    for name in ("handoff.json", "preflight.json"):
        path = transfer / name
        included[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        paths[name] = path
    manifest = {
        "schema_version": 1,
        "files": included,
        "omitted_retained_on_Windows": omitted,
        "phase17_complete": False,
        "scope": "Original failed attempt plus reviewed recovery/evaluation; "
        "final model if produced. "
        "Not a full checkpoint backup.",
    }
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in paths.items():
            archive.write(path, name)
        archive.writestr("return.json", json.dumps(manifest, sort_keys=True))
    with zipfile.ZipFile(output) as archive:
        import hashlib

        for name, record in included.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                    raise ValueError("Return readback mismatch")
    receipt = {
        "archive_sha256": digest(output),
        "archive_bytes": output.stat().st_size,
        "included_files": len(included),
        "omitted_retained_files": len(omitted),
        "original_ledger_sha256": digest(roots["attempt"] / "ledger.json"),
        "original_failure_ledger_sha256": digest(roots["attempt"] / "ledger.json.tmp"),
        "scope": manifest["scope"],
        "training_source_commit": read(transfer / "handoff.json")["source_commit"],
    }
    write_once(output.with_suffix(".receipt.json"), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
