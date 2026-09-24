"""Pack diagnostic evidence only; retain all synthetic checkpoints on Windows."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from run import artifact_bytes, hash_file, verify_bundle, write_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.run.resolve(), args.output.resolve()
    bundle = Path(__file__).resolve().parent
    plan, manifest = verify_bundle(bundle)
    if output.exists() or output.is_relative_to(root) or output.is_relative_to(bundle):
        raise ValueError("Use a new archive outside run/bundle roots")
    paths = {"run/" + p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}
    paths.update(
        {"supervisor/" + p.name: p for p in root.parent.glob(root.name + ".*") if p.is_file()}
    )
    paths.update(
        {
            "bundle/" + p.relative_to(bundle).as_posix(): p
            for p in bundle.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }
    )
    included, omitted = {}, {}
    for name, path in paths.items():
        if path.is_symlink():
            raise ValueError("Symlink in diagnostic evidence")
        record = {"bytes": path.stat().st_size, "sha256": hash_file(path)}
        target = omitted if path.suffix in (".safetensors", ".u32", ".u64") else included
        target[name] = record
    if sum(r["bytes"] for r in included.values()) > plan["administrative_artifact_reserve_bytes"]:
        raise ValueError("Evidence return exceeds its reserved artifact allowance")
    if artifact_bytes(root, bundle, plan) > plan["max_artifact_bytes"]:
        raise ValueError("Diagnostic artifact bound exceeded")
    record = {
        "files": included,
        "omitted_retained_on_Windows": omitted,
        "diagnostic_source_commit": manifest["source_commit"],
        "scope": "Synthetic mechanics evidence subset; not a checkpoint backup or Phase 17.1 model",
        "phase17_1_training_authorized": False,
        "publication_authorized": False,
    }
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in included:
            archive.write(paths[name], name)
        archive.writestr("return.json", json.dumps(record, sort_keys=True))
    with zipfile.ZipFile(output) as archive:
        for name, expected in included.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
                    raise ValueError("Return readback mismatch")
    receipt = {
        "sha256": hash_file(output),
        "bytes": output.stat().st_size,
        "included": len(included),
        "omitted_retained": len(omitted),
        "diagnostic_source_commit": manifest["source_commit"],
        "scope": record["scope"],
    }
    write_once(output.with_suffix(".receipt.json"), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
