"""Checksummed evidence subset; original Windows tensors remain preserved."""

import argparse
import json
import zipfile
from contextlib import ExitStack
from pathlib import Path

from common import (
    ATTEMPT,
    artifact_bytes,
    hash_file,
    lock,
    record,
    require,
    verify_bundle,
    write_once,
)


def package(transfer, destination):
    root = transfer / "phase17-1-results"
    with ExitStack() as stack:
        held = {}
        if root.exists():
            for name in ("supervisor.lock", "worker.lock"):
                path = root / name
                held[path] = stack.enter_context(lock(path))
        return package_locked(transfer, destination, held)


def package_locked(transfer, destination, held):
    bundle = Path(__file__).resolve().parent
    verify_bundle(bundle)
    root = transfer / "phase17-1-results"
    require(
        not destination.exists()
        and not destination.resolve().is_relative_to(root)
        and not destination.resolve().is_relative_to(bundle),
        "Use a new archive outside evidence roots",
    )
    paths = {"run/" + p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}
    paths.update(
        {
            "bundle/" + p.relative_to(bundle).as_posix(): p
            for p in bundle.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts
        }
    )
    if (transfer / "phase17-1-launch.json").exists():
        paths["launch.json"] = transfer / "phase17-1-launch.json"
    for p in transfer.glob("phase17-1-*.log"):
        if p.is_file():
            paths["console/" + p.name] = p
    included, omitted = {}, {}
    final_weights = "run/checkpoints/step-00007485/model/model.safetensors"
    for name, path in paths.items():
        require(
            not path.is_symlink()
            and not any(p in (".private", ".venv", ".wheel-env") for p in Path(name).parts),
            "Private/environment/symlink path in return",
        )
        binary = path.suffix in (".safetensors", ".u32", ".u64") and name != final_weights
        (omitted if binary else included)[name] = record(path, held.get(path))
    require(
        sum(v["bytes"] for v in included.values()) < 1024**3,
        "Return exceeds administrative reserve",
    )
    require(
        artifact_bytes(root, bundle) < 20 * 1024**3, "Artifact ceiling exceeded; preserve originals"
    )
    manifest = {
        "attempt_id": ATTEMPT,
        "bundle_sha256": hash_file(bundle / "bundle.json"),
        "files": included,
        "omitted_retained_on_Windows": omitted,
        "scope": "Evidence subset; new candidate bytes if present, never automatic acceptance",
        "phase18_authorized": False,
        "publication_authorized": False,
    }
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name in included:
            if paths[name] in held:
                stream = held[paths[name]]
                stream.seek(0)
                z.writestr(name, stream.read())
                stream.seek(0)
            else:
                z.write(paths[name], name)
        z.writestr("return.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    import hashlib

    with zipfile.ZipFile(destination) as z:
        for name, expected in included.items():
            with z.open(name) as stream:
                require(
                    hashlib.file_digest(stream, "sha256").hexdigest() == expected["sha256"],
                    "Return readback mismatch",
                )
    receipt = {
        "attempt_id": ATTEMPT,
        "sha256": hash_file(destination),
        "bytes": destination.stat().st_size,
        "included": len(included),
        "omitted_retained": len(omitted),
    }
    write_once(destination.with_suffix(".receipt.json"), receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.transfer.resolve(), args.output.resolve()), indent=2))
