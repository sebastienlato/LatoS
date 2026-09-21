"""Package only the reviewed tiny fixture artifacts and explicit release notices."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = {
    "model/config.json": "final/model/config.json",
    "model/metadata.json": "final/model/metadata.json",
    "model/model.safetensors": "final/model/model.safetensors",
    "tokenizer/metadata.json": "tokenizer/metadata.json",
    "tokenizer/tokenizer.json": "tokenizer/tokenizer.json",
}
NOTICES = {
    "LICENSE": "LICENSE",
    "MODEL_CARD.md": "docs/MODEL_CARD.md",
    "DATA_CARD.md": "docs/DATA_CARD.md",
}


def identity(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read_regular(root, relative):
    path = root / relative
    # Reject symlinked files and intermediate directories, not just the leaf.
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError(f"Symlink is not a release input: {relative}")
        if item == root:
            break
    if not path.is_file():
        raise ValueError(f"Missing regular release input: {relative}")
    return path.read_bytes()


def package(tiny_dir, output, *, root=ROOT):
    inventory = json.loads(read_regular(root, "experiments/phase-8/tiny-artifacts.json"))
    if inventory.keys() != ARTIFACTS.keys():
        raise ValueError("Tiny release inventory differs from the explicit allowlist")
    payloads = {}
    for name, source in ARTIFACTS.items():
        data = read_regular(tiny_dir, source)
        if identity(data) != inventory[name]:
            raise ValueError(f"Release artifact identity mismatch: {name}")
        payloads[name] = data
    for name, source in NOTICES.items():
        payloads[name] = read_regular(root, source)
    payloads["MANIFEST.json"] = (
        json.dumps(
            {
                "schema_version": 1,
                "name": "LatoS 1.0.0 tiny fixture",
                "license": "MIT",
                "purpose": "Fixture memorization/mechanics only; not the English base or SFT",
                "reproduction": "experiments/phase-4/validate.py, 400 CPU updates, seed 17",
                "members": {k: identity(v) for k, v in sorted(payloads.items())},
            },
            indent=2,
        )
        + "\n"
    ).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 21, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    # Exclusive creation: no overwrite; validate all inputs before creating output.
    with output.open("xb") as stream:
        stream.write(buffer.getvalue())
    return identity(buffer.getvalue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.tiny_dir, args.output), indent=2))
