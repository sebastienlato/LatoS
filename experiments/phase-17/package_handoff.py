"""Reviewed source plus explicit pinned inputs; never archive a whole workspace."""

import argparse
import io
import json
import subprocess
import zipfile
from pathlib import Path

from latos.data.manifest import canonical_json, sha256
from latos.model.storage import file_hash, load_model
from latos.tokenization import LatoTokenizer
from latos.training.base2 import BASE_HASH, frozen
from latos.training.probe import verify_inputs


def evaluation_files(root):
    """Opaque-copy reserved LM test only; no parse or scoring in this packager."""
    base = root / "outputs/phase-5-english-pilot/step-00003000/model"
    if file_hash(base / "model.safetensors") != BASE_HASH:
        raise ValueError("Historical baseline weights mismatch")
    model = load_model(base)
    tokenizer = root / "artifacts/tokenizers/english-bpe-v1"
    LatoTokenizer.load(tokenizer)
    if file_hash(tokenizer / "tokenizer.json") != model.config.tokenizer_sha256:
        raise ValueError("Historical tokenizer mismatch")
    files = {
        f"evaluation/base/{name}": base / name
        for name in ("config.json", "metadata.json", "model.safetensors")
    }
    files.update(
        {
            f"evaluation/tokenizer/{name}": tokenizer / name
            for name in ("metadata.json", "tokenizer.json")
        }
    )
    corpus = root / "data/processed/english-books-v1"
    protocol = json.loads((root / "configs/evaluation/protocol-v1.json").read_text())
    for split in ("validation", "test"):
        path = corpus / f"{split}.jsonl"
        if file_hash(path) != protocol[f"lm_{split}_sha256"]:
            raise ValueError("Historical evaluation split mismatch")
        files[f"evaluation/corpus/{split}.jsonl"] = path
    files["evaluation/corpus/report.json"] = corpus / "report.json"
    manifest = json.loads((root / "configs/evaluation/external-v1.json").read_text())
    for record in manifest["datasets"]:
        path = root / "data/cache/evaluation-v1" / record["file"]
        if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
            raise ValueError("Pinned ARC input mismatch")
        files["evaluation/external/" + record["file"]] = path
    return files


def package(root, inputs, output):
    if output.exists():
        raise ValueError("Handoff exists")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Reviewed clean source commit required")
    frozen(root)
    verify_inputs(inputs)
    extras = evaluation_files(root)
    bundle = json.loads((inputs / "bundle.json").read_text())
    extras.update({"inputs/" + name: inputs / name for name in [*bundle["files"], "bundle.json"]})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    source = subprocess.check_output(["git", "archive", "--format=zip", commit], cwd=root)
    manifest = {
        "schema_version": 1,
        "phase": 17,
        "execution_authorized": True,
        "source_commit": commit,
        "phase_complete": False,
        "publication_authorized": False,
        "phase18_authorized": False,
        "scope": "One frozen RTX 4070 SUPER attempt and paired development evaluation. "
        "Final requires reviewed declaration.",
        "reserved_handling": "Original LM test is opaque-copied for gated final acceptance only; "
        "never training/monitoring.",
        "files": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        zipfile.ZipFile(io.BytesIO(source)) as reviewed,
        zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive,
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
        for name, path in sorted(extras.items()):
            archive.write(path, name)
            manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        archive.writestr("handoff.json", canonical_json(manifest))
    with zipfile.ZipFile(output) as archive:
        if set(archive.namelist()) != set(manifest["files"]) | {"handoff.json"}:
            raise ValueError("Archive member mismatch")
        for name, record in manifest["files"].items():
            data = archive.read(name)
            if len(data) != record["bytes"] or sha256(data) != record["sha256"]:
                raise ValueError("Archive readback mismatch")
    return {
        "source_commit": commit,
        "files": len(manifest["files"]),
        "bytes": output.stat().st_size,
        "sha256": file_hash(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(Path.cwd(), args.inputs, args.output), indent=2))


if __name__ == "__main__":
    main()
