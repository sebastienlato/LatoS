"""Train a fresh BPE on verified training text; record a reproducible local artifact."""

import json
import platform
import shutil
import tempfile
from pathlib import Path

import tokenizers
from tokenizers import AddedToken, pre_tokenizers, trainers

from latos import __version__
from latos.data.manifest import canonical_json, sha256
from latos.tokenization.core import CONTRACT, SPECIAL_TOKENS, LatoTokenizer, load_config, new_engine
from latos.tokenization.corpus import read_split


def measure(codec: LatoTokenizer, texts: list[str]) -> dict:
    tokens = characters = utf8_bytes = 0
    for text in texts:
        ids = codec.encode(text)
        if codec.decode(ids) != text:
            raise ValueError("Tokenizer round-trip mismatch")
        tokens += len(ids)
        characters += len(text)
        utf8_bytes += len(text.encode("utf-8"))
    return {
        "records": len(texts),
        "characters": characters,
        "utf8_bytes": utf8_bytes,
        "tokens": tokens,
        "bytes_per_token": utf8_bytes / tokens if tokens else None,
        "characters_per_token": characters / tokens if tokens else None,
        "round_trip_failures": 0,
        "unknown_tokens": 0,
        "boundary_tokens_included": False,
    }


def implementation_hash() -> str:
    root = Path(__file__).parent
    return sha256(
        b"".join((root / name).read_bytes() for name in ("core.py", "corpus.py", "training.py"))
    )


def train(manifest_path: Path, corpus_dir: Path, config_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError("Tokenizer output already exists; choose a new destination")
    config = load_config(config_path)
    # There is deliberately no train-split argument and no read of validation/test files.
    texts, corpus = read_split(manifest_path, corpus_dir, "train")
    engine = new_engine()
    trainer = trainers.BpeTrainer(
        vocab_size=config["vocab_size"],
        min_frequency=config["min_frequency"],
        max_token_length=config["max_token_length"],
        show_progress=False,
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
        special_tokens=[AddedToken(t, normalized=False, special=True) for t in SPECIAL_TOKENS],
    )
    engine.train_from_iterator(iter(texts), trainer=trainer, length=len(texts))
    state = json.loads(engine.to_str())
    data = canonical_json(state)
    codec = LatoTokenizer(engine)
    metadata = {
        "schema_version": 1,
        "contract": CONTRACT,
        "config": config,
        "implementation_sha256": implementation_hash(),
        "latos_version": __version__,
        "python_version": platform.python_version(),
        "tokenizers_version": tokenizers.__version__,
        "corpus": corpus,
        "tokenizer_sha256": sha256(data),
        "vocab_size": codec.vocab_size,
        "vocab_sha256": sha256(canonical_json(state["model"]["vocab"])),
        "merges_sha256": sha256(canonical_json(state["model"]["merges"])),
        "merges": len(state["model"]["merges"]),
        "training_metrics": measure(codec, texts),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".latos-tokenizer-", dir=output_dir.parent))
    try:
        (staging / "tokenizer.json").write_bytes(data)
        (staging / "metadata.json").write_bytes(canonical_json(metadata))
        loaded = LatoTokenizer.load(staging, expected_sha256=metadata["tokenizer_sha256"])
        for text in texts:
            if loaded.encode(text) != codec.encode(text):
                raise ValueError("Tokenizer encoding changed after save/load")
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return metadata


def evaluate(manifest_path: Path, corpus_dir: Path, artifact_dir: Path, split: str) -> dict:
    codec = LatoTokenizer.load(artifact_dir)
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    texts, corpus = read_split(manifest_path, corpus_dir, split)
    if corpus["corpus_report_sha256"] != metadata["corpus"]["corpus_report_sha256"]:
        raise ValueError("Evaluation corpus differs from the tokenizer's recorded corpus")
    return {
        "schema_version": 1,
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "corpus": corpus,
        "metrics": measure(codec, texts),
    }
