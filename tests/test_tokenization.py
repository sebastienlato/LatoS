"""Train-only fitting, lossless text, boundary semantics, and artifact integrity."""

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from latos.data.acquire import acquire
from latos.data.manifest import canonical_json, sha256
from latos.data.prepare import prepare
from latos.tokenization import SPECIAL_TOKENS, LatoTokenizer
from latos.tokenization.core import validate_config
from latos.tokenization.corpus import read_split
from latos.tokenization.training import evaluate, train

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/fixtures/tiny/manifest.json"
CONFIG = ROOT / "configs/tokenizer/debug.json"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("tokenizer")
    acquire(MANIFEST, root / "raw")
    prepare(MANIFEST, root / "raw", root / "corpus")
    metadata = train(MANIFEST, root / "corpus", CONFIG, root / "artifact")
    return root, metadata


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "   leading and trailing   ",
        "\tline one\r\nline two\n",
        "Café cafe\u0301 — naïve résumé",
        "日本語 العربية हिन्दी 한국어",
        "🧑🏽‍🔬👨‍👩‍👧‍👦🇨🇦",
        "\x00\x01\x7f\ufeff\u200b",
        "<|pad|><|bos|><|eos|><|unk|>",
        "Literal <|eos|> is text, not an instruction.",
        '1234567890 + 3.14159 =?\n"quoted" \\ path',
        "x" * 10000,
    ],
)
def test_exact_round_trips(prepared, text):
    root, _ = prepared
    codec = LatoTokenizer.load(root / "artifact")
    ids = codec.encode(text)
    assert not set(ids) & set(SPECIAL_TOKENS.values())
    assert codec.decode(ids) == text


def test_seeded_unicode_round_trips(prepared):
    root, _ = prepared
    codec = LatoTokenizer.load(root / "artifact")
    rng = random.Random(204)
    for _ in range(100):
        points = [rng.randrange(0x110000) for _ in range(30)]
        text = "".join(chr(c) for c in points if not 0xD800 <= c <= 0xDFFF)
        assert codec.decode(codec.encode(text)) == text
    text = "".join(chr(i) for i in range(256))
    assert codec.decode(codec.encode(text)) == text


def test_lone_surrogate_is_rejected(prepared):
    root, _ = prepared
    with pytest.raises(UnicodeEncodeError):
        LatoTokenizer.load(root / "artifact").encode("\ud800")


def test_explicit_boundaries_and_padding_id(prepared):
    root, _ = prepared
    codec = LatoTokenizer.load(root / "artifact")
    assert SPECIAL_TOKENS == {"<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2, "<|unk|>": 3}
    ids = codec.encode("Hello", add_bos=True, add_eos=True)
    assert ids[0] == 1 and ids[-1] == 2
    assert codec.decode([0, *ids, 0]) == "Hello"
    assert codec.decode(ids, skip_special_tokens=False) == "<|bos|>Hello<|eos|>"
    assert codec.encode("", add_bos=True, add_eos=True) == [1, 2]


@pytest.mark.parametrize("bad", [[-1], [True], [1.5], [1000000]])
def test_invalid_ids_are_rejected(prepared, bad):
    root, _ = prepared
    with pytest.raises(ValueError, match="within the learned vocabulary"):
        LatoTokenizer.load(root / "artifact").decode(bad)


def test_loader_restores_runtime_special_token_policy(prepared):
    root, _ = prepared
    raw = Tokenizer.from_file(str(root / "artifact/tokenizer.json"))
    assert raw.encode_special_tokens is False
    codec = LatoTokenizer.load(root / "artifact")
    assert 2 not in codec.encode("<|eos|>")
    assert codec.decode(codec.encode("<|eos|>")) == "<|eos|>"


def test_training_does_not_open_held_out_files(prepared, tmp_path, monkeypatch):
    root, original = prepared
    corpus = tmp_path / "corpus"
    shutil.copytree(root / "corpus", corpus)
    (corpus / "validation.jsonl").unlink()
    (corpus / "test.jsonl").unlink()
    real_open = Path.open

    def guarded(path, *args, **kwargs):
        assert path.name not in ("validation.jsonl", "test.jsonl"), (
            "Held-out text opened during fit"
        )
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    result = train(MANIFEST, corpus, CONFIG, tmp_path / "artifact")
    assert result == original
    for name in ("metadata.json", "tokenizer.json"):
        assert (root / "artifact" / name).read_bytes() == (
            tmp_path / "artifact" / name
        ).read_bytes()
    assert result["corpus"]["split"] == "train"
    assert result["corpus"]["source_ids"] == ["fixture-train"]


def test_train_rejects_held_out_record_even_with_updated_hash(prepared, tmp_path):
    root, _ = prepared
    corpus = tmp_path / "corpus"
    shutil.copytree(root / "corpus", corpus)
    path = corpus / "train.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["split"] = "test"
    data = b"".join(canonical_json(r) for r in records)
    path.write_bytes(data)
    report_path = corpus / "report.json"
    report = json.loads(report_path.read_text())
    report["splits"]["train"].update(sha256=sha256(data), bytes=len(data))
    report_path.write_bytes(canonical_json(report))
    with pytest.raises(ValueError, match="not assigned"):
        train(MANIFEST, corpus, CONFIG, tmp_path / "artifact")
    assert not (tmp_path / "artifact").exists()


def test_vocabulary_and_merge_identity(prepared):
    root, metadata = prepared
    codec = LatoTokenizer.load(root / "artifact", expected_sha256=metadata["tokenizer_sha256"])
    state = json.loads((root / "artifact/tokenizer.json").read_text(encoding="utf-8"))
    assert codec.vocab_size <= 512
    assert codec.vocab_size == 260 + len(state["model"]["merges"])
    assert metadata["vocab_sha256"] == sha256(canonical_json(state["model"]["vocab"]))
    assert metadata["training_metrics"]["round_trip_failures"] == 0


def test_corrupt_artifact_fails_before_parse(prepared, tmp_path):
    root, _ = prepared
    target = tmp_path / "artifact"
    shutil.copytree(root / "artifact", target)
    (target / "tokenizer.json").write_text("not JSON")
    with pytest.raises(ValueError, match="checksum"):
        LatoTokenizer.load(target)


def test_contract_change_is_rejected_even_with_updated_hash(prepared, tmp_path):
    root, _ = prepared
    target = tmp_path / "artifact"
    shutil.copytree(root / "artifact", target)
    state = json.loads((target / "tokenizer.json").read_text(encoding="utf-8"))
    state["normalizer"] = {"type": "Lowercase"}
    data = canonical_json(state)
    (target / "tokenizer.json").write_bytes(data)
    metadata = json.loads((target / "metadata.json").read_text())
    metadata["tokenizer_sha256"] = sha256(data)
    (target / "metadata.json").write_bytes(canonical_json(metadata))
    with pytest.raises(ValueError, match="normalizer"):
        LatoTokenizer.load(target)


def test_expected_hash_and_existing_output(prepared):
    root, _ = prepared
    with pytest.raises(ValueError, match="checksum"):
        LatoTokenizer.load(root / "artifact", expected_sha256="0" * 64)
    before = (root / "artifact/tokenizer.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        train(MANIFEST, root / "corpus", CONFIG, root / "artifact")
    assert (root / "artifact/tokenizer.json").read_bytes() == before


def test_declared_merge_count_is_checked(prepared, tmp_path):
    root, _ = prepared
    target = tmp_path / "artifact"
    shutil.copytree(root / "artifact", target)
    metadata = json.loads((target / "metadata.json").read_text())
    metadata["merges"] += 1
    (target / "metadata.json").write_bytes(canonical_json(metadata))
    with pytest.raises(ValueError, match="Merge count"):
        LatoTokenizer.load(target)


def test_corrupt_training_file_fails_before_fit(prepared, tmp_path):
    root, _ = prepared
    corpus = tmp_path / "corpus"
    shutil.copytree(root / "corpus", corpus)
    (corpus / "train.jsonl").write_bytes(b"changed input")
    with pytest.raises(ValueError, match="checksum"):
        train(MANIFEST, corpus, CONFIG, tmp_path / "artifact")
    assert not (tmp_path / "artifact").exists()


def test_evaluation_matches_independent_counts(prepared):
    root, _ = prepared
    result = evaluate(MANIFEST, root / "corpus", root / "artifact", "validation")
    texts, _ = read_split(MANIFEST, root / "corpus", "validation")
    raw = Tokenizer.from_file(str(root / "artifact/tokenizer.json"))
    tokens = sum(len(raw.encode(text).ids) for text in texts)
    assert result["metrics"]["tokens"] == tokens
    assert result["metrics"]["bytes_per_token"] == sum(len(t.encode()) for t in texts) / tokens
    assert result["corpus"]["split"] == "validation"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("vocab_size", 259),
        ("vocab_size", 999999),
        ("min_frequency", 0),
        ("max_token_length", 0),
        ("vocab_size", True),
        ("extra", 1),
    ],
)
def test_invalid_configs(key, value):
    config = json.loads(CONFIG.read_text())
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_cli_encode(prepared):
    root, _ = prepared
    text = "Whitespace\tand <|eos|> text 😊"
    result = subprocess.run(
        [
            "latos",
            "tokenizer",
            "encode",
            "--artifact-dir",
            str(root / "artifact"),
            "--text",
            text,
            "--bos",
            "--eos",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["decoded"] == text and data["ids"][0] == 1 and data["ids"][-1] == 2
