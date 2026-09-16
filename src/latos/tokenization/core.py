"""Lossless byte-level text encoding with explicit boundary IDs and verified loading."""

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from latos.data.manifest import canonical_json, sha256

SPECIAL_TOKENS = {"<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2, "<|unk|>": 3}
CONTRACT = {
    "version": 1,
    "normalization": "identity; valid Unicode scalar strings encoded as UTF-8",
    "literal_special_tokens": "ordinary text; no control IDs from input strings",
    "boundaries": "explicit BOS/EOS flags only; default neither",
    "padding": "none",
    "truncation": "none",
    "special_tokens": SPECIAL_TOKENS,
}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    expected = {"schema_version", "vocab_size", "min_frequency", "max_token_length"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("Tokenizer config must contain exactly the four versioned fields")
    if any(type(value) is not int for value in config.values()) or config["schema_version"] != 1:
        raise ValueError("Tokenizer config requires integer fields and schema_version 1")
    if not 260 <= config["vocab_size"] <= 32768:
        raise ValueError("Vocabulary budget must be between 260 and 32768")
    if not 1 <= config["min_frequency"] <= 1000 or not 8 <= config["max_token_length"] <= 128:
        raise ValueError("Invalid frequency or maximum token length")


def new_engine() -> Tokenizer:
    engine = Tokenizer(models.BPE(unk_token="<|unk|>"))
    engine.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    engine.decoder = decoders.ByteLevel()
    engine.encode_special_tokens = True
    return engine


class LatoTokenizer:
    """Public codec; input text never implicitly inserts a special-token ID."""

    def __init__(self, engine: Tokenizer):
        self._engine = engine
        # Tokenizers does not serialize this runtime setting in tokenizer.json.
        self._engine.encode_special_tokens = True
        self.vocab_size = engine.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        text.encode("utf-8", errors="strict")  # Reject lone surrogates instead of losing text.
        ids = self._engine.encode(text, add_special_tokens=False).ids
        if any(token in SPECIAL_TOKENS.values() for token in ids):
            raise ValueError("Text encoding unexpectedly emitted a reserved token")
        return ([1] if add_bos else []) + ids + ([2] if add_eos else [])

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        if any(type(token) is not int or not 0 <= token < self.vocab_size for token in ids):
            raise ValueError("Token IDs must be integers within the learned vocabulary")
        return self._engine.decode(ids, skip_special_tokens=skip_special_tokens)

    @classmethod
    def load(cls, directory: Path, *, expected_sha256: str | None = None) -> LatoTokenizer:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        with (directory / "tokenizer.json").open("rb") as stream:
            data = stream.read(10_000_001)
        if len(data) > 10_000_000:
            raise ValueError("Tokenizer artifact exceeds the size budget")
        digest = sha256(data)
        if digest != metadata["tokenizer_sha256"] or (
            expected_sha256 is not None and digest != expected_sha256
        ):
            raise ValueError("Tokenizer checksum mismatch")
        if metadata["schema_version"] != 1 or metadata["contract"] != CONTRACT:
            raise ValueError("Unsupported tokenizer contract")
        validate_config(metadata["config"])
        state = json.loads(data)
        baseline = json.loads(new_engine().to_str())
        for key in (
            "normalizer",
            "pre_tokenizer",
            "decoder",
            "post_processor",
            "padding",
            "truncation",
        ):
            if state[key] != baseline[key]:
                raise ValueError(f"Unexpected tokenizer component: {key}")
        model = state["model"]
        for key, value in baseline["model"].items():
            if key not in ("vocab", "merges") and model[key] != value:
                raise ValueError(f"Unexpected BPE setting: {key}")
        vocab = model["vocab"]
        if len(vocab) != metadata["vocab_size"] or set(vocab.values()) != set(range(len(vocab))):
            raise ValueError("Vocabulary IDs are missing, repeated, or noncontiguous")
        if not 260 <= len(vocab) <= metadata["config"]["vocab_size"]:
            raise ValueError("Actual vocabulary is outside the configured budget")
        if metadata["merges"] != len(model["merges"]) or len(vocab) != 260 + len(model["merges"]):
            raise ValueError("Merge count does not explain the learned vocabulary")
        if any(vocab.get(token) != token_id for token, token_id in SPECIAL_TOKENS.items()):
            raise ValueError("Special-token identity mismatch")
        if not set(pre_tokenizers.ByteLevel.alphabet()).issubset(vocab):
            raise ValueError("Incomplete byte alphabet")
        expected_added = [
            {
                "id": i,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for token, i in SPECIAL_TOKENS.items()
        ]
        if state["added_tokens"] != expected_added:
            raise ValueError("Unexpected added tokens")
        if (
            sha256(canonical_json(vocab)) != metadata["vocab_sha256"]
            or sha256(canonical_json(model["merges"])) != metadata["merges_sha256"]
        ):
            raise ValueError("Vocabulary or merge checksum mismatch")
        return cls(Tokenizer.from_str(data.decode("utf-8")))
