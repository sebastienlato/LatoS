"""Training-only candidate fitting and development-only codec selection."""

import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tokenizers
from tokenizers import AddedToken, pre_tokenizers, trainers

from latos.chat import format_chat
from latos.data.manifest import canonical_json, sha256
from latos.data2.acquire import file_hash, write_json
from latos.data2.filtering import validate_messages
from latos.data2.lexical import digest
from latos.data2.prepare import split_for, text_of
from latos.tokenization.core import CONTRACT, SPECIAL_TOKENS, LatoTokenizer, new_engine
from latos.tokenization.training import measure


def read_records(corpus: Path, kind: str, split: str):
    if kind not in ("pretrain", "instruction") or split not in ("train", "development"):
        raise ValueError("Tokenizer access limited to new train/development")
    report = json.loads((corpus / "report.json").read_text())
    name = f"{kind}-{split}.jsonl"
    path = corpus / name
    if file_hash(path) != report["files"][name]["sha256"]:
        raise ValueError("Corpus identity mismatch")
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if (
                row["split"] != split
                or split_for(row["group_id"]) != split
                or row["kind"] != kind
                or row["content_sha256"] != digest(text_of(row))
            ):
                raise ValueError("Record identity mismatch")
            if kind == "instruction":
                validate_messages(row["messages"])
            yield row


def fit_text(row):
    if row["kind"] == "pretrain":
        return row["text"]
    return "".join(f"\n{m['role'].title()}:\n{m['content']}" for m in row["messages"])


def training_sample(corpus: Path):
    texts, identities = [], []
    for kind, budget in (("pretrain", 24_000_000), ("instruction", 8_000_000)):
        rows = sorted(
            read_records(corpus, kind, "train"), key=lambda r: digest("data2-fit-v1:" + r["id"])
        )
        size = 0
        for row in rows:
            text = fit_text(row)
            n = len(text.encode())
            if size + n <= budget:
                texts.append(text)
                identities.append({"id": row["id"], "sha256": digest(text)})
                size += n
    return texts, identities


def fit(corpus: Path, destination: Path, vocab: int):
    if vocab not in (8192, 16384, 512):
        raise ValueError("Only bounded declared candidates or a test fixture are permitted")
    destination.mkdir(parents=True, exist_ok=False)
    texts, identities = training_sample(corpus)
    if not texts:
        raise ValueError("Empty tokenizer fitting input")
    config = {"schema_version": 1, "vocab_size": vocab, "min_frequency": 2, "max_token_length": 32}
    engine = new_engine()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        min_frequency=2,
        max_token_length=32,
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
        "python_version": platform.python_version(),
        "tokenizers_version": tokenizers.__version__,
        "corpus": {
            "report_sha256": file_hash(corpus / "report.json"),
            "fitting_split": "train",
            "sample_identity_sha256": sha256(canonical_json(identities)),
            "sample_records": len(texts),
        },
        "tokenizer_sha256": sha256(data),
        "vocab_size": codec.vocab_size,
        "vocab_sha256": sha256(canonical_json(state["model"]["vocab"])),
        "merges_sha256": sha256(canonical_json(state["model"]["merges"])),
        "merges": len(state["model"]["merges"]),
        "training_metrics": measure(codec, texts),
    }
    (destination / "tokenizer.json").write_bytes(data)
    (destination / "metadata.json").write_bytes(canonical_json(metadata))
    write_json(destination / "fitting-records.json", identities)
    loaded = LatoTokenizer.load(destination)
    if any(loaded.encode(t) != codec.encode(t) for t in texts):
        raise ValueError("Save/load changed fitting encodings")
    return metadata


def distribution(lengths):
    a = np.asarray(lengths, dtype=np.int64)
    return {
        "count": len(lengths),
        "min": int(a.min()) if len(a) else None,
        "p50": float(np.quantile(a, 0.5)) if len(a) else None,
        "p95": float(np.quantile(a, 0.95)) if len(a) else None,
        "p99": float(np.quantile(a, 0.99)) if len(a) else None,
        "max": int(a.max()) if len(a) else None,
    }


def codec_checks(codec):
    examples = [
        "",
        " \t\n  end  ",
        "Café Cafe\u0301 東京 مرحبا 🧪",
        "<|pad|><|bos|><|eos|><|unk|>",
        "x\r\ny\u00a0z",
    ]
    for text in examples:
        ids = codec.encode(text)
        assert codec.decode(ids) == text and not set(ids).intersection(range(4))
        assert codec.encode(text, add_bos=True, add_eos=True) == [1] + ids + [2]
    messages = [{"role": "user", "content": "A\tB?"}, {"role": "assistant", "content": "A\nB."}]
    full, mask = format_chat(codec, messages, max_length=256)
    prompt, _ = format_chat(codec, messages[:1], max_length=256, generation_prompt=True)
    assert (
        full[: len(prompt)] == prompt and sum(mask) == len(codec.encode(messages[1]["content"])) + 1
    )
    from latos.evaluation.scoring import byte_spans

    for span in byte_spans(("Café 東京 🧪 <|eos|> \t\n" * 100), 192):
        encoded = codec.encode(span, add_bos=True)
        assert len(encoded) <= 193 and codec.decode(encoded) == span
    return {
        "round_trip_examples": len(examples),
        "chat_contract": 1,
        "fixed_192_byte_BPB_compatible": True,
        "passed": True,
    }


def analyze(corpus: Path, artifact: Path, split: str):
    codec = LatoTokenizer.load(artifact)
    stats, lengths, usage = (
        defaultdict(Counter),
        defaultdict(list),
        np.zeros(codec.vocab_size, dtype=np.int64),
    )
    distinct_text = set()
    for kind in ("pretrain", "instruction"):
        for row in read_records(corpus, kind, split):
            content_parts = (
                [row["text"]] if kind == "pretrain" else [m["content"] for m in row["messages"]]
            )
            encoded = []
            for part in content_parts:
                ids = codec.encode(part)
                if codec.decode(ids) != part:
                    raise ValueError("Lossless encoding failed")
                usage += np.bincount(ids, minlength=codec.vocab_size)
                encoded.append(ids)
                fingerprint = digest(part)
                if fingerprint not in distinct_text:
                    stats["unique_exact_text_segments"]["tokens"] += len(ids)
                    stats["unique_exact_text_segments"]["bytes"] += len(part.encode())
                    distinct_text.add(fingerprint)
            tokens = sum(map(len, encoded))
            nbytes = sum(len(t.encode()) for t in content_parts)
            keys = [kind, row["source"], row["source"] + "/" + row["category"]]
            for key in keys:
                stats[key]["records"] += 1
                stats[key]["content_tokens"] += tokens
                stats[key]["utf8_bytes"] += nbytes
            if kind == "instruction":
                total = 1 + sum(
                    len(codec.encode(f"\n{m['role'].title()}:\n")) + len(ids) + 1
                    for m, ids in zip(row["messages"], encoded, strict=True)
                )
                targets = sum(
                    len(ids) + 1
                    for m, ids in zip(row["messages"], encoded, strict=True)
                    if m["role"] == "assistant"
                )
                stats[kind]["assistant_targets"] += targets
                stats[kind]["serialized_tokens"] += total
                if total <= 4096:
                    ids, mask = format_chat(codec, row["messages"], max_length=4096)
                    assert len(ids) == total and sum(mask) == targets
                for context in (256, 512, 1024, 2048, 4096):
                    if total > context:
                        stats[kind][f"overflow_{context}"] += 1
                    else:
                        stats[kind][f"fits_{context}"] += 1
                        stats[kind][f"targets_at_{context}"] += targets
                        stats[kind][f"padding_at_{context}"] += context - total
                lengths[kind].append(total)
            else:
                lengths[kind].append(tokens + 2)
                for context in (256, 512, 1024, 2048):
                    # Historical isolated-window objective: BOS/EOS and overlap one.
                    windows = (tokens + 1 + context - 2) // (context - 1)
                    stats[kind][f"windows_at_{context}"] += windows
                    stats[kind][f"padding_at_{context}"] += windows * (context - 1) - (tokens + 1)
    for counter in stats.values():
        if counter.get("content_tokens"):
            counter["bytes_per_token"] = counter["utf8_bytes"] / counter["content_tokens"]
    return {
        "split": split,
        "tokenizer_sha256": file_hash(artifact / "tokenizer.json"),
        "vocab_size": codec.vocab_size,
        "stats": dict(stats),
        "lengths": {k: distribution(v) for k, v in lengths.items()},
        "used_content_vocabulary": int(np.count_nonzero(usage)),
        "unused_content_vocabulary": int(np.count_nonzero(usage[4:] == 0)),
        "contract_checks": codec_checks(codec),
        "reserved_text_access": False,
    }
