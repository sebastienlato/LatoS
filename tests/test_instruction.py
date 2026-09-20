"""Independent target alignment, split isolation, weighted SFT, and recovery checks."""

import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from latos.chat import format_chat
from latos.data.acquire import acquire
from latos.data.manifest import canonical_json
from latos.data.prepare import prepare
from latos.instruction import load_conversations, prepare_conversations
from latos.model import ModelConfig, create_model
from latos.model.network import causal_loss
from latos.tokenization import LatoTokenizer
from latos.tokenization.training import train
from latos.training import Trainer, TrainingConfig, evaluate
from latos.training.checkpoint import load_checkpoint, save_checkpoint
from latos.training.data import TokenDataset, collate

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "instruction_data", ROOT / "experiments/phase-6/data.py"
)
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def codec(tmp_path_factory):
    path = tmp_path_factory.mktemp("instruction-codec")
    manifest = ROOT / "data/fixtures/tiny/manifest.json"
    acquire(manifest, path / "raw")
    prepare(manifest, path / "raw", path / "corpus")
    metadata = train(
        manifest, path / "corpus", ROOT / "configs/tokenizer/debug.json", path / "codec"
    )
    return LatoTokenizer.load(path / "codec"), metadata["tokenizer_sha256"]


def test_multiturn_mask_and_shared_prompt_prefix(codec):
    codec, _ = codec
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Keep <|eos|> literal. Café?"},
        {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": "Again?"},
        {"role": "assistant", "content": "No."},
    ]
    ids, mask = format_chat(codec, messages, max_length=512)
    expected_ids, expected_mask = [1], [False]
    for message in messages:
        header = codec.encode("\n" + message["role"].title() + ":\n")
        content = codec.encode(message["content"])
        expected_ids += header + content + [2]
        expected_mask += [False] * len(header)
        expected_mask += [message["role"] == "assistant"] * (len(content) + 1)
    assert list(ids) == expected_ids
    assert list(mask) == expected_mask
    assert ids.count(1) == 1 and ids.count(2) == 5
    assert sum(t == 2 and m for t, m in zip(ids, mask, strict=True)) == 2
    assert all(t not in (0, 3) for t in ids)
    for turn in (2, 4):
        prompt, _ = format_chat(codec, messages[:turn], max_length=512, generation_prompt=True)
        completed, _ = format_chat(codec, messages[: turn + 1], max_length=512)
        assert completed[: len(prompt)] == prompt
        assert completed[len(prompt) :] == tuple(codec.encode(messages[turn]["content"]) + [2])


@pytest.mark.parametrize(
    "messages,generation",
    [
        ([], False),
        ([{"role": "user", "content": "Hello"}], False),
        ([{"role": "assistant", "content": "Hi"}], False),
        ([{"role": "tool", "content": "Hi"}], False),
        ([{"role": "user", "content": " "}], True),
        ([{"role": "user", "content": "Hi", "extra": 1}], True),
        ([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": ""}], False),
        ([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hi"}], True),
    ],
)
def test_invalid_or_targetless_chat_rejected(codec, messages, generation):
    with pytest.raises(ValueError):
        format_chat(codec[0], messages, max_length=512, generation_prompt=generation)


def test_overflow_rejected_without_partial_answer_or_synthetic_eos(codec):
    messages = [{"role": "user", "content": "Go"}, {"role": "assistant", "content": "Hello"}]
    ids, _ = format_chat(codec[0], messages, max_length=512)
    assert format_chat(codec[0], messages, max_length=len(ids))[0] == ids
    with pytest.raises(ValueError, match="truncation"):
        format_chat(codec[0], messages, max_length=len(ids) - 1)
    with pytest.raises(ValueError, match="truncation"):
        format_chat(codec[0], messages[:1], max_length=3, generation_prompt=True)


def masked_dataset():
    return TokenDataset(
        ((1, 40, 41, 42, 2, 43, 44, 2), (1, 50, 51, 52, 2)),
        16,
        260,
        "a" * 64,
        "train",
        "b" * 64,
        ((False, False, True, True, True, False, True, True), (False, False, False, True, True)),
    )


def test_loss_alignment_padding_and_zero_gradient_on_ignored_predictions():
    dataset = masked_dataset()
    ids, labels, count = collate(dataset, [0, 1], "cpu")
    assert count == 7
    assert labels.tolist() == [
        [-100, -100, 41, 42, 2, -100, 44, 2],
        [-100, -100, -100, 52, 2, -100, -100, -100],
    ]
    logits = torch.randn(2, 8, 260, requires_grad=True)
    # Explicit (batch, preceding logit position, target ID), independent of collation.
    positions = [(0, 1, 41), (0, 2, 42), (0, 3, 2), (0, 5, 44), (0, 6, 2), (1, 2, 52), (1, 3, 2)]
    expected = sum(-F.log_softmax(logits[b, t], dim=-1)[token] for b, t, token in positions) / 7
    actual = causal_loss(logits, labels)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    active = {(b, t) for b, t, _ in positions}
    for b in range(2):
        for t in range(8):
            assert bool(logits.grad[b, t].abs().sum() > 0) == ((b, t) in active)
    config = ModelConfig(1, 260, 16, 16, 2, 1, 32, 10000.0, 1e-5, "a" * 64)
    model = create_model(config)
    original = model(ids, labels).loss
    ids[1, 5:] = 99
    torch.testing.assert_close(model(ids, labels).loss, original)


@pytest.mark.parametrize(
    "mask",
    [
        None,
        ((False,),),
        ((False,) * 8, (False,) * 5),
        ((True,) * 8, (False, False, False, True, True)),
    ],
)
def test_invalid_masks_and_identity(mask):
    data = masked_dataset()
    if mask is None:
        plain = replace(data, target_masks=None)
        assert plain.identity != data.identity
        assert "objective" not in plain.identity
    else:
        with pytest.raises(ValueError):
            replace(data, target_masks=mask)


def test_split_integrity_and_reserved_payload_never_read(codec, tmp_path):
    DATA.materialize(tmp_path / "conversations")
    root = tmp_path / "conversations"
    (root / "test.jsonl").unlink()
    for split, count in (("train", 192), ("validation", 32)):
        assert len(load_conversations(root, split)) == count
        prepared = prepare_conversations(root, *codec, 512, split)
        assert prepared.identity["targets"] < sum(len(w) - 1 for w in prepared.windows)
    with pytest.raises(ValueError, match="reserved"):
        load_conversations(root, "test")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["splits"]["validation"]["source_groups"].append("apple")
    (root / "manifest.json").write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError, match="overlap"):
        load_conversations(root, "train")


def test_payload_tampering_rejected(tmp_path):
    DATA.materialize(tmp_path / "conversations")
    root = tmp_path / "conversations"
    with (root / "train.jsonl").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="checksum"):
        load_conversations(root, "train")


def make_trainer(batch=1, accumulation=2, device="cpu"):
    data = masked_dataset()
    config = ModelConfig(1, 260, 16, 16, 2, 1, 32, 10000.0, 1e-5, "a" * 64)
    return Trainer(
        create_model(config, 19),
        TrainingConfig(
            sequence_length=16,
            batch_size=batch,
            accumulation_steps=accumulation,
            max_steps=4,
            warmup_steps=0,
            learning_rate=0.0003,
        ),
        data,
        replace(data, split="validation"),
        device,
    )


def test_assistant_weighted_accumulation_and_readonly_validation():
    small, large = make_trainer(), make_trainer(2, 1)
    a, b = small.update(), large.update()
    assert a["targets"] == b["targets"] == 7
    assert a["loss"] == pytest.approx(b["loss"], abs=1e-6)
    for x, y in zip(small.model.parameters(), large.model.parameters(), strict=True):
        torch.testing.assert_close(x, y, atol=2e-7, rtol=2e-5)
    weights = copy.deepcopy(small.model.state_dict())
    rng = torch.get_rng_state().clone()
    cursor = small.stream.cursor
    measured = evaluate(small.model, small.validation_data, 2)
    assert measured["targets"] == 7 and small.stream.cursor == cursor
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(weights[k], v) for k, v in small.model.state_dict().items())


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_masked_recovery_and_mask_binding(tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS hardware unavailable")
    trainer = make_trainer(device=device)
    trainer.update()
    save_checkpoint(trainer, tmp_path / "checkpoint")
    restored = load_checkpoint(
        tmp_path / "checkpoint", trainer.train_data, trainer.validation_data, device
    )
    for _ in range(3):
        a, b = trainer.update(), restored.update()
        assert a["tokens_seen"] == b["tokens_seen"]
        assert a["loss"] == pytest.approx(b["loss"], abs=1e-6, rel=1e-5)
    for x, y in zip(trainer.model.parameters(), restored.model.parameters(), strict=True):
        torch.testing.assert_close(
            x, y, atol=0 if device == "cpu" else 1e-6, rtol=0 if device == "cpu" else 1e-5
        )
    assert trainer.tokens_seen == 28
    changed = replace(
        trainer.train_data,
        target_masks=(
            (False, True, False, True, True, False, True, True),
            trainer.train_data.target_masks[1],
        ),
    )
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(tmp_path / "checkpoint", changed, trainer.validation_data, device)
