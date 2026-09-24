"""Independent boundary, integrity, precision and recovery checks for Phase 16."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from latos.data.acquire import acquire
from latos.data.manifest import canonical_json
from latos.data.prepare import prepare
from latos.data2.prepare import split_for
from latos.model import ModelConfig, create_model
from latos.model.storage import file_hash
from latos.tokenization import LatoTokenizer
from latos.tokenization.training import train
from latos.training.checkpoint import load_checkpoint
from latos.training.config import TrainingConfig
from latos.training.data import TokenDataset, collate
from latos.training.data2 import coalesce_document, prepare_probe_dataset
from latos.training.engine import Trainer
from latos.training.precision import check_precision
from latos.training.probe import (
    assert_state_equal,
    host_peak_bytes,
    inventory,
    numerical_control,
    verify_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def row(document, paragraph, chunk, text, group="group", split="train"):
    return {
        "id": f"{document}:p{paragraph}:c{chunk}",
        "document_id": document,
        "group_id": group,
        "split": split,
        "text": text,
        "kind": "pretrain",
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def test_numeric_order_chunk_seams_and_missing_material():
    rows = [
        row("a", 10, 0, "last"),
        row("a", 2, 1, "word"),
        row("a", 2, 0, "half"),
        row("a", 2, 3, "after gap"),
        row("a", 2, 4, "é"),
    ]
    assert coalesce_document(rows) == "halfword\n\nafter gapé\n\nlast"
    with pytest.raises(ValueError, match="Duplicate"):
        coalesce_document(rows + rows[:1])
    with pytest.raises(ValueError, match="Inconsistent"):
        coalesce_document(rows + [row("b", 0, 0, "other")])
    with pytest.raises(ValueError, match="Inconsistent"):
        coalesce_document([rows[0], {**rows[1], "split": "development"}])


@pytest.fixture
def inputs(tmp_path):
    manifest = ROOT / "data/fixtures/tiny/manifest.json"
    acquire(manifest, tmp_path / "raw")
    prepare(manifest, tmp_path / "raw", tmp_path / "old")
    train(manifest, tmp_path / "old", ROOT / "configs/tokenizer/debug.json", tmp_path / "tokenizer")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    groups = {}
    for split in ("train", "development"):
        groups[split] = next(str(i) for i in range(10000) if split_for(str(i)) == split)
    report, all_rows = {"files": {}}, {}
    for split in groups:
        g = groups[split]
        rows = [
            row("a", 0, 0, "An original half", g, split),
            row("a", 0, 1, "word ends here.", g, split),
            row("a", 2, 0, "A separate retained paragraph.", g, split),
            row("b", 0, 0, "Another document has its own context.", g, split),
        ]
        name = f"pretrain-{split}.jsonl"
        (corpus / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        report["files"][name] = {"sha256": file_hash(corpus / name)}
        all_rows[split] = rows
    (corpus / "report.json").write_text(json.dumps(report))
    return corpus, tmp_path / "tokenizer", all_rows


def test_all_transitions_once_no_cross_document_or_reserved_read(inputs):
    corpus, tokenizer, rows = inputs
    dataset, info = prepare_probe_dataset(corpus, tokenizer, "train", 16, max_documents=2)
    codec = LatoTokenizer.load(tokenizer)
    expected = []
    for text in (
        "An original halfword ends here.\n\nA separate retained paragraph.",
        "Another document has its own context.",
    ):
        ids = codec.encode(text, add_bos=True, add_eos=True)
        expected.extend(zip(ids[:-1], ids[1:], strict=True))
    observed = [pair for w in dataset.windows for pair in zip(w[:-1], w[1:], strict=True)]
    assert observed == expected
    assert (2, 1) not in observed
    assert info["targets"] == len(expected)
    assert info["included"] and dataset.split == "train"
    repeated, info2 = prepare_probe_dataset(corpus, tokenizer, "train", 16, max_documents=2)
    assert repeated == dataset and info == info2
    dev, _ = prepare_probe_dataset(corpus, tokenizer, "development", 16, max_documents=2)
    assert dev.split == "validation" and dataset.source_sha256 != dev.source_sha256
    for split in ("reserved", "test", "validation"):
        with pytest.raises(ValueError, match="reservations"):
            prepare_probe_dataset(corpus, tokenizer, split, 16, max_documents=2)
    ids, labels, targets = collate(dataset, [0, len(dataset.windows) - 1], "cpu")
    assert targets == int((labels[:, 1:] != -100).sum())
    assert ids.shape == labels.shape


def test_record_corruption_and_noncontiguous_documents_fail(inputs):
    corpus, tokenizer, all_rows = inputs
    path = corpus / "pretrain-train.jsonl"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Corpus identity"):
        prepare_probe_dataset(corpus, tokenizer, "train", 16, max_documents=2)
    rows = all_rows["train"]
    path.write_text("".join(json.dumps(r) + "\n" for r in [rows[0], rows[-1], *rows[1:-1]]))
    report_path = corpus / "report.json"
    report = json.loads(report_path.read_text())
    report["files"][path.name]["sha256"] = file_hash(path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not contiguous"):
        prepare_probe_dataset(corpus, tokenizer, "train", 16, max_documents=2)


def test_coalescing_preserves_document_causality(inputs):
    corpus, tokenizer, _ = inputs
    dataset, _ = prepare_probe_dataset(corpus, tokenizer, "train", 64, max_documents=2)
    cfg = ModelConfig(
        1, dataset.vocab_size, 64, 16, 2, 1, 32, 10000.0, 1e-5, dataset.tokenizer_sha256
    )
    model = create_model(cfg)
    ids, labels, _ = collate(dataset, [0, len(dataset.windows) - 1], "cpu")
    before = model(ids).logits.detach()
    ids[1, :] = 40
    after = model(ids).logits.detach()
    torch.testing.assert_close(before[0], after[0], atol=0, rtol=0)
    ids[0, -1] = 41
    future = model(ids).logits.detach()
    torch.testing.assert_close(after[0, :-1], future[0, :-1], atol=0, rtol=0)


@pytest.mark.parametrize(
    "device,precision", [("cpu", "bfloat16"), ("mps", "bfloat16"), ("cpu", "float16")]
)
def test_no_silent_precision_fallback(device, precision):
    with pytest.raises(ValueError):
        check_precision(device, precision)


def test_cpu_numerical_control_and_precision_resume_rejection(tmp_path):
    result = numerical_control(tmp_path, "cpu", "float32")
    assert result["gradient_relative_l2"] == 0
    assert result["four_update_weight_relative_l2"] == 0
    path = tmp_path / "recovery/state.json"
    state = json.loads(path.read_text())
    assert state["schema_version"] == 1 and "precision" not in state
    rows = TokenDataset(((1, 4, 2),), 32, 260, "a" * 64, "train", "b" * 64)
    with pytest.raises(ValueError, match="precision mismatch"):
        load_checkpoint(tmp_path / "recovery", rows, rows, expected_precision="bfloat16")
    state.update(schema_version=2, kind="latos-training-autocast-v2", precision="float16")
    path.write_bytes(canonical_json(state))
    with pytest.raises(ValueError, match="Unsupported checkpoint precision"):
        load_checkpoint(tmp_path / "recovery", rows, rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA unavailable; not simulated")
def test_cuda_bf16_numerical_control(tmp_path):
    assert numerical_control(tmp_path)["status"] == "passed"


def test_inventory_and_state_comparison_detect_changes(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/a.txt").write_text("original evidence")
    inventory(tmp_path)
    contents = json.loads((tmp_path / "inventory.json").read_text())
    assert set(contents) == {"nested/a.txt"}
    inventory(tmp_path)
    assert json.loads((tmp_path / "inventory.json").read_text()) == contents
    with pytest.raises(AssertionError):
        assert_state_equal(torch.ones(3), torch.zeros(3))
    with pytest.raises(ValueError):
        assert_state_equal({"a": 1}, {"a": 2})
    peak = host_peak_bytes()
    assert peak is None or peak > 0


def test_package_pin_and_help_entrypoint(tmp_path):
    (tmp_path / "bundle.json").write_text("{}")
    with pytest.raises(ValueError, match="exact accepted"):
        verify_inputs(tmp_path)
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments/phase-16/run.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "--inputs" in result.stdout


def test_device_completion_failure_blocks_checkpoint(monkeypatch, tmp_path):
    from dataclasses import replace

    from latos.training.checkpoint import save_checkpoint

    cfg = ModelConfig(1, 260, 16, 16, 2, 1, 32, 10000.0, 1e-5, "a" * 64)
    train = TokenDataset(((1, 8, 9, 2),), 16, 260, "a" * 64, "train", "b" * 64)
    dev = replace(train, split="validation")
    trainer = Trainer(create_model(cfg), TrainingConfig(sequence_length=16), train, dev)
    calls = []

    def fail_at_completion(device):
        calls.append(device)
        if len(calls) == 2:
            raise RuntimeError("simulated device completion failure")

    monkeypatch.setattr("latos.training.engine.synchronize", fail_at_completion)
    with pytest.raises(RuntimeError, match="device completion"):
        trainer.update()
    assert not trainer.ready
    with pytest.raises(ValueError, match="completed update"):
        save_checkpoint(trainer, tmp_path / "invalid")


def test_suite_rejects_unavailable_cuda_and_retains_failure(tmp_path, monkeypatch):
    from argparse import Namespace

    from latos.training.probe import suite

    monkeypatch.setattr("latos.training.probe.verify_inputs", lambda p: {"test_stub": True})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "blocked"
    with pytest.raises(ValueError, match="actual CUDA"):
        suite(Namespace(inputs=tmp_path, output=output))
    assert (output / "failure.json").exists() and (output / "inventory.json").exists()
    assert not (output / "summary.json").exists()
    assert not (output / "control").exists()
    with pytest.raises(FileExistsError):
        suite(Namespace(inputs=tmp_path, output=output))
