"""Single-pass conservation, independent gradients, cache integrity and recovery."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from latos.data.acquire import acquire
from latos.data.manifest import canonical_json
from latos.data.prepare import prepare
from latos.data2.prepare import split_for
from latos.model import ModelConfig, create_model
from latos.model.storage import file_hash
from latos.tokenization import LatoTokenizer
from latos.tokenization.training import train as fit_codec
from latos.training.base2 import frozen, training_config
from latos.training.checkpoint import load_checkpoint, save_checkpoint
from latos.training.config import TrainingConfig
from latos.training.data import TokenDataset, collate
from latos.training.engine import Trainer
from latos.training.full_data import FullDataset, build_cache
from latos.training.probe import assert_state_equal

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fixture_data(size=19):
    rows = tuple((1,) + tuple(range(4, 6 + i % 7)) + (2,) for i in range(size))
    training = TokenDataset(rows, 16, 260, "a" * 64, "train", "b" * 64)
    development = replace(
        training, windows=((1, 19, 20, 2),), split="validation", source_sha256="c" * 64
    )
    model = ModelConfig(1, 260, 16, 16, 2, 1, 32, 10000.0, 1e-5, "a" * 64)
    config = TrainingConfig(
        sequence_length=16,
        batch_size=2,
        accumulation_steps=4,
        max_steps=(size + 7) // 8,
        warmup_steps=0,
        seed=160,
        learning_rate=0.0003,
    )
    return model, config, training, development


def test_frozen_contract():
    plan = frozen(ROOT)
    config = training_config(plan)
    assert config.max_steps == 7485
    assert config.accumulation_steps == 8
    assert plan["execution_authorized"] is False  # Historical plan stays byte-identical.


@pytest.mark.parametrize("size", [1, 3, 8, 16, 19, 20])
def test_one_pass_conservation_and_recovery(tmp_path, size):
    model, config, training, dev = fixture_data(size)
    trainer = Trainer(create_model(model, 160), config, training, dev, single_pass=True)
    for step in range(config.max_steps):
        saved = tmp_path / str(step)
        save_checkpoint(trainer, saved)
        restored = load_checkpoint(saved, training, dev, expected_single_pass=True)
        a, b = trainer.update(), restored.update()
        assert a["targets"] == b["targets"]
        assert_state_equal(trainer.model.state_dict(), restored.model.state_dict())
        assert_state_equal(trainer.optimizer.state_dict(), restored.optimizer.state_dict())
    assert trainer.tokens_seen == sum(len(w) - 1 for w in training.windows)
    assert trainer.windows_seen == size
    assert trainer.stream.epoch == 0 and trainer.stream.cursor == size
    save_checkpoint(trainer, tmp_path / "final")
    restored = load_checkpoint(tmp_path / "final", training, dev, expected_single_pass=True)
    with pytest.raises(ValueError, match="complete"):
        restored.update()
    with pytest.raises(ValueError, match="sampling policy"):
        load_checkpoint(tmp_path / "final", training, dev, expected_single_pass=False)


def test_partial_update_matches_independent_token_sum(tmp_path):
    model, config, training, dev = fixture_data(20)
    trainer = Trainer(create_model(model, 160), config, training, dev, single_pass=True)
    trainer.update()
    trainer.update()
    reference = copy.deepcopy(trainer.model)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in reference.parameters() if p.ndim >= 2],
                "weight_decay": config.weight_decay,
            },
            {"params": [p for p in reference.parameters() if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=config.learning_rate_at(3),
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        foreach=False,
        fused=False,
    )
    optimizer.load_state_dict(copy.deepcopy(trainer.optimizer.state_dict()))
    for group in optimizer.param_groups:
        group["lr"] = config.learning_rate_at(3)
    indices = trainer.stream.order[16:].tolist()
    ids, labels, count = collate(training, indices, "cpu")
    logits = reference(ids).logits[:, :-1].float()
    # Independent sum reduction and denominator; no Trainer loss weighting.
    loss = (
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, 260), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum"
        )
        / count
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        reference.parameters(), config.max_grad_norm, error_if_nonfinite=True, foreach=False
    )
    optimizer.step()
    metric = trainer.update()
    assert metric["microbatches"] == 2 and metric["targets"] == count
    assert metric["loss"] == pytest.approx(loss.item(), abs=1e-6)
    for actual, expected in zip(trainer.model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)


def test_single_pass_rejects_budget_and_corrupt_recovery(tmp_path):
    model, config, training, dev = fixture_data()
    with pytest.raises(ValueError, match="budget"):
        Trainer(
            create_model(model, 160), replace(config, max_steps=4), training, dev, single_pass=True
        )
    trainer = Trainer(create_model(model, 160), config, training, dev, single_pass=True)
    trainer.update()
    save_checkpoint(trainer, tmp_path / "checkpoint")
    path = tmp_path / "checkpoint/state.json"
    data = json.loads(path.read_text())
    data["tokens_seen"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="exposure"):
        load_checkpoint(path.parent, training, dev, expected_single_pass=True)


def test_nonfinite_optimizer_cannot_checkpoint(tmp_path):
    model, config, training, dev = fixture_data()
    trainer = Trainer(create_model(model, 160), config, training, dev, single_pass=True)
    trainer.update()
    original = trainer.optimizer.step

    def poison():
        original()
        next(iter(trainer.optimizer.state.values()))["exp_avg"].fill_(float("inf"))

    trainer.optimizer.step = poison
    with pytest.raises(ValueError, match="state is nonfinite"):
        trainer.update()
    with pytest.raises(ValueError, match="completed"):
        save_checkpoint(trainer, tmp_path / "bad")


@pytest.fixture
def cache_inputs(tmp_path):
    manifest = ROOT / "data/fixtures/tiny/manifest.json"
    acquire(manifest, tmp_path / "raw")
    prepare(manifest, tmp_path / "raw", tmp_path / "old")
    fit_codec(
        manifest, tmp_path / "old", ROOT / "configs/tokenizer/debug.json", tmp_path / "tokenizer"
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    group = next(str(i) for i in range(10000) if split_for(str(i)) == "train")
    texts = ["A retained document has words. " * 80, "One half", "word after a seam."]
    positions = [("a", 0, 0), ("b", 0, 0), ("b", 0, 1)]
    rows = [
        {
            "id": f"{d}:p{p}:c{c}",
            "document_id": d,
            "group_id": group,
            "split": "train",
            "text": text,
            "kind": "pretrain",
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        for (d, p, c), text in zip(positions, texts, strict=True)
    ]
    file = corpus / "pretrain-train.jsonl"
    file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (corpus / "report.json").write_text(
        json.dumps({"files": {file.name: {"sha256": file_hash(file)}}})
    )
    codec = LatoTokenizer.load(tmp_path / "tokenizer")
    windows = []
    for text in [texts[0], texts[1] + texts[2]]:
        ids = codec.encode(text, add_bos=True, add_eos=True)
        windows.extend(tuple(ids[i : i + 512]) for i in range(0, len(ids) - 1, 511))
    digest = hashlib.sha256()
    for window in windows:
        digest.update(canonical_json(window))
    expected = {
        "windows": len(windows),
        "targets_with_EOS": sum(len(w) - 1 for w in windows),
        "window_sequence_sha256": digest.hexdigest(),
        "source_file_sha256": file_hash(file),
    }
    return corpus, tmp_path / "tokenizer", expected, windows


def test_compact_cache_transitions_and_corruption(cache_inputs, tmp_path):
    corpus, tokenizer, expected, windows = cache_inputs
    output = tmp_path / "cache"
    data = build_cache(corpus, tokenizer, "train", output, expected)
    assert list(data.windows) == windows
    assert (
        sum(data.target_count(i) for i in range(len(data.windows))) == expected["targets_with_EOS"]
    )
    a = collate(data, [0, len(windows) - 1], "cpu")
    reference = TokenDataset(
        tuple(windows), 512, data.vocab_size, data.tokenizer_sha256, "train", data.source_sha256
    )
    b = collate(reference, [0, len(windows) - 1], "cpu")
    assert_state_equal(a, b)
    # Windows requires closing memory mappings before writing the same file.
    del data
    token_path = output / "tokens.u32"
    with token_path.open("r+b") as stream:
        stream.seek(4)
        stream.write(np.asarray([42], dtype="<u4").tobytes())
    meta = json.loads((output / "cache.json").read_text())
    meta["files"]["tokens.u32"] = file_hash(token_path)
    (output / "cache.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="sequence"):
        FullDataset(output, expected, file_hash(tokenizer / "tokenizer.json"), "train")
    with pytest.raises(ValueError, match="Reserved"):
        build_cache(corpus, tokenizer, "reserved", tmp_path / "forbidden", expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA unavailable")
def test_full_single_pass_cuda(tmp_path):
    model, config, training, dev = fixture_data(20)
    trainer = Trainer(
        create_model(model, 160),
        config,
        training,
        dev,
        "cuda",
        precision="bfloat16",
        single_pass=True,
    )
    trainer.update()
    trainer.update()
    save_checkpoint(trainer, tmp_path / "before-tail")
    recovered = load_checkpoint(
        tmp_path / "before-tail",
        training,
        dev,
        "cuda",
        expected_precision="bfloat16",
        expected_single_pass=True,
    )
    a, b = trainer.update(), recovered.update()
    assert a["microbatches"] == b["microbatches"] == 2
    assert a["targets"] == b["targets"]
    assert_state_equal(trainer.model.state_dict(), recovered.model.state_dict())
    assert_state_equal(trainer.optimizer.state_dict(), recovered.optimizer.state_dict())
    assert trainer.windows_seen == 20
    assert trainer.tokens_seen == sum(len(w) - 1 for w in training.windows)
    save_checkpoint(trainer, tmp_path / "after-tail")
    load_checkpoint(tmp_path / "after-tail", training, dev, "cuda", expected_single_pass=True)


def test_supervisor_blocks_retries_and_final_without_gates(tmp_path):
    from types import SimpleNamespace

    from latos.training.base2 import supervise, write

    output = tmp_path / "attempt"
    output.mkdir()
    write(
        output / "ledger.json",
        {
            "training_seconds": 1.0,
            "evaluation_seconds": 0.0,
            "segments": 1,
            "jobs": [{"stage": "train", "status": "failed"}],
        },
    )
    args = SimpleNamespace(
        output=output, transfer=tmp_path, stage="train", resume_external_interruption="power loss"
    )
    with pytest.raises(ValueError, match="cannot be retried"):
        supervise(args)
    args.stage = "development"
    args.resume_external_interruption = None
    write(output / "training-result.json", {"status": "complete"})
    with pytest.raises(ValueError, match="Training incomplete"):
        supervise(args)
    write(
        output / "ledger.json",
        {
            "training_seconds": 7200.0,
            "evaluation_seconds": 3600.0,
            "segments": 1,
            "jobs": [{"stage": "train", "status": "complete"}],
        },
    )
    with pytest.raises(ValueError, match="budget exhausted"):
        supervise(args)
    args.stage = "final"
    write(output / "development-comparison.json", {"passed": False})
    with pytest.raises(ValueError, match="Final access blocked"):
        supervise(args)


def test_supervisor_hard_timeout_retains_ledger(tmp_path, monkeypatch):
    import subprocess
    import sys
    from types import SimpleNamespace

    from latos.training import base2

    output = tmp_path / "attempt"
    output.mkdir()
    base2.write(tmp_path / "handoff.json", {"files": {}})
    base2.write(
        output / "ledger.json",
        {
            "training_seconds": 7199.99,
            "evaluation_seconds": 0.0,
            "segments": 1,
            "jobs": [{"stage": "train", "status": "external-interruption"}],
        },
    )
    original = subprocess.Popen

    def sleeping_child(command, **kwargs):
        return original([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)

    monkeypatch.setattr(base2.subprocess, "Popen", sleeping_child)
    args = SimpleNamespace(
        output=output,
        transfer=tmp_path,
        stage="train",
        resume_external_interruption="documented power loss",
    )
    with pytest.raises(ValueError, match="budget-exceeded"):
        base2.supervise(args)
    ledger = base2.read(output / "ledger.json")
    assert ledger["segments"] == 2 and ledger["jobs"][-1]["status"] == "budget-exceeded"
    assert ledger["training_seconds"] >= 7200
    assert ledger["jobs"][-1]["returncode"] is not None


def test_attempt_lease_blocks_concurrent_supervisor(tmp_path):
    from latos.training.base2 import attempt_lock

    with attempt_lock(tmp_path), pytest.raises(OSError), attempt_lock(tmp_path):
        pytest.fail("Second supervisor acquired active attempt")


def test_phase17_entrypoint_help_from_other_directory(tmp_path):
    import subprocess
    import sys

    for name in ("preflight.py", "package_handoff.py", "verify_handoff.py", "package_return.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "experiments/phase-17" / name), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_preflight_rejects_skipped_cuda_and_failed_cases(tmp_path):
    import runpy

    helpers = runpy.run_path(str(ROOT / "experiments/phase-17/preflight.py"))
    names = helpers["REQUIRED"]
    path = tmp_path / "tests.xml"
    path.write_text(
        "<testsuite>" + "".join(f'<testcase name="{n}"/>' for n in names) + "</testsuite>"
    )
    assert helpers["check_junit"](path)["passed"] == len(names)
    path.write_text(
        path.read_text().replace(
            'name="test_full_single_pass_cuda"/>',
            'name="test_full_single_pass_cuda"><skipped/></testcase>',
        )
    )
    with pytest.raises(ValueError, match="did not execute"):
        helpers["check_junit"](path)


def test_return_archive_rejects_corruption_and_traversal(tmp_path):
    import runpy
    import zipfile

    unpack = runpy.run_path(str(ROOT / "experiments/phase-17/verify_return.py"))["unpack"]
    for index, name in enumerate(("safe.txt", "../escape")):
        archive = tmp_path / f"{index}.zip"
        data = b"original evidence"
        manifest = {"files": {name: {"bytes": len(data), "sha256": "0" * 64}}}
        with zipfile.ZipFile(archive, "x") as output:
            output.writestr("return.json", json.dumps(manifest))
            output.writestr(name, data)
        with pytest.raises(ValueError, match="hash mismatch|Unsafe"):
            unpack(archive, tmp_path / f"return-{index}")


def test_disk_start_prerequisite_is_not_reapplied_after_training(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from latos.training import probe

    monkeypatch.setattr(probe, "check_precision", lambda *args: None)
    monkeypatch.setattr(
        probe.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(name="RTX 4070 SUPER", total_memory=12 * 1024**3),
    )
    monkeypatch.setattr(probe.torch.cuda, "mem_get_info", lambda: (11 * 1024**3, 12 * 1024**3))
    monkeypatch.setattr(probe.torch.cuda, "get_device_capability", lambda: (8, 9))
    monkeypatch.setattr(probe.shutil, "disk_usage", lambda _: SimpleNamespace(free=19 * 1024**3))
    monkeypatch.setattr(probe, "runtime_identity", lambda _: {})
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *a, **k: "RTX 4070 SUPER, 616.92")
    with pytest.raises(ValueError, match="insufficient free disk"):
        probe.cuda_environment(tmp_path)
    assert probe.cuda_environment(tmp_path, min_free_bytes=0)["disk_free_bytes"] == 19 * 1024**3
