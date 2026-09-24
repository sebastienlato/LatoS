"""Diagnostic orchestration/data/state checks; zero optimizer updates, no held-out access."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from latos.model import ModelConfig, create_model
from latos.training.checkpoint import load_checkpoint, save_checkpoint
from latos.training.config import TrainingConfig
from latos.training.data import TokenDataset, collate
from latos.training.engine import Trainer

DIRECTORY = Path(__file__).with_name("mechanics.py").parent
if not (DIRECTORY / "mechanics.py").exists():
    DIRECTORY = Path(__file__).resolve().parents[1] / "experiments/recovery-diagnostic"


def module(name):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


m, runner = module("mechanics"), module("run")


@pytest.fixture(autouse=True)
def no_updates_or_cuda():
    # The engineering checks must not consume the selected-scale diagnostic allowance.
    with patch.object(
        Trainer, "update", side_effect=AssertionError("No optimizer updates in preflight")
    ):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        yield
        torch.set_num_threads(old)


def model_config():
    return ModelConfig(1, 260, 512, 16, 2, 1, 32, 10000.0, 1e-5, "a" * 64)


def test_synthetic_shape_order_and_integrity(tmp_path):
    lengths = [2, 512, 8, 126, 333, 14, 45, 92]
    path = tmp_path / "lengths.json"
    path.write_text(json.dumps({"windows": len(lengths), "lengths": lengths}))
    plan = {"synthetic_seed": 17102, "shuffle_seed": 160}
    cfg = model_config()
    m.make_cache(path, tmp_path / "first", cfg, plan)
    m.make_cache(path, tmp_path / "second", cfg, plan)
    data = m.SyntheticDataset(tmp_path / "first", m.descriptor(path, cfg, plan))
    other = m.SyntheticDataset(tmp_path / "second", m.descriptor(path, cfg, plan))
    assert data.identity == other.identity
    order = torch.randperm(len(lengths), generator=torch.Generator().manual_seed(160)).tolist()
    assert [len(data.windows[i]) for i in order] == lengths
    ids, labels, targets = collate(data, order[:2], "cpu")
    assert ids.shape == (2, 512) and targets == 512
    assert labels[0, 2:].eq(-100).all()
    with pytest.raises(ValueError, match="descriptor"):
        m.SyntheticDataset(
            tmp_path / "first", {**m.descriptor(path, cfg, plan), "synthetic_seed": 0}
        )
    del data
    with (tmp_path / "first/tokens.u32").open("r+b") as stream:
        stream.write(np.asarray([42], dtype="<u4").tobytes())
    with pytest.raises(ValueError, match="checksum"):
        m.SyntheticDataset(tmp_path / "first", m.descriptor(path, cfg, plan))


def test_initial_checkpoint_state_is_exact_without_optimization(tmp_path):
    cfg = model_config()
    train = TokenDataset(((1, 4, 5, 2), (1, 6, 2)), 512, 260, "a" * 64, "train", "b" * 64)
    val = TokenDataset(((1, 7, 2),), 512, 260, "a" * 64, "validation", "c" * 64)
    config = TrainingConfig(sequence_length=512, max_steps=1, warmup_steps=0)
    trainer = Trainer(create_model(cfg, 17101), config, train, val, single_pass=True)
    before = m.snapshot(trainer)
    save_checkpoint(trainer, tmp_path / "initial")
    restored = load_checkpoint(tmp_path / "initial", train, val, expected_single_pass=True)
    assert m.snapshot(trainer) == before == m.snapshot(restored)
    assert trainer.step == 0 and trainer.optimizer.state == {}
    with torch.no_grad():
        next(restored.model.parameters()).flatten()[0] += 1
    assert m.snapshot(restored)["state_sha256"] != before["state_sha256"]


def trace(step, delta=0, changed=False):
    return {
        "metric": {
            "step": step,
            "targets": 5,
            "tokens_seen": 5 * step,
            "windows_seen": 2 * step,
            "epoch": 0,
            "microbatches": 1,
            "learning_rate": 0.001,
            "loss": 1.0 + delta,
        },
        "input": {"sha256": str(step)},
        "state": {
            "state_sha256": str(step) + ("bad" if changed else ""),
            "model": {"embedding": "bad" if changed else "ok"},
            "optimizer": {"0.exp_avg": "ok"},
        },
    }


def test_comparison_does_not_confuse_state_and_loss_gate():
    reference = [trace(i) for i in range(1, 5)]
    replay = copy.deepcopy(reference[2:])
    assert m.compare_pair(reference, replay, 2, 1e-5)["replay_states_identical"]
    replay[0] = trace(3, delta=0.0007, changed=True)
    result = m.compare_pair(reference, replay, 2, 1e-5)
    assert not result["loss_gate_passed"] and result["first_state_difference"]["step"] == 3
    assert not result["replay_states_identical"]
    replay[0] = trace(3, delta=0, changed=True)
    result = m.compare_pair(reference, replay, 2, 1e-5)
    assert result["loss_gate_passed"] and not result["replay_states_identical"]
    replay[0]["metric"]["targets"] = 6
    with pytest.raises(ValueError, match="counter"):
        m.compare_pair(reference, replay, 2, 1e-5)


def test_exact_predeclared_job_budget():
    plan = m.read(DIRECTORY / "plan.json")
    jobs = runner.job_specs(plan)
    assert len(jobs) == 5 and sum(j[2] for j in jobs) == 2964 < 4096
    assert plan["steps"] - plan["checkpoint_step"] == 485
    assert plan["max_seconds"] == 3600 and plan["max_artifact_bytes"] == 6 * 1024**3
    config = m.training_config(plan)
    assert config.max_steps == 997 and config.batch_size == 2 and config.accumulation_steps == 8


def test_child_workspace_is_set_before_torch_start_without_parent_mutation():
    plan = m.read(DIRECTORY / "plan.json")
    base = {"CUBLAS_WORKSPACE_CONFIG": "unrelated", "OTHER": "keep"}
    legacy = runner.profile_environment(base, "legacy", plan)
    corrected = runner.profile_environment(base, "deterministic", plan)
    assert "CUBLAS_WORKSPACE_CONFIG" not in legacy
    assert corrected["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert base["CUBLAS_WORKSPACE_CONFIG"] == "unrelated"
    assert corrected["OTHER"] == "keep" and corrected["PYTHONHASHSEED"] == "0"


def test_fingerprint_detects_single_value_or_layout_change():
    a = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    b = a.clone()
    assert m.tensor_digest(a) == m.tensor_digest(b)
    b[0, 0] = 1
    assert m.tensor_digest(a) != m.tensor_digest(b)
    assert m.tensor_digest(a) != m.tensor_digest(a.T)


def test_comparison_rejects_nonfinite_or_missing_replay():
    reference = [trace(i) for i in range(1, 5)]
    with pytest.raises(ValueError, match="counts"):
        m.compare_pair(reference, [trace(3)], 2, 1e-5)
    with pytest.raises(ValueError, match="Nonfinite"):
        m.compare_pair(reference, [trace(3, delta=float("nan")), trace(4)], 2, 1e-5)


def test_watchdog_stops_a_real_child_without_any_optimizer_work(tmp_path):
    import os
    import time

    with (tmp_path / "child.log").open("w") as log, pytest.raises(TimeoutError):
        runner.bounded_child(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            os.environ.copy(),
            log,
            lambda event: None,
            time.monotonic(),
            0.05,
            lambda: 0,
            100,
        )


def test_watchdog_does_not_accept_failed_worker(tmp_path):
    import os
    import time

    with (tmp_path / "child.log").open("w") as log, pytest.raises(RuntimeError):
        runner.bounded_child(
            [sys.executable, "-c", "raise SystemExit(9)"],
            os.environ.copy(),
            log,
            lambda event: None,
            time.monotonic(),
            30,
            lambda: 0,
            100,
        )


def test_monitor_tolerates_completed_checkpoint_rename(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    tensor = staging / "tensor.bin"
    tensor.write_bytes(b"state")
    old_path = tensor
    staging.rename(tmp_path / "final")
    assert runner.file_size(old_path) == 0
    assert runner.size(tmp_path) == 5
    with patch.object(Path, "stat", side_effect=PermissionError("denied")):
        with pytest.raises(PermissionError):
            runner.file_size(tmp_path / "final/tensor.bin")
