"""Read-only closure rejection tests; no model creation, optimizer or scoring run."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

from latos.data.manifest import canonical_json

PATH = Path(__file__).resolve().parents[1] / "experiments/phase-17-1/verify_closure.py"
spec = importlib.util.spec_from_file_location("phase171_closure", PATH)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def test_changed_component_rejects_state_fingerprint():
    state = {"model": {"weight": "original"}, "sampler": {"step": 1000}}
    state["state_sha256"] = hashlib.sha256(canonical_json(state)).hexdigest()
    v.state_check(state)
    state["sampler"]["step"] = 1001
    with pytest.raises(ValueError, match="digest"):
        v.state_check(state)


def test_final_partial_update_cannot_be_recounted_as_full():
    metric = {
        "step": 7485,
        "windows_seen": 119748,
        "targets": 1520,
        "tokens_seen": 37811418,
        "epoch": 0,
        "microbatches": 2,
        "learning_rate": 0.00003,
        "loss": 4.0,
        "grad_norm_before_clip": 1.0,
        "seconds": 0.1,
        "targets_per_second": 15200.0,
    }
    v.check_metric(metric, 7485, 119748, 1520, 37811418, 0.00003)
    metric["microbatches"] = 8
    with pytest.raises(ValueError, match="exposure"):
        v.check_metric(metric, 7485, 119748, 1520, 37811418, 0.00003)


def test_no_rate_tolerance_or_nonfinite_completed_metrics():
    metric = {
        "step": 1,
        "windows_seen": 16,
        "targets": 5000,
        "tokens_seen": 5000,
        "epoch": 0,
        "microbatches": 8,
        "learning_rate": 0.0003,
        "loss": 4.0,
        "grad_norm_before_clip": 1.0,
        "seconds": 0.1,
        "targets_per_second": 50000.0,
    }
    with pytest.raises(ValueError, match="rate"):
        v.check_metric(metric, 1, 16, 5000, 5000, 0.0003 + 1e-15)
    metric["loss"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        v.check_metric(metric, 1, 16, 5000, 5000, 0.0003)


def test_execution_success_does_not_waive_quality_or_final_review():
    assert (
        v.disposition(True, False)
        == "development-quality-failed; final prohibited; Phase 18 blocked"
    )
    assert v.disposition(True, True) == "requires final-selection review; not accepted"
    with pytest.raises(ValueError, match="not complete"):
        v.disposition(False, False)
