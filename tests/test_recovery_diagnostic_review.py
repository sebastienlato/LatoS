"""Read-only evidence rejection checks; no model, optimizer or CUDA execution."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/recovery-diagnostic/review_return.py"
spec = importlib.util.spec_from_file_location("diagnostic_review", PATH)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def pair():
    rows = [
        {
            "metric": {
                "step": i,
                "loss": 0.0,
                "learning_rate": 0.0003,
                "targets": 2,
                "tokens_seen": 2 * i,
                "windows_seen": i,
                "epoch": 0,
                "microbatches": 1,
            },
            "input": {"sha256": str(i)},
            "state": {"model": str(i)},
        }
        for i in range(1, 998)
    ]
    return rows, copy.deepcopy(rows[512:])


def test_loss_boundary_unchanged_and_exact_state_required():
    a, b = pair()
    b[0]["metric"]["loss"] = 1e-5
    assert review.compare(a, b)["loss_gate_passed"]
    b[0]["metric"]["loss"] = math.nextafter(1e-5, math.inf)
    assert not review.compare(a, b)["loss_gate_passed"]
    b[0]["metric"]["loss"] = 0
    b[0]["state"]["model"] = "changed"
    outcome = review.compare(a, b)
    assert outcome["loss_gate_passed"] and not outcome["exact_states"]


def test_same_windows_schedule_has_no_rounding_allowance():
    a, b = pair()
    b[0]["metric"]["learning_rate"] = math.nextafter(0.0003, math.inf)
    with pytest.raises(ValueError, match="Same-Windows"):
        review.compare(a, b)


def test_pair_rejects_incomplete_nonfinite_and_wrong_input():
    a, b = pair()
    with pytest.raises(ValueError, match="Incomplete"):
        review.compare(a, b[:-1])
    b[0]["metric"]["loss"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        review.compare(a, b)
    b[0]["metric"]["loss"] = 0
    b[0]["input"] = {"sha256": "wrong"}
    with pytest.raises(ValueError, match="input"):
        review.compare(a, b)


def test_control_must_fail_and_corrected_state_must_match():
    passing = {"loss_gate_passed": True, "exact_states": True}
    failing = {"loss_gate_passed": False, "exact_states": False}
    assert not review.decision({"legacy": passing, "deterministic": passing})
    assert review.decision({"legacy": failing, "deterministic": passing})
    assert not review.decision(
        {"legacy": failing, "deterministic": {"loss_gate_passed": True, "exact_states": False}}
    )


def test_digest_binds_nested_state():
    state = {"model": {"weight": "original"}, "sampler": {"step": 1}}
    state["state_sha256"] = review.digest(review.canonical(state))
    review.state_check(state)
    state["sampler"]["step"] = 2
    with pytest.raises(ValueError, match="digest"):
        review.state_check(state)


def test_cross_platform_diagnostic_is_not_an_arbitrary_rate_tolerance():
    config = {"learning_rate": 0.0003, "warmup_steps": 375, "max_steps": 997, "min_lr_ratio": 0.1}
    values = review.schedule_candidates(862, config)
    assert 6.018549288083756e-05 in values
    assert 6.018549288083756e-05 + 1e-12 not in values
    assert review.schedule_candidates(1, config) == {0.0003 / 375}
