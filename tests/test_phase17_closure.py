"""Frozen arithmetic gate reporting: strict boundary and identity mismatches."""

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
statistics = runpy.run_path(str(ROOT / "experiments/phase-17/verify_closure.py"))[
    "replay_statistics"
]


def records():
    original = [
        {
            "step": i,
            "targets": 2,
            "tokens_seen": i * 2,
            "windows_seen": i,
            "microbatches": 2 if i == 7485 else 8,
            "epoch": 0,
            "learning_rate": 0.00003,
            "loss": 0.0,
        }
        for i in range(1, 7486)
    ]
    return original, [dict(row) for row in original[7000:]]


def test_frozen_boundary_and_physical_exposure():
    original, replay = records()
    replay[0]["loss"] = 1e-5
    result = statistics(original, replay)
    assert result["passed"] is True  # The frozen rule is >, not >=.
    assert result["logical_targets"] == 14970
    assert result["physical_targets"] == 15940
    replay[124]["loss"] = 0.00026229445423409103
    replay[254]["loss"] = 0.0007099797320835322
    result = statistics(original, replay)
    assert result["passed"] is False
    assert result["tolerance"] == 1e-5
    assert result["updates_over_tolerance"] == 2
    assert result["first_over_tolerance"]["step"] == 7125
    assert result["maximum_delta"]["step"] == 7255


@pytest.mark.parametrize("defect", ["counter", "schedule", "missing", "nonfinite"])
def test_malformed_or_changed_replay_is_not_accepted(defect):
    original, replay = records()
    if defect == "counter":
        replay[-1]["tokens_seen"] += 1
    elif defect == "schedule":
        replay[-1]["learning_rate"] *= 2
    elif defect == "missing":
        replay.pop()
    else:
        replay[-1]["loss"] = float("nan")
    with pytest.raises(ValueError):
        statistics(original, replay)
