"""Counterexamples for the frozen selection rule and returned-evidence integrity."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "experiments/phase-16" / f"{name}.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def evidence():
    return json.loads((ROOT / "experiments/phase-16/windows-verification.json").read_text())


def layout():
    return json.loads((ROOT / "experiments/phase-16/full-corpus-layout.json").read_text())


def test_selection_is_smallest_eligible_in_band_not_lowest_loss_or_largest():
    data = evidence()
    for c in data["candidates"]:
        if c["precision"] == "bfloat16":
            c["final_development"]["loss"] = {8: 7.01, 14: 7.0, 20: 6.99}[c["layers"]]
    selected = module("select_configuration").select(data, layout())
    assert selected["selected_layers"] == 8
    assert selected["within_two_percent_layers"] == [8, 14, 20]
    # A real resource failure excludes the smallest even if its loss is acceptable.
    for c in data["candidates"]:
        if c["layers"] == 8:
            c["resources"]["peak_reserved_bytes"] = c["total_vram_bytes"]
    assert module("select_configuration").select(data, layout())["selected_layers"] == 14


def test_selection_band_and_all_failed_gates_cannot_be_waived():
    data = evidence()
    for c in data["candidates"]:
        if c["layers"] == 8:
            c["final_development"]["loss"] = 8.0
    assert module("select_configuration").select(data, layout())["selected_layers"] == 14
    for c in data["candidates"]:
        c["inference"]["median_decode_tokens_per_second"] = 1
    with pytest.raises(ValueError, match="No eligible"):
        module("select_configuration").select(data, layout())


def test_reviewed_plan_conserves_one_pass_and_does_not_enable_training():
    plan = json.loads((ROOT / "configs/training2/phase17-plan.json").read_text())
    corpus = layout()["splits"]["train"]
    assert (plan["max_steps"] - 1) * plan["batch_size"] * plan["accumulation_steps"] + plan[
        "batch_size"
    ] * plan["last_update_accumulation_steps"] == corpus["windows"]
    assert (
        plan["data"]["targets_with_EOS"] == corpus["content_token_positions"] + corpus["documents"]
    )
    assert not plan["execution_authorized"] and plan["data"]["passes"] == 1
    assert plan["attempts"]["serious_training"] == plan["attempts"]["seeds"] == 1


def test_hash_verifier_rejects_changed_bytes_and_missing_file(tmp_path):
    verify = module("verify_return")
    import hashlib

    path = tmp_path / "evidence.json"
    path.write_bytes(b"original")
    record = {"bytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()}
    verify.verify_record(path, record)
    path.write_bytes(b"changed!")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify.verify_record(path, record)
    path.unlink()
    with pytest.raises(ValueError, match="Missing evidence"):
        verify.verify_record(path, record)
