"""Saved-record regression and cumulative-budget checks; never execute a model."""

import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "experiments/phase-17-1/small-subset"
sys.path.insert(0, str(DIR))
spec = importlib.util.spec_from_file_location("packfix", DIR / "packaging-correction/correct.py")
fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)


@pytest.fixture(autouse=True)
def prohibit_model_execution(monkeypatch):
    import torch

    import latos.evaluation.runner as runner
    import latos.model as model
    import latos.model.storage as storage
    from latos.model.network import LatoModel
    from latos.training.engine import Trainer

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved-evidence test must not execute a model")

    for target, name in (
        (Trainer, "update"),
        (torch.optim.AdamW, "step"),
        (LatoModel, "forward"),
        (model, "create_model"),
        (storage, "load_model"),
        (runner, "run"),
        (torch.cuda, "_lazy_init"),
    ):
        monkeypatch.setattr(target, name, forbidden)


def test_cumulative_budget_floor_and_rounding():
    assert fix.charge({"total_wall_seconds": 1272.1, "exit_code": 1}) == (1273, 300)
    assert fix.charge({"total_wall_seconds": 1273.1, "exit_code": 1}) == (1274, 300)
    assert fix.charge({"total_wall_seconds": 2450.1, "exit_code": 1}) == (2451, 249)


@pytest.mark.parametrize(
    "guardian",
    [
        {"total_wall_seconds": 2700, "exit_code": 1},
        {"total_wall_seconds": float("nan"), "exit_code": 1},
        {"total_wall_seconds": -1, "exit_code": 1},
        {"total_wall_seconds": "1273", "exit_code": 1},
        {"total_wall_seconds": 1273, "exit_code": 0},
        {"total_wall_seconds": 1273, "exit_code": 1, "deadline_terminated": True},
    ],
)
def test_invalid_prior_budget_or_disposition_is_terminal(guardian):
    with pytest.raises(ValueError):
        fix.charge(guardian)


def make_preconditions(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "bootstrap.log").write_text("ValueError: Child failed: 1")
    fix.write(tmp_path / "guardian.json", {"total_wall_seconds": 1273, "exit_code": 1})
    fix.write(
        run / "outcome.json",
        {
            "execution_complete": True,
            "failure": None,
            "accepted_base": False,
            "final_scored": False,
            "elapsed_before_package": 1200,
        },
    )
    (run / "events.jsonl").write_text(
        "".join(
            json.dumps({"event": p + "-complete", "elapsed": 1100}) + "\n"
            for p in ("prepare", "training", "evaluation")
        )
    )
    (run / "package-events.jsonl").write_text(json.dumps({"event": "child-exit", "code": 1}) + "\n")
    (run / "package.log").write_text(
        'File "ss_evidence.py", in summarize\n diagnostic = compare(...)\n' + fix.ERROR
    )
    fix.write(run / "verification.json", {"scripted": True})
    return run


def test_exact_failure_prerequisite_and_no_overwrite(tmp_path):
    run = make_preconditions(tmp_path)
    assert fix.diagnosis(tmp_path)["prior_debit_seconds"] == 1273
    (run / "package.log").write_text(
        fix.ERROR
    )  # Generic message does not prove cross-arm location.
    with pytest.raises(ValueError, match="Traceback"):
        fix.diagnosis(tmp_path)


def test_existing_completion_cannot_be_replaced(tmp_path):
    run = make_preconditions(tmp_path)
    fix.write(run / "completed.json", {})
    with pytest.raises(ValueError, match="Unexpected original"):
        fix.diagnosis(tmp_path)


def test_original_mutation_detected_and_deadline(tmp_path):
    (tmp_path / "evidence.txt").write_text("original")
    before = fix.inventory(tmp_path, time.monotonic() + 3)
    (tmp_path / "evidence.txt").write_text("changed")
    assert fix.inventory(tmp_path, time.monotonic() + 3) != before
    with pytest.raises(ValueError, match="deadline"):
        fix.inventory(tmp_path, time.monotonic() - 1)


@pytest.fixture(scope="module")
def saved_panel(tmp_path_factory):
    # Historical Phase 17.1 saved outputs are a regression FIXTURE, not small-subset results.
    source = ROOT / "outputs/phase17-1-return-reviewed/run"
    panel = tmp_path_factory.mktemp("historical_saved_row_fixture")
    shutil.copytree(source / "development-reference", panel / "evaluation-reference")
    for name in fix.ENDPOINTS:
        shutil.copytree(source / "development-candidate", panel / f"evaluation-{name}")
    return panel


def test_original_bug_reproduced_with_valid_historical_reference(saved_panel):
    import ss_evidence

    with pytest.raises(ValueError, match=fix.ERROR):
        ss_evidence.summarize(saved_panel)


def test_corrected_summary_keeps_real_gates_and_uses_descriptive_pairing(saved_panel):
    from latos.evaluation.compare import HISTORICAL_BASE, compare, paired_interval

    before = fix.inventory(saved_panel, time.monotonic() + 60)
    result = fix.summarize_saved(saved_panel)
    for name in fix.ENDPOINTS:
        assert result["fixed_gate_comparisons"][name] == compare(
            saved_panel / "evaluation-reference", saved_panel / f"evaluation-{name}", "base"
        )
        assert result["fixed_gate_comparisons"][name]["reference_weights"] == HISTORICAL_BASE
    assert len(result["descriptive_contrasts"]) == 4
    for a, b in (("A1", "B1"), ("B1", "B2"), ("A1", "B2"), ("B2", "C2")):
        paired = result["descriptive_contrasts"][a + ":" + b]["paired"]
        for name in ("ARC-Easy", "ARC-Challenge"):
            assert paired[name] == paired_interval(
                fix.read(saved_panel / f"evaluation-{a}/{name}.json"),
                fix.read(saved_panel / f"evaluation-{b}/{name}.json"),
            )
        assert "checks" not in result["descriptive_contrasts"][a + ":" + b]
    assert result["accepted_base"] is False and result["phase18_authorized"] is False
    assert fix.inventory(saved_panel, time.monotonic() + 60) == before


def test_genuinely_invalid_reference_is_still_rejected(saved_panel, monkeypatch):
    from latos.evaluation import compare as c

    original = c.verified_run

    def altered(path):
        manifest, summary = original(path)
        if Path(path).name == "evaluation-reference":
            manifest["model"]["weights_sha256"] = "f" * 64
        return manifest, summary

    monkeypatch.setattr(c, "verified_run", altered)
    with pytest.raises(ValueError, match=fix.ERROR):
        fix.summarize_saved(saved_panel)


def test_missing_reference_never_triggers_scoring(tmp_path):
    with pytest.raises(FileNotFoundError):
        fix.summarize_saved(tmp_path)


def test_packaging_stage_budget_is_not_reset():
    assert fix.budget_ledger(
        {"total_wall_seconds": 1273, "exit_code": 1}, {"elapsed_before_package": 1200}
    ) == (1273, 227, 73)
    with pytest.raises(ValueError, match="packaging allowance"):
        fix.budget_ledger(
            {"total_wall_seconds": 1273, "exit_code": 1}, {"elapsed_before_package": 973}
        )
    with pytest.raises(ValueError, match="packaging start"):
        fix.budget_ledger(
            {"total_wall_seconds": 1273, "exit_code": 1}, {"elapsed_before_package": 1274}
        )


def test_corrective_worker_preserves_original_failure_and_existing_verification(
    tmp_path, monkeypatch
):
    """Scripted execution receipts; real packaging/readback. No Windows result claim."""
    from types import SimpleNamespace

    import ss_evidence

    from latos.training.checkpoint import implementation_hash

    original, destination = tmp_path / "original", tmp_path / "correction"
    original.mkdir()
    destination.mkdir()
    run = make_preconditions(original)
    tools = run / "tools"
    tools.mkdir()
    fix.write(tools / "bundle.json", {"source_commit": fix.ORIGINAL_COMMIT, "files": {}})
    fix.write(
        tools / "legacy-plan.json",
        {"source": {"original_core_implementation_sha256": implementation_hash()}},
    )
    monkeypatch.setattr(fix, "BUNDLE", fix.sha(tools / "bundle.json"))
    monkeypatch.setattr(ss_evidence, "verify_training", lambda *args, **kwargs: {"scripted": True})
    monkeypatch.setattr(ss_evidence, "verify_execution", lambda *args, **kwargs: {})
    monkeypatch.setattr(fix, "summarize_saved", lambda root: {"scripted_saved_comparison": True})
    # The complete real saved-row comparison is covered separately above.
    digest = fix.sha(Path(fix.__file__))
    import hashlib

    ticket = "scripted-ticket"
    monkeypatch.setenv("LATOS_PACKAGING_ONLY", ticket)
    origin = time.monotonic()
    fix.write(
        destination / "authorization.json",
        {
            "ticket_sha256": hashlib.sha256(ticket.encode()).hexdigest(),
            "corrector_sha256": digest,
            "origin": origin,
            "prior_debit_seconds": 1273,
        },
    )
    before = fix.inventory(original, origin + 30)
    fix.worker(
        SimpleNamespace(sha256=digest, debit=1273, origin=origin, deadline=origin + 217),
        original,
        destination,
    )
    assert fix.inventory(original, time.monotonic() + 30) == before
    receipt = fix.read(destination / "return.receipt.json")
    assert receipt["verified"] and receipt["additional_optimizer_updates"] == 0
    assert not (run / "comparisons.json").exists()
    import zipfile

    with zipfile.ZipFile(destination / "small-subset-corrected-return.zip") as z:
        names = z.namelist()
        assert "original/run/package.log" in names
        assert "original/guardian.json" in names
        assert "correction/comparisons.json" in names
        assert json.loads(z.read("return.json"))["original_packaging_failure_preserved"]

    # Complete synthetic return round-trip through the separately reviewed Mac reader.
    sys.path.insert(0, str(DIR / "packaging-correction"))
    reader_spec = importlib.util.spec_from_file_location(
        "packfix_reader", DIR / "packaging-correction/verify_return.py"
    )
    reader = importlib.util.module_from_spec(reader_spec)
    reader_spec.loader.exec_module(reader)
    monkeypatch.setattr(reader, "BUNDLE", fix.BUNDLE)
    monkeypatch.setattr(reader, "verify_training", lambda *args, **kwargs: {"scripted": True})
    monkeypatch.setattr(reader, "verify_execution", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        reader, "summarize_saved", lambda *args: {"scripted_saved_comparison": True}
    )
    handoff = tmp_path / "known-handoff.zip"
    with zipfile.ZipFile(handoff, "w") as z:
        z.write(tools / "bundle.json", "bundle.json")
    correction_receipt = tmp_path / "mac-receipt.json"
    fix.write(correction_receipt, {"script": {"sha256": digest}})
    elapsed = receipt["packaged_elapsed_seconds"] + 1
    guardian = destination / "guardian.json"
    fix.write(
        guardian,
        {
            "exit_code": 0,
            "return_receipt_sha256": fix.sha(destination / "return.receipt.json"),
            "prior_debit_seconds": 1273,
            "correction_wall_seconds": elapsed,
            "cumulative_charged_seconds": 1273 + elapsed,
            "cumulative_packaging_seconds": 73 + elapsed,
            "original_guardian_sha256": fix.sha(original / "guardian.json"),
        },
    )
    report = reader.review(
        destination / "small-subset-corrected-return.zip",
        destination / "return.receipt.json",
        guardian,
        handoff,
        correction_receipt,
        tmp_path / "reviewed",
    )
    assert report["original_packaging_failure_preserved"] and report["new_optimizer_updates"] == 0
    assert report["accepted_base"] is False
    terminal = fix.read(guardian)
    terminal["cumulative_charged_seconds"] = 2701
    guardian.write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="Cumulative"):
        reader.review(
            destination / "small-subset-corrected-return.zip",
            destination / "return.receipt.json",
            guardian,
            handoff,
            correction_receipt,
            tmp_path / "rejected",
        )
