"""Explicit final-selection declarations; routine runs cannot consume reserved inputs."""

from pathlib import Path

from latos.evaluation.inputs import read_json
from latos.model.storage import file_hash


def evaluation_access(protocol, suite_path, weights_sha256, selection_path=None):
    if selection_path is None:
        if file_hash(suite_path) != protocol["suite_sha256"]:
            raise ValueError("Routine evaluation requires the development suite")
        return "validation", None
    selection = read_json(selection_path)
    required = {
        "schema_version",
        "phase",
        "candidate_weights",
        "reference_weights",
        "development_comparison",
        "development_comparison_sha256",
        "contamination_audit",
        "contamination_audit_sha256",
        "declaration",
    }
    if (
        set(selection) != required
        or selection["schema_version"] != 1
        or selection["phase"] not in (17, 18)
    ):
        raise ValueError("Invalid final-selection declaration")
    if selection["declaration"] != "locked candidate; no training or selection from final feedback":
        raise ValueError("Missing final-access declaration")
    if weights_sha256 not in [selection["candidate_weights"], *selection["reference_weights"]]:
        raise ValueError("Model is not in the locked final comparison")
    parent = Path(selection_path).parent
    for key in ("development_comparison", "contamination_audit"):
        path = parent / selection[key]
        if file_hash(path) != selection[key + "_sha256"]:
            raise ValueError("Final-access evidence checksum mismatch")
    comparison = read_json(parent / selection["development_comparison"])
    audit = read_json(parent / selection["contamination_audit"])
    if (
        comparison.get("passed") is not True
        or comparison.get("candidate_weights") != selection["candidate_weights"]
        or comparison.get("stage") != ("base" if selection["phase"] == 17 else "assistant")
        or audit.get("accepted_for_final") is not True
        or audit.get("candidate_weights") != selection["candidate_weights"]
    ):
        raise ValueError("Final access requires passing candidate gates and contamination review")
    expected = protocol["suite_sha256" if selection["phase"] == 17 else "final_suite_sha256"]
    if file_hash(suite_path) != expected:
        raise ValueError("Wrong suite for the final acceptance phase")
    return "test", {"selection_sha256": file_hash(selection_path), "selection": selection}
