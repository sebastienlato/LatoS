"""Review the corrective return using known Mac code; saved rows only, no model execution."""

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

from correct import (
    BUNDLE,
    budget_ledger,
    diagnosis,
    prevent_execution,
    read,
    record,
    require,
    sha,
    summarize_saved,
    write,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ss_common import safe  # noqa: E402
from ss_evidence import verify_execution, verify_training  # noqa: E402


def review(archive, receipt, guardian, handoff, correction_receipt, destination):
    require(
        sha(Path(__file__).with_name("correct.py")) == read(correction_receipt)["script"]["sha256"],
        "Use the reviewed corrective reader/source",
    )
    expected = read(receipt)
    require(
        record(archive) == {k: expected[k] for k in ("bytes", "sha256")}, "Corrected ZIP checksum"
    )
    terminal = read(guardian)
    require(
        terminal["exit_code"] == 0
        and not terminal.get("deadline_terminated", False)
        and terminal["return_receipt_sha256"] == sha(receipt),
        "Corrective guardian/receipt",
    )
    destination.mkdir(exist_ok=False)
    bundle = destination / "known-bundle"
    bundle.mkdir()
    with zipfile.ZipFile(handoff) as z:
        require(
            hashlib.sha256(z.read("bundle.json")).hexdigest() == BUNDLE, "Known original bundle"
        )
        manifest = json.loads(z.read("bundle.json"))
        require(set(z.namelist()) == set(manifest["files"]) | {"bundle.json"}, "Handoff members")
        for name, identity in manifest["files"].items():
            path = safe(bundle, name)
            raw = z.read(name)
            require(
                len(raw) == identity["bytes"]
                and hashlib.sha256(raw).hexdigest() == identity["sha256"],
                "Known source/cache hash",
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    root = destination / "return"
    root.mkdir()
    with zipfile.ZipFile(archive) as z:
        returned = json.loads(z.read("return.json"))
        require(
            len(z.namelist()) == len(set(z.namelist()))
            and set(z.namelist()) == set(returned["files"]) | {"return.json"},
            "Return members",
        )
        require(
            returned["original_bundle_sha256"] == BUNDLE
            and returned["correction_source_sha256"]
            == read(correction_receipt)["script"]["sha256"],
            "Corrective source binding",
        )
        for name, identity in returned["files"].items():
            path = safe(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, path.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            require(record(path) == identity, "Returned file integrity")
    require(
        sha(root / "correction/correct.py") == read(correction_receipt)["script"]["sha256"],
        "Returned corrector differs from reviewed source",
    )
    require(
        returned["additional_optimizer_updates"] == 0
        and returned["new_model_scoring"] is False
        and returned["accepted_base"] is False
        and returned["phase18_authorized"] is False,
        "Corrective scope changed",
    )
    original = root / "original"
    diagnosis(original)
    debit, cap, packaging_debit = budget_ledger(
        read(original / "guardian.json"), read(original / "run/outcome.json")
    )
    require(
        terminal["prior_debit_seconds"] == expected["prior_debit_seconds"] == debit,
        "Original time debit mismatch",
    )
    elapsed = terminal["correction_wall_seconds"]
    require(
        0 <= expected["packaged_elapsed_seconds"] <= elapsed <= cap
        and terminal["cumulative_charged_seconds"] == debit + elapsed <= 2700
        and terminal["cumulative_packaging_seconds"] == packaging_debit + elapsed <= 300,
        "Cumulative original budget exceeded",
    )
    require(
        terminal["original_guardian_sha256"] == sha(original / "guardian.json"),
        "Original timing receipt",
    )
    before = read(root / "correction/before.json")
    for name, identity in before.items():
        path = original / name
        if path.is_file():
            require(record(path) == identity, "Original preserved file changed")
        else:
            require(
                returned["omitted_on_Windows"].get("original/" + name) == identity,
                "Unbound omission",
            )
    require(
        read(root / "correction/preservation.json")["unchanged"] is True
        and read(root / "correction/preservation.json")["snapshot_sha256"]
        == sha(root / "correction/before.json"),
        "Preservation record",
    )
    prevent_execution()
    run = original / "run"
    omitted = {
        k.removeprefix("original/run/"): v
        for k, v in returned["omitted_on_Windows"].items()
        if k.startswith("original/run/")
    }
    reconstruction = {
        **verify_training(run, bundle=bundle, omitted=omitted),
        **verify_execution(run, bundle),
    }
    require(
        reconstruction
        == read(run / "verification.json")
        == read(root / "correction/verification.json"),
        "Execution reconstruction",
    )
    require(
        summarize_saved(run) == read(root / "correction/comparisons.json"),
        "Saved score reconstruction",
    )
    report = {
        "included_files_verified": len(returned["files"]),
        "omitted_not_rehashed": len(returned["omitted_on_Windows"]),
        "cumulative_charged_seconds": debit + elapsed,
        "original_packaging_failure_preserved": True,
        "new_optimizer_updates": 0,
        "new_model_scoring": False,
        "accepted_base": False,
        "phase18_authorized": False,
        "status": "corrective evidence verified; owner scientific review still required",
    }
    write(destination / "review.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "receipt", "guardian", "handoff", "correction-receipt", "destination"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(
        json.dumps(
            review(
                a.archive, a.receipt, a.guardian, a.handoff, a.correction_receipt, a.destination
            ),
            indent=2,
        )
    )
