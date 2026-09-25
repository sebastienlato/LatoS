"""One owner-authorized, cumulative-budget packaging correction. No model execution."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
ORIGINAL_COMMIT = "15cf9d96e8baa00cdc29e76840e283c2ab4bc814"
BUNDLE = "786a5dfa7b9a88270d26195675c2229d5a1bc396c769c0ba7617f350ca29250a"
GUARDIAN = "e4db1825629a34b586da867cedabf07c99808d5672f6f2e298c4c79ff532a6b4"
ERROR = "Base gate requires the frozen Phase 5 historical reference"
REPORTED_SECONDS = 1273  # Owner's 21m13s; floor, never substituted for a larger receipt.
TOTAL = 2700
CORRECTION_CAP = 300
ENDPOINTS = ("A1", "B1", "B2", "C2")
CLAIM = "small-subset-packaging-fix1"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def record(path):
    return {"bytes": path.stat().st_size, "sha256": sha(path)}


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def charge(guardian):
    elapsed = guardian["total_wall_seconds"]
    require(
        type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
        "Invalid original timing",
    )
    require(
        guardian["exit_code"] == 1 and not guardian.get("deadline_terminated", False),
        "Original exit is not the reported packaging exception",
    )
    debit = max(REPORTED_SECONDS, math.ceil(elapsed))
    require(debit < TOTAL, "No remaining original budget")
    return debit, min(CORRECTION_CAP, TOTAL - debit)


def budget_ledger(guardian, outcome):
    debit, remaining_cap = charge(guardian)
    before = outcome["elapsed_before_package"]
    require(
        type(before) in (int, float)
        and math.isfinite(before)
        and 0 <= before <= guardian["total_wall_seconds"],
        "Invalid original packaging start",
    )
    packaging_debit = math.ceil(debit - before)
    cap = min(remaining_cap, CORRECTION_CAP - packaging_debit)
    require(cap > 10, "No usable original packaging allowance remains")
    return debit, cap, packaging_debit


def summarize_saved(root):
    """Keep all four primary gate calls; descriptive contrasts never invoke a gate."""
    from latos.evaluation.compare import compare, paired_interval

    for name in ("reference", *ENDPOINTS):
        manifest = read(root / f"evaluation-{name}/manifest.json")
        require(
            manifest["final_access"] is None
            and manifest["data"]["split"] == "validation"
            and manifest["suite_sha256"] == manifest["protocol"]["suite_sha256"],
            "Development-only saved outputs required",
        )
    primary = {
        name: compare(root / "evaluation-reference", root / f"evaluation-{name}", "base")
        for name in ENDPOINTS
    }
    # Primary comparisons already verify every saved run's inventory/rows/aggregates
    # and its identities/runtime against the SAME historical reference. Compatibility
    # is therefore transitive. paired_interval also requires identical ordered IDs.
    benchmarks = read(root / "evaluation-reference/summary.json")["external"]
    contrasts = {}
    for before, after in (("A1", "B1"), ("B1", "B2"), ("A1", "B2"), ("B2", "C2")):
        a, b = root / f"evaluation-{before}", root / f"evaluation-{after}"
        paired = {
            name: paired_interval(read(a / f"{name}.json"), read(b / f"{name}.json"))
            for name in benchmarks
        }
        paired["instructions"] = paired_interval(
            read(a / "instructions.json"), read(b / "instructions.json"), cluster_key="family"
        )
        contrasts[f"{before}:{after}"] = {
            "paired": paired,
            "scope": "diagnostic only; not acceptance reference",
        }
    return {
        "primary_reference": "preserved Phase 5",
        "fixed_gate_comparisons": primary,
        "descriptive_contrasts": contrasts,
        "accepted_base": False,
        "phase18_authorized": False,
    }


def diagnosis(original):
    """Refuse a different failure; the owner summary alone never clears prerequisites."""
    run = original / "run"
    outcome = read(run / "outcome.json")
    debit, cap, packaging_debit = budget_ledger(read(original / "guardian.json"), outcome)
    require(
        outcome["execution_complete"] is True
        and outcome["failure"] is None
        and outcome["accepted_base"] is False
        and outcome["final_scored"] is False,
        "Pre-packaging execution incomplete; no repair/training permitted",
    )
    require((original / "bootstrap.log").is_file(), "Missing preserved bootstrap log")
    require(
        "Child failed: 1"
        in (original / "bootstrap.log").read_text(encoding="utf-8", errors="replace"),
        "Bootstrap does not record the reported packaging child failure",
    )
    log = (run / "package.log").read_text(encoding="utf-8", errors="replace")
    require(
        ERROR in log and "diagnostic = compare" in log and "in summarize" in log,
        "Traceback does not identify the cross-arm packaging bug",
    )
    events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    phases = [r["event"] for r in events if r["event"].endswith("-complete")]
    require(
        phases == ["prepare-complete", "training-complete", "evaluation-complete"],
        "Original stage sequence incomplete",
    )
    require(not any(r["event"] == "study-failure" for r in events), "Earlier study failure")
    require(
        events[-1]["elapsed"] <= outcome["elapsed_before_package"] <= debit,
        "Original time ordering",
    )
    pack_events = [json.loads(x) for x in (run / "package-events.jsonl").read_text().splitlines()]
    require(
        len(pack_events) == 1
        and pack_events[0]["event"] == "child-exit"
        and pack_events[0]["code"] == 1,
        "Packaging process exit mismatch",
    )
    require((run / "verification.json").is_file(), "Original reconstruction did not finish")
    for name in (
        "comparisons.json",
        "small-subset-return.zip",
        "return.receipt.json",
        "completed.json",
    ):
        require(
            not (run / name).exists(), f"Unexpected original artifact; preserve and stop: {name}"
        )
    return {
        "prior_debit_seconds": debit,
        "correction_cap_seconds": cap,
        "reported_seconds_floor": REPORTED_SECONDS,
        "prior_packaging_debit_seconds": packaging_debit,
        "diagnosis": "cross-arm call to base gate",
    }


def inventory(root, deadline):
    result = {}
    require(not root.is_symlink(), "Symlink root")
    for path in sorted(root.rglob("*")):
        require(time.monotonic() < deadline, "Correction deadline")
        require(not path.is_symlink(), "Symlink in preserved evidence")
        require(
            not any(
                x in (".private", ".git", ".venv", ".env") for x in path.relative_to(root).parts
            ),
            "Unexpected private/environment material in study root",
        )
        if path.is_file():
            result[path.relative_to(root).as_posix()] = record(path)
    return result


def prevent_execution():
    """Tripwires against accidental inference, training, model loading and CUDA."""
    import torch

    import latos.evaluation.runner as evaluator
    import latos.model as model_api
    import latos.model.storage as storage
    from latos.model.network import LatoModel
    from latos.training.engine import Trainer

    def forbidden(*args, **kwargs):
        raise RuntimeError("Packaging only: model/optimizer/CUDA execution prohibited")

    Trainer.update = forbidden
    torch.optim.AdamW.step = forbidden
    LatoModel.forward = forbidden
    model_api.create_model = forbidden
    storage.load_model = forbidden
    evaluator.run = forbidden
    torch.cuda._lazy_init = forbidden


def worker(args, original, destination):
    ticket = os.environ.pop("LATOS_PACKAGING_ONLY", "")
    auth = read(destination / "authorization.json")
    require(
        ticket
        and hashlib.sha256(ticket.encode()).hexdigest() == auth["ticket_sha256"]
        and auth["corrector_sha256"] == args.sha256
        and auth["origin"] == args.origin
        and auth["prior_debit_seconds"] == args.debit,
        "Guardian ticket/clock binding",
    )
    require(sha(Path(__file__)) == args.sha256, "Corrector source differs")
    require(time.monotonic() < args.deadline, "Correction deadline before checks")
    checked = diagnosis(original)
    require(checked["prior_debit_seconds"] == args.debit, "Original budget binding changed")
    require(
        args.deadline <= args.origin + checked["correction_cap_seconds"] - 10,
        "Invalid corrective deadline",
    )
    tools = original / "run/tools"
    require(sha(tools / "bundle.json") == BUNDLE, "Original bundle changed")
    manifest = read(tools / "bundle.json")
    require(manifest["source_commit"] == ORIGINAL_COMMIT, "Unexpected original source")
    for name, expected in manifest["files"].items():
        require(record(tools / name) == expected, f"Original tool/cache changed: {name}")
    before = inventory(original, args.deadline)
    write(destination / "before.json", before)
    write(destination / "diagnosis.json", checked)
    sys.path.insert(0, str(tools))
    from ss_evidence import verify_execution, verify_training

    from latos.training.checkpoint import implementation_hash

    require(
        implementation_hash()
        == read(tools / "legacy-plan.json")["source"]["original_core_implementation_sha256"],
        "Native implementation changed",
    )
    prevent_execution()
    run = original / "run"
    verified = {**verify_training(run), **verify_execution(run, tools)}
    require(verified == read(run / "verification.json"), "Original verification disagrees")
    write(destination / "verification.json", verified)
    write(destination / "comparisons.json", summarize_saved(run))
    require(time.monotonic() < args.deadline, "Correction deadline after saved-row reconstruction")
    paths, included, omitted = {}, {}, {}
    retained_models = {
        f"run/jobs/{arm}/step-{step:08d}/model/model.safetensors"
        for arm, step in (("A", 382), ("B", 382), ("B", 764), ("C", 764))
    }
    for name, expected in before.items():
        key = "original/" + name
        excluded = name.endswith((".u32", ".u64", ".pyc")) or (
            name.endswith(".safetensors") and name not in retained_models
        )
        if excluded:
            omitted[key] = expected
        else:
            included[key] = expected
            paths[key] = original / name
    for name in (
        "before.json",
        "diagnosis.json",
        "verification.json",
        "comparisons.json",
        "authorization.json",
    ):
        paths["correction/" + name] = destination / name
        included["correction/" + name] = record(destination / name)
    paths["correction/correct.py"] = Path(__file__)
    included["correction/correct.py"] = record(Path(__file__))
    projected = sum(x["bytes"] for x in before.values()) + sum(
        x["bytes"] for x in included.values()
    )
    require(projected + 128 * 1024**2 < 8 * 1024**3, "Combined original/corrective artifact cap")
    archive = destination / "small-subset-corrected-return.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as z:
        for name, path in paths.items():
            require(time.monotonic() < args.deadline, "Correction deadline during packaging")
            z.write(path, name)
        after = inventory(original, args.deadline)
        require(after == before, "Preserved study was modified; stop")
        preservation = {
            "unchanged": True,
            "original_files": len(before),
            "snapshot_sha256": sha(destination / "before.json"),
        }
        write(destination / "preservation.json", preservation)
        included["correction/preservation.json"] = record(destination / "preservation.json")
        z.write(destination / "preservation.json", "correction/preservation.json")
        returned = {
            "kind": "small-subset-packaging-correction-v1",
            "files": included,
            "omitted_on_Windows": omitted,
            "original_bundle_sha256": BUNDLE,
            "correction_source_sha256": args.sha256,
            "prior_debit_seconds": args.debit,
            "original_packaging_failure_preserved": True,
            "additional_optimizer_updates": 0,
            "new_model_scoring": False,
            "accepted_base": False,
            "phase18_authorized": False,
        }
        z.writestr("return.json", json.dumps(returned, indent=2) + "\n")
    with zipfile.ZipFile(archive) as z:
        require(set(z.namelist()) == set(included) | {"return.json"}, "Return members")
        for name, expected in included.items():
            require(time.monotonic() < args.deadline, "Correction deadline during readback")
            with z.open(name) as f:
                require(
                    hashlib.file_digest(f, "sha256").hexdigest() == expected["sha256"],
                    "Return readback",
                )
    require(time.monotonic() < args.deadline, "Correction deadline before receipt")
    receipt = {
        **record(archive),
        "verified": True,
        "files": len(included) + 1,
        "omitted_files": len(omitted),
        "original_unchanged": True,
        "prior_debit_seconds": args.debit,
        "additional_optimizer_updates": 0,
        "new_model_scoring": False,
        "packaged_elapsed_seconds": time.monotonic() - args.origin,
    }
    write(destination / "return.receipt.json", receipt)
    require(time.monotonic() < args.deadline, "Correction deadline at closure")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transfer", required=True, type=Path)
    p.add_argument("--origin", required=True, type=float)
    p.add_argument("--sha256", required=True)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--debit", type=int, help=argparse.SUPPRESS)
    p.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = p.parse_args()
    require(os.name == "nt", "Windows correction only")
    require(
        "QueryPerformanceCounter" in time.get_clock_info("monotonic").implementation,
        "Unsupported clock",
    )
    require(
        0 <= time.monotonic() - args.origin < CORRECTION_CAP, "Expired/missing correction origin"
    )
    require(sha(Path(__file__)) == args.sha256, "Corrective file checksum differs")
    original = args.transfer.resolve() / "small-subset-v1"
    destination = args.transfer.resolve() / CLAIM
    if args.worker:
        with (destination / "worker.log").open("x", encoding="utf-8") as log:
            sys.stdout = sys.stderr = log
            try:
                worker(args, original, destination)
            except BaseException as exc:
                import traceback

                traceback.print_exc()
                write(
                    destination / "failure.json",
                    {
                        "type": type(exc).__name__,
                        "error": str(exc),
                        "elapsed_seconds": time.monotonic() - args.origin,
                        "additional_optimizer_updates": 0,
                        "new_model_scoring": False,
                    },
                )
                raise
        return
    destination.mkdir(exist_ok=False)  # One packaging-only attempt; no reset or reuse.
    debit, cap, packaging_debit = budget_ledger(
        read(original / "guardian.json"), read(original / "run/outcome.json")
    )
    require(cap > 10 and time.monotonic() < args.origin + cap - 10, "Insufficient remaining budget")
    ticket = os.urandom(32).hex()
    write(
        destination / "authorization.json",
        {
            "scope": "Owner-authorized packaging-only correction",
            "ticket_sha256": hashlib.sha256(ticket.encode()).hexdigest(),
            "prior_reported_seconds": REPORTED_SECONDS,
            "prior_debit_seconds": debit,
            "prior_packaging_debit_seconds": packaging_debit,
            "original_guardian": record(original / "guardian.json"),
            "correction_cap_seconds": cap,
            "origin": args.origin,
            "absolute_cumulative_limit_seconds": TOTAL,
            "corrector_sha256": args.sha256,
            "additional_optimizer_updates": 0,
            "new_model_scoring": False,
            "publication_authorized": False,
            "phase18_authorized": False,
        },
    )
    guard = original / "run/tools/launch.py"
    require(sha(guard) == GUARDIAN, "Pinned original deadline guardian changed")
    spec = importlib.util.spec_from_file_location("packaging_deadline_guard", guard)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.environ.update(
        PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", LATOS_PACKAGING_ONLY=ticket
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--transfer",
        str(args.transfer.resolve()),
        "--origin",
        str(args.origin),
        "--sha256",
        args.sha256,
        "--worker",
        "--debit",
        str(debit),
        "--deadline",
        str(args.origin + cap - 10),
    ]
    try:
        result = module.guarded(command, args.origin + cap - 5)
    except BaseException as exc:
        result = {"exit_code": 125, "error": str(exc)}
    elapsed = time.monotonic() - args.origin
    result.update(
        correction_wall_seconds=elapsed,
        prior_debit_seconds=debit,
        cumulative_charged_seconds=debit + elapsed,
        cumulative_packaging_seconds=packaging_debit + elapsed,
        original_guardian_sha256=sha(original / "guardian.json"),
        additional_optimizer_updates=0,
        new_model_scoring=False,
    )
    if result["exit_code"] == 0:
        receipt = destination / "return.receipt.json"
        if not receipt.is_file() or elapsed > cap or debit + elapsed > TOTAL:
            result.update(exit_code=124, error="Missing receipt or invalid completed budget")
        else:
            result["return_receipt_sha256"] = sha(receipt)
    write(destination / "guardian.json", result)
    print(json.dumps(result))
    raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
