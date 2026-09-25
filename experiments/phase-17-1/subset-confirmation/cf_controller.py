"""One append-only study; timed preparation, fixed jobs, then timed evidence return."""

import argparse
import hashlib
import os
import secrets
import shutil
import sys
import time
import zipfile
from pathlib import Path

from cf_common import (
    ARMS,
    ENDPOINTS,
    STAGES,
    budget,
    canonical,
    child,
    read,
    require,
    safe,
    sha,
    verify,
    write,
)


def fixtures():
    jobs = []
    for runtime in ("checkout", "wheel"):
        for fixture in ("ordinary", "partial-tail"):
            reference = f"{runtime}-{fixture}-reference"
            for mode, count in (("reference", 8), ("resumed", 5)):
                jobs.append(
                    {
                        "id": f"{runtime}-{fixture}-{mode}",
                        "kind": f"fixture-{mode}",
                        "runtime": runtime,
                        "fixture": fixture,
                        "reference": reference,
                        "updates_reserved": count,
                    }
                )
    return jobs


def check_fixture(root, job):
    from cf_evidence import rows

    a, b = (rows(root / "jobs" / name / "updates.jsonl") for name in (job["reference"], job["id"]))
    require(len(a) == 8 and len(b) == 5, "Fixture count")
    for left, right in zip(a[3:], b, strict=True):
        require(
            left["input"] == right["input"] and left["state"] == right["state"], "Fixture replay"
        )
        for key in (
            "step",
            "targets",
            "tokens_seen",
            "windows_seen",
            "epoch",
            "microbatches",
            "learning_rate",
        ):
            require(left["metric"][key] == right["metric"][key], "Fixture counters")
        require(abs(left["metric"]["loss"] - right["metric"]["loss"]) <= 1e-5, "Fixture tolerance")
    require(
        a[-1]["metric"]["microbatches"] == (1 if job["fixture"] == "partial-tail" else 2),
        "Fixture tail",
    )


def extract(archive, tools, digest):
    require(sha(archive) == digest, "Transfer ZIP hash differs")
    tools.mkdir()
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist()) == len(set(z.namelist())), "Duplicate members")
        require(sum(i.file_size for i in z.infolist()) < 64 * 1024**2, "Oversized transfer")
        for name in z.namelist():
            dest = safe(tools, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as f:
                f.write(z.read(name))
    manifest = read(tools / "bundle.json")
    require(set(z.namelist()) == set(manifest["files"]) | {"bundle.json"}, "Transfer members")
    verify(tools, manifest["files"])
    require(manifest["attempt"] == "subset-confirmation-v1", "Wrong study")
    return manifest


def run(args):
    require(
        os.name == "nt" and os.environ.pop("LATOS_SUBSET_JOB_ACTIVE", "") == "1",
        "Launch only through the Windows deadline guardian",
    )
    started = args.started
    deadline = started + 1790  # Outer OS job kills the full process tree before 30 minutes.
    root = args.output.resolve()
    root.mkdir(exist_ok=False)
    (root / "jobs").mkdir()
    (root / "requests").mkdir()
    events = (root / "events.jsonl").open("xb")

    def journal(value):
        events.write(canonical({"elapsed": time.monotonic() - started, **value}))
        events.flush()
        os.fsync(events.fileno())

    transfer = args.transfer.resolve()
    tools = root / "tools"
    failed = None
    try:
        prepare_deadline = min(deadline, started + STAGES["prepare"])
        manifest = extract(args.archive, tools, args.sha256)
        require(
            sha(Path(__file__)) == manifest["files"]["cf_controller.py"]["sha256"],
            "Bootstrap controller differs",
        )
        require(shutil.disk_usage(root).free >= 12 * 1024**3, "Free disk prerequisite")
        budget(prepare_deadline)
        auth = read(tools / "authorization.json")
        require(
            auth["authorized"] is True
            and auth["attempt"] == "subset-confirmation-v1"
            and auth["proposal_sha256"] == sha(tools / "PROPOSAL.md")
            and auth["hard_wall_seconds"] == 1800
            and auth["new_paid_compute"] == 0
            and auth["phase18_authorized"] is False
            and auth["publication_authorized"] is False,
            "Authorization binding",
        )
        write(root / "study-manifest.json", manifest)
        environment = dict(os.environ)
        environment.update(
            CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1"
        )

        def dispatch(spec, stop):
            budget(stop)
            token = secrets.token_hex(32)
            request = {
                **spec,
                "bundle": str(tools),
                "transfer": str(transfer),
                "output": str(root),
                "deadline": stop,
                "ticket": hashlib.sha256(token.encode()).hexdigest(),
            }
            path = root / "requests" / (spec["id"] + ".json")
            write(path, request)
            journal(
                {
                    "event": "job-start",
                    "id": spec["id"],
                    "updates_reserved": spec.get("updates_reserved", 0),
                }
            )
            python = (
                transfer
                / "source"
                / (".venv" if spec["runtime"] == "checkout" else ".wheel-env")
                / "Scripts/python.exe"
            )
            env = {**environment, "LATOS_SUBSET_TICKET": token, "LATOS_SUBSET_REQUEST": sha(path)}
            child(
                [str(python), str(tools / "cf_worker.py"), str(path)],
                env,
                root / (spec["id"] + ".log"),
                stop,
                root.parent,
                journal,
            )
            result = read(root / "jobs" / spec["id"] / "result.json")
            require(
                result["optimizer_updates"] == spec.get("updates_reserved", 0),
                "Physical work count",
            )

        dispatch({"id": "prepare", "kind": "prepare", "runtime": "checkout"}, prepare_deadline)
        for spec in fixtures():
            dispatch(spec, prepare_deadline)
            if spec["kind"] == "fixture-resumed":
                check_fixture(root, spec)
        write(root / "preflight.json", {"passed": True, "optimizer_updates": 52})
        journal({"event": "prepare-complete"})
        stop = min(
            deadline - STAGES["evaluate"] - STAGES["package"], time.monotonic() + STAGES["train"]
        )
        for arm, (_, _, count) in ARMS.items():
            dispatch(
                {
                    "id": arm,
                    "arm": arm,
                    "kind": "train",
                    "runtime": "checkout",
                    "updates_reserved": count,
                },
                stop,
            )
        journal({"event": "training-complete"})
        stop = min(deadline - STAGES["package"], time.monotonic() + STAGES["evaluate"])
        for name in ("reference", *ENDPOINTS):
            dispatch(
                {"id": "eval-" + name, "endpoint": name, "kind": "evaluate", "runtime": "checkout"},
                stop,
            )
        journal({"event": "evaluation-complete"})
    except BaseException as exc:
        failed = {"type": type(exc).__name__, "error": str(exc)}
        journal({"event": "study-failure", **failed})
    finally:
        events.close()
    # Packaging is part of this invocation and deadline. Never hand it to a later task.
    status = {
        "execution_complete": failed is None,
        "failure": failed,
        "accepted_base": False,
        "phase18_authorized": False,
        "final_scored": False,
        "base_model_acceptance_run": False,
        "further_fixed_subset_exposure_authorized": False,
        "owner_review_required": True,
        "elapsed_before_package": time.monotonic() - started,
    }
    write(root / "outcome.json", status)
    package_deadline = min(deadline, time.monotonic() + STAGES["package"])
    env = dict(os.environ)
    env["LATOS_SUBSET_PACKAGE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # If extraction failed there may be no bundle. The bootstrap carries stdlib packager too.
    script = (
        tools / "cf_evidence.py"
        if (tools / "cf_evidence.py").exists()
        else Path(__file__).with_name("cf_evidence.py")
    )
    with (root / "package-events.jsonl").open("xb") as packlog:

        def packjournal(value):
            packlog.write(canonical(value))
            packlog.flush()

        child(
            [sys.executable, str(script), "package", str(root), str(package_deadline)],
            env,
            root / "package.log",
            package_deadline,
            root.parent,
            packjournal,
        )
    budget(deadline)
    require((root / "return.receipt.json").is_file(), "No completed verified return")
    write(
        root / "completed.json",
        {
            **status,
            "total_elapsed_seconds": time.monotonic() - started,
            "return_receipt_sha256": sha(root / "return.receipt.json"),
        },
    )

    if failed is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transfer", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--started", type=float, required=True)
    run(p.parse_args())
