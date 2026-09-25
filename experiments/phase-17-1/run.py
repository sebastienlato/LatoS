"""One fixed Windows attempt: bounded preflight, fresh training, paired development.

Separate reviewed declarations are mandatory for any resume or final acceptance.
No historical supervisor is invoked, and no active status file is replaced.
"""

import argparse
import hashlib
import os
import secrets
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from common import (
    ATTEMPT,
    Journal,
    artifact_bytes,
    bounded_child,
    check_final_review,
    equal_replay,
    hash_file,
    lock,
    read,
    require,
    trace,
    verify_bundle,
    verify_records,
    write_once,
)


def checked_cpu_result(path, expected):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    require(
        len(cases) == expected
        and not any(c.find(k) is not None for c in cases for k in ("skipped", "error", "failure")),
        "CPU preflight failed or skipped",
    )
    return {"passed": len(cases), "skipped": 0, "optimizer_updates": 0, "sha256": hash_file(path)}


def fixture_jobs():
    jobs = []
    for runtime in ("checkout", "wheel"):
        for fixture in ("ordinary", "partial-tail"):
            ref = f"{runtime}-{fixture}-reference"
            for mode, updates in (("reference", 8), ("resumed", 5)):
                jobs.append(
                    {
                        "id": f"{runtime}-{fixture}-{mode}",
                        "runtime": runtime,
                        "kind": "fixture-" + mode,
                        "fixture": fixture,
                        "reference_id": ref,
                        "reserved_updates": updates,
                    }
                )
    require(sum(j["reserved_updates"] for j in jobs) == 52, "Fixture budget changed")
    return jobs


def compare_fixture(output, job):
    a = trace(output / "jobs" / job["reference_id"] / "updates.jsonl")
    b = trace(output / "jobs" / job["id"] / "updates.jsonl")
    require(len(a) == 8 and len(b) == 5, "Incomplete conformance pair")
    for old, new in zip(a[3:], b, strict=True):
        equal_replay(old, new)
    require(
        a[-1]["metric"]["microbatches"] == (1 if job["fixture"] == "partial-tail" else 2),
        "Fixture tail accounting differs",
    )


def job_updates_upper(output, req):
    root = output / "jobs" / req["id"]
    if (root / "result.json").exists():
        return read(root / "result.json")["optimizer_updates"]
    rows = (
        trace(root / "updates.jsonl", allow_torn=True) if (root / "updates.jsonl").exists() else []
    )
    # One update may have completed after the last complete durable record.
    return min(req["reserved_updates"], len(rows) + 1)


def update_accounting(output):
    result = {"preflight_updates_upper": 0, "serious_updates_upper": 0}
    for path in sorted((output / "requests").glob("*.json")):
        req = read(path)
        if not req["reserved_updates"]:
            continue
        completed = job_updates_upper(output, req)
        key = (
            "preflight_updates_upper"
            if req["kind"].startswith("fixture-")
            else "serious_updates_upper"
        )
        result[key] += completed
    return result


def review_resume(output, path, bundle_sha, held_files=None):
    require(path is not None, "Resume requires independent Mac review")
    r = read(path)
    require(
        r["kind"] == "phase17.1-external-interruption-Mac-review"
        and r["attempt_id"] == ATTEMPT
        and r["bundle_sha256"] == bundle_sha
        and r["external_interruption_verified"] is True,
        "Wrong resume review",
    )
    require(not (output / "training-result.json").exists(), "Training already complete")
    require(
        not list((output / "jobs").rglob("failure.json")), "Recorded worker failure is terminal"
    )
    sessions = sorted((output / "sessions").iterdir())
    prior_training, prior_evaluation = 0.0, 0.0
    prior_serious_upper = 0
    for s in sessions:
        receipt = s / "receipt.json"
        if receipt.exists():
            value = read(receipt)
            require(
                value["status"] in ("complete", "external-interruption"),
                "Terminal supervisor failure",
            )
            prior_training = max(prior_training, value["training_seconds"])
            prior_evaluation = max(prior_evaluation, value["evaluation_seconds"])
            prior_serious_upper = max(prior_serious_upper, value["serious_updates_upper"])
        else:
            events = trace(s / "events.jsonl", allow_torn=True)
            require(
                events[0]["stage"] != "final"
                and not any(e.get("event") == "evaluation-start" for e in events),
                "Interrupted evaluation cannot be retried",
            )
            prior_training += max(e.get("seconds", 0) for e in events)
    require(
        r["training_seconds_charged"] >= prior_training
        and r["evaluation_seconds_charged"] >= prior_evaluation,
        "Recovery time undercharged",
    )
    verify_records(output, r["prior_files"], held_files)
    # The review must bind all prior files, not an arbitrarily selected subset.
    existing = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    require(set(r["prior_files"]) == existing, "Incomplete prior preservation inventory")
    require(
        read(output / "preflight.json")["passed"] is True,
        "Preflight incomplete; no replay of tests",
    )
    current = update_accounting(output)
    require(
        current["preflight_updates_upper"] == 52
        and r["serious_updates_upper"]
        >= max(current["serious_updates_upper"], prior_serious_upper),
        "Recovery work undercharged",
    )
    checkpoints = sorted(
        (output / "checkpoints").glob("step-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]")
    )
    require(checkpoints, "No completed checkpoint")
    latest = checkpoints[-1]
    require(
        r["checkpoint"] == latest.relative_to(output).as_posix()
        and r["checkpoint_policy_sha256"] == hash_file(latest / "policy.json"),
        "Wrong recovery checkpoint",
    )
    step = read(latest / "state.json")["step"]
    remaining = 7485 - step
    require(
        r["serious_updates_upper"] + remaining <= 9485,
        "Physical serious-work cap would be exceeded",
    )
    overlap = []
    for p in sorted((output / "requests").glob("*.json")):
        req = read(p)
        if req["kind"] in ("train", "resume"):
            log = output / "jobs" / req["id"] / "updates.jsonl"
            if log.exists():
                overlap.append(log.relative_to(output).as_posix())
    return r, remaining, overlap


def run(args):
    began = time.monotonic()
    require(os.name == "nt", "Full Phase 17.1 execution is Windows-only; no Mac run")
    bundle = Path(__file__).resolve().parent
    plan, manifest = verify_bundle(bundle)
    transfer = args.transfer.resolve()
    output = transfer / "phase17-1-results"
    require(
        bundle.parent == transfer and bundle.name == "phase17-1-tools",
        "Use separate prescribed tools folder",
    )
    bundle_sha = hash_file(bundle / "bundle.json")
    initial = args.stage == "start"
    if initial:
        require(not output.exists(), "Attempt root already exists; no new attempt/retry")
        require(
            shutil.disk_usage(transfer).free >= 20 * 1024**3, "Need 20 GiB free before execution"
        )
        write_once(
            transfer / "phase17-1-launch.json",
            {
                "attempt_id": ATTEMPT,
                "bundle_sha256": bundle_sha,
                "controller_source_commit": manifest["source_commit"],
            },
        )
        output.mkdir()
        (output / "sessions").mkdir()
        (output / "requests").mkdir()
        (output / "jobs").mkdir()
    require(
        read(transfer / "phase17-1-launch.json")["bundle_sha256"] == bundle_sha,
        "Attempt source changed",
    )
    with lock(output / "supervisor.lock") as supervisor_lease:
        with lock(output / "worker.lock"):
            pass  # An orphaned worker blocks all continuation.
        execute_session(
            args,
            bundle,
            transfer,
            output,
            plan,
            manifest,
            began=began,
            held_files={"supervisor.lock": supervisor_lease},
        )


def execute_session(args, bundle, transfer, output, plan, manifest, *, began=None, held_files=None):
    began = time.monotonic() if began is None else began
    training_seconds, evaluation_seconds = 0.0, 0.0
    prior_serious = 0
    previous = sorted((output / "sessions").iterdir())
    resumes = sum(read(p / "start.json")["stage"] == "resume" for p in previous)
    bundle_sha = hash_file(bundle / "bundle.json")
    resume_info, remaining, overlap = None, None, None
    if args.stage == "resume":
        require(resumes < 2, "Two resumption ceiling reached")
        resume_info, remaining, overlap = review_resume(output, args.review, bundle_sha, held_files)
        training_seconds = resume_info["training_seconds_charged"]
        evaluation_seconds = resume_info["evaluation_seconds_charged"]
        prior_serious = resume_info["serious_updates_upper"]
    elif args.stage == "final":
        require(
            previous and all((p / "receipt.json").exists() for p in previous),
            "Prior session incomplete",
        )
        last = read(previous[-1] / "receipt.json")
        require(last["status"] == "complete", "Prior stage failed")
        training_seconds, evaluation_seconds = last["training_seconds"], last["evaluation_seconds"]
        require(
            not any(read(p)["kind"] == "final" for p in (output / "requests").glob("*.json")),
            "Final already attempted",
        )
        check_final_review(output, args.review, bundle_sha)
    else:
        require(not previous, "Fresh session already attempted")
    session = output / "sessions" / f"{len(previous):02d}"
    session.mkdir(exist_ok=False)
    write_once(
        session / "start.json",
        {
            "stage": args.stage,
            "attempt_id": ATTEMPT,
            "bundle_sha256": bundle_sha,
            "prior_training_seconds": training_seconds,
            "prior_evaluation_seconds": evaluation_seconds,
            "wall_start_unix": time.time(),
        },
    )
    if args.review:
        write_once(session / "Mac-review.json", read(args.review))
    journal = Journal(session / "events.jsonl")
    journal.add({"event": "start", "stage": args.stage, "bundle_sha256": bundle_sha})
    category = "evaluation" if args.stage == "final" else "training"
    clock_start = began
    environment = os.environ.copy()
    environment.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0")
    require(not environment.get("PYTHONPATH"), "Do not override package resolution")
    interpreters = {
        "checkout": transfer / "source/.venv/Scripts/python.exe",
        "wheel": transfer / "source/.wheel-env/Scripts/python.exe",
    }

    def dispatch(job):
        nonlocal training_seconds, evaluation_seconds
        require(all(p.is_file() for p in interpreters.values()), "Existing pinned runtimes missing")
        seconds = min(
            (7200 - training_seconds if category == "training" else 3600 - evaluation_seconds),
            10800 - training_seconds - evaluation_seconds,
        )
        token = secrets.token_hex(32)
        request = {
            **job,
            "bundle": str(bundle),
            "output": str(output),
            "transfer": str(transfer),
            "ticket_sha256": hashlib.sha256(token.encode()).hexdigest(),
        }
        path = output / "requests" / (job["id"] + ".json")
        write_once(path, request)
        journal.add(
            {
                "event": "job-start",
                "id": job["id"],
                "kind": job["kind"],
                "reserved_updates": job["reserved_updates"],
                "category": category,
            }
        )
        env = {
            **environment,
            "LATOS_PHASE171_TICKET": token,
            "LATOS_PHASE171_REQUEST_SHA256": hash_file(path),
        }
        command = [
            str(interpreters[job["runtime"]]),
            str(bundle / "worker.py"),
            "--request",
            str(path),
        ]
        if job["kind"] == "cpu":
            command = [
                str(interpreters[job["runtime"]]),
                "-m",
                "pytest",
                str(bundle / "test_controller.py"),
                "-q",
                "--junitxml",
                str(output / (job["id"] + ".xml")),
                "--basetemp",
                str(output / (job["id"] + "-fixtures")),
            ]
        with (output / (job["id"] + ".log")).open("xb") as log:
            bounded_child(
                command,
                env,
                log,
                journal,
                clock_start,
                seconds,
                lambda: artifact_bytes(output, bundle),
                20 * 1024**3,
                cwd=bundle,
            )
        if job["kind"] == "cpu":
            checked_cpu_result(output / (job["id"] + ".xml"), manifest["cpu_test_count"])
        else:
            result = read(output / "jobs" / job["id"] / "result.json")
            require(
                result["status"] == "complete"
                and result["optimizer_updates"] == job["reserved_updates"],
                "Worker completion mismatch",
            )
        journal.add(
            {
                "event": "job-complete",
                "id": job["id"],
                "category": category,
                "seconds": time.monotonic() - clock_start,
            }
        )

    status = "failed"
    try:
        if args.stage == "start":
            dispatch(
                {
                    "id": "integrity",
                    "kind": "integrity",
                    "runtime": "checkout",
                    "reserved_updates": 0,
                }
            )
            for runtime in ("checkout", "wheel"):
                dispatch(
                    {
                        "id": "cpu-" + runtime,
                        "kind": "cpu",
                        "runtime": runtime,
                        "reserved_updates": 0,
                    }
                )
            for job in fixture_jobs():
                dispatch(job)
                if job["kind"] == "fixture-resumed":
                    compare_fixture(output, job)
            write_once(
                output / "preflight.json",
                {
                    "passed": True,
                    "CUDA_updates": 52,
                    "bundle_sha256": bundle_sha,
                    "scope": "Fixed synthetic adapter checks, not learned quality",
                },
            )
            dispatch(
                {"id": "train-0", "kind": "train", "runtime": "checkout", "reserved_updates": 7485}
            )
        elif args.stage == "resume":
            dispatch(
                {
                    "id": f"train-{resumes + 1}",
                    "kind": "resume",
                    "runtime": "checkout",
                    "reserved_updates": remaining,
                    "resume_checkpoint": resume_info["checkpoint"],
                    "overlap_traces": overlap,
                }
            )
        if args.stage != "final":
            training_seconds += time.monotonic() - clock_start
            clock_start = time.monotonic()
            category = "evaluation"
            journal.add({"event": "evaluation-start", "training_seconds": training_seconds})
        dispatch(
            {
                "id": "final" if args.stage == "final" else "development",
                "kind": "final" if args.stage == "final" else "development",
                "runtime": "checkout",
                "reserved_updates": 0,
            }
        )
        status = "complete"
    except KeyboardInterrupt:
        status = "external-interruption"
        raise
    except BaseException as exc:
        journal.add({"event": "failure", "type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        elapsed = time.monotonic() - clock_start
        if category == "training":
            training_seconds += elapsed
        else:
            evaluation_seconds += elapsed
        if (
            training_seconds >= 7200
            or evaluation_seconds >= 3600
            or training_seconds + evaluation_seconds >= 10800
        ):
            status = "budget-exceeded"
        accounting = update_accounting(output)
        if prior_serious:
            # Preserve any conservative unlogged-work charge made by Mac review.
            new_request = output / "requests" / f"train-{resumes + 1}.json"
            new_count = job_updates_upper(output, read(new_request)) if new_request.exists() else 0
            accounting["serious_updates_upper"] = max(
                accounting["serious_updates_upper"], prior_serious + new_count
            )
        if accounting["serious_updates_upper"] > 9485 or accounting["preflight_updates_upper"] > 52:
            status = "budget-exceeded"
        value = {
            "status": status,
            "stage": args.stage,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            **accounting,
            "artifact_bytes_charged": artifact_bytes(output, bundle),
            "phase18_authorized": False,
            "publication_authorized": False,
            "next": "Return evidence to Mac review; never self-authorize final or retry",
        }
        if value["artifact_bytes_charged"] >= 20 * 1024**3:
            value["status"] = status = "budget-exceeded"
        write_once(session / "receipt.json", value)
        journal.add({"event": "session-end", "receipt": value})
        journal.close()
    require(status == "complete", "Session did not pass its execution bounds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--stage", choices=("start", "resume", "final"), default="start")
    parser.add_argument("--review", type=Path)
    run(parser.parse_args())
