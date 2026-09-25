"""Standalone Phase 17.1 integrity and append-only process controls; no Torch import."""

import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

ATTEMPT = "phase17-1-base2-attempt1"
PLAN = "117ad0d1e761114f3403113b1a7f371ad286d4cc681f72311d3c69455e6182e4"
PROPOSAL = "60446bd2885ccffb48b69d16223224896e1d71969bd74618907f5c4c399a0840"
ORIGINAL = "25c361f9668f55c1624894b33cd8351dc788a94b36d7fb9e05a88438bad87149"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_once(path, value):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def safe_path(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and not any(p in name for p in ("\\", ":")), "Unsafe path")
    require(
        not any(p in ("..", ".private", ".venv", ".wheel-env") for p in relative.parts),
        "Unsafe path",
    )
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), "Path escapes root")
    require(not any(p.is_symlink() for p in [path, *path.parents] if p != root.parent), "Symlink")
    return path


def record(path, held_stream=None):
    if held_stream is None:
        sha = hash_file(path)
    else:
        held_stream.seek(0)
        sha = hashlib.file_digest(held_stream, "sha256").hexdigest()
        held_stream.seek(0)
    return {"bytes": path.stat().st_size, "sha256": sha}


def verify_records(root, records, held_files=None):
    for name, expected in records.items():
        stream = (held_files or {}).get(name)
        require(record(safe_path(root, name), stream) == expected, f"Integrity mismatch: {name}")


def verify_bundle(bundle):
    manifest = read(bundle / "bundle.json")
    require(manifest["attempt_id"] == ATTEMPT, "Wrong bundle")
    verify_records(bundle, manifest["files"])
    require(hash_file(bundle / "plan.json") == PLAN, "Approved plan changed")
    require(hash_file(bundle / "PROPOSAL.md") == PROPOSAL, "Approved proposal changed")
    auth = read(bundle / "authorization.json")
    require(
        auth["owner_authorized"] is True
        and auth["attempt_id"] == ATTEMPT
        and auth["plan_sha256"] == PLAN
        and auth["proposal_sha256"] == PROPOSAL
        and auth["original_transfer_manifest_sha256"] == ORIGINAL
        and auth["phase18_authorized"] is False
        and auth["publication_authorized"] is False,
        "Missing matching owner authorization",
    )
    return read(bundle / "plan.json"), manifest


def verify_transfer(transfer, bundle, plan):
    require(hash_file(transfer / "handoff.json") == ORIGINAL, "Original transfer identity changed")
    require(hash_file(bundle / "original-transfer.json") == ORIGINAL, "Expected manifest changed")
    manifest = read(transfer / "handoff.json")
    verify_records(transfer, manifest["files"])
    for name, sha in plan["identities"].items():
        require(hash_file(transfer / "source" / name) == sha, f"Frozen input changed: {name}")
    return manifest


def file_size(path):
    try:
        require(not path.is_symlink(), "Symlink in artifact root")
        return path.stat().st_size if path.is_file() else 0
    except FileNotFoundError:
        return 0  # An atomic checkpoint rename can race this read-only size scan.


def tree_bytes(root):
    return sum(file_size(p) for p in root.rglob("*"))


def artifact_bytes(output, bundle):
    return tree_bytes(output) + 2 * tree_bytes(bundle) + 1024**3


@contextmanager
def lock(path):
    with Path(path).open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield stream
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Journal:
    def __init__(self, path):
        self.stream = path.open("xb")

    def add(self, value):
        self.stream.write(canonical(value))
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def bounded_child(command, environment, log, journal, started, seconds, measure, cap, **kwargs):
    require(time.monotonic() - started < seconds and measure() < cap, "Budget exhausted")
    child = subprocess.Popen(
        command, env=environment, stdout=log, stderr=subprocess.STDOUT, **kwargs
    )
    try:
        while child.poll() is None:
            elapsed, used = time.monotonic() - started, measure()
            journal.add({"event": "running", "seconds": elapsed, "artifact_bytes": used})
            if elapsed >= seconds or used >= cap:
                raise TimeoutError("Cumulative active time/artifact budget reached")
            time.sleep(0.5)
        require(child.returncode == 0, f"Worker exited {child.returncode}; stop for review")
        require(time.monotonic() - started < seconds and measure() < cap, "Budget reached at exit")
    except BaseException:
        if child.poll() is None:
            child.kill()
        child.wait()
        raise


def trace(path, *, allow_torn=False):
    raw = path.read_bytes()
    rows = raw.splitlines(keepends=True)
    result = []
    for index, row in enumerate(rows):
        if not row.endswith(b"\n"):
            require(allow_torn and index == len(rows) - 1, "Torn trace")
            break
        result.append(json.loads(row))
    return result


def equal_replay(old, new):
    require(
        old["input"] == new["input"] and old["state"] == new["state"],
        "Replay full state/input differs",
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
        require(old["metric"][key] == new["metric"][key], "Replay counter/schedule differs")
    import math

    delta = abs(old["metric"]["loss"] - new["metric"]["loss"])
    require(math.isfinite(delta) and delta <= 1e-5, "Replay loss exceeds frozen 1e-5")


def check_final_review(output, path, bundle_sha):
    require(path is not None, "Final requires independent Mac review")
    review = read(path)
    candidate = read(output / "training-result.json")["candidate_weights"]
    require(
        review["kind"] == "phase17.1-final-Mac-review"
        and review["attempt_id"] == ATTEMPT
        and review["bundle_sha256"] == bundle_sha
        and review["candidate_weights"] == candidate
        and review["accepted_for_final"] is True,
        "Wrong final review",
    )
    require(
        read(output / "development-comparison.json")["passed"] is True
        and read(output / "inference.json")["passed"] is True,
        "Development/resource gate failed",
    )
    for name in ("development-comparison.json", "inference.json", "acceptance-plan.json"):
        require(hash_file(output / name) == review["files"][name], "Final review evidence changed")
    selection = read(output / "acceptance-plan.json")
    require(
        selection["phase"] == 17
        and selection["candidate_weights"] == candidate
        and selection["reference_weights"]
        == ["f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa"],
        "Wrong final phase or locked model panel",
    )
    return review
