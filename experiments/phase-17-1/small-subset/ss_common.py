"""Small-subset contract, immutable evidence and bounded process control (stdlib only)."""

import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

ATTEMPT = "small-subset-v1"
ARMS = {"A": (8, 1, 382), "B": (8, 2, 764), "C": (4, 2, 764)}
ENDPOINTS = {"A1": ("A", 382), "B1": ("B", 382), "B2": ("B", 764), "C2": ("C", 764)}
STAGES = {"prepare": 180, "train": 1440, "evaluate": 480, "package": 300}
TOTAL = 2700
TRAIN_TARGETS = 1943015
WINDOWS = 6105
BASE = "f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa"
PROTOCOL = "a83477dba5fb9ff26661384621e0bf07748dd36cb83069aa9fe9093865121b38"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path):
    return {"bytes": Path(path).stat().st_size, "sha256": sha(path)}


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def safe(root, name):
    path = Path(name)
    require(not path.is_absolute() and not any(c in name for c in ("\\", ":")), "Unsafe name")
    require(not any(p in ("..", ".private", ".venv") for p in path.parts), "Unsafe component")
    value = root / path
    require(value.resolve().is_relative_to(root.resolve()), "Path escape")
    require(not any(p.is_symlink() for p in [value, *value.parents]), "Symlink")
    return value


def verify(root, records):
    for name, expected in records.items():
        require(record(safe(root, name)) == expected, f"Changed input: {name}")


def budget(deadline, *, now=None):
    now = time.monotonic() if now is None else now
    if now >= deadline:
        raise TimeoutError("Study/stage deadline reached; no extension")
    return deadline - now


def accumulation(size, cursor, batch, maximum):
    require(0 <= cursor < size, "Pass exhausted")
    return min(maximum, math.ceil((size - cursor) / batch))


def boundary(step, size=WINDOWS, batch=2, accumulation_steps=8):
    per_pass = math.ceil(math.ceil(size / batch) / accumulation_steps)
    passes, remainder = divmod(step, per_pass)
    epoch = passes - 1 if step and not remainder else passes
    cursor = size if step and not remainder else remainder * batch * accumulation_steps
    return epoch, cursor, epoch * size + cursor


def tree_bytes(root):
    total = 0
    for p in root.rglob("*"):
        require(not p.is_symlink(), "Symlink in output")
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            pass  # Atomic completion of our own checkpoint staging.
    return total


def child(command, env, log_path, deadline, root, journal):
    budget(deadline)
    with log_path.open("xb") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                budget(deadline)
                require(tree_bytes(root) + 64 * 1024**2 < 8 * 1024**3, "Artifact cap")
                time.sleep(min(0.1, budget(deadline)))
            budget(deadline)
            journal({"event": "child-exit", "command": command, "code": proc.returncode})
            require(proc.returncode == 0, f"Child failed: {proc.returncode}")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=2)


def selfcheck():
    """No torch/model/optimizer; executes the controller's actual accounting helpers."""
    require(accumulation(6105, 6096, 2, 8) == 5, "Tail")
    require(boundary(382) == (0, 6105, 6105), "First boundary")
    require(boundary(383) == (1, 16, 6121), "Second pass")
    require(boundary(764) == (1, 6105, 12210), "Final boundary")
    require(sum(a[2] for a in ARMS.values()) == 1910, "Updates")
    require(sum(a[1] for a in ARMS.values()) * TRAIN_TARGETS == 9715075, "Exposure")
    require(sum(STAGES.values()) + 300 == TOTAL, "Cumulative budget")
    return {"passed": True, "optimizer_updates": 0, "kind": "scripted-accounting"}
