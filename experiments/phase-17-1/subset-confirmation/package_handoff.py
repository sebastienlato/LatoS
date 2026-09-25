"""Allowlisted local transfer from a clean reviewed Git commit plus verified subset cache."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from cf_common import read, record, require, verify, write

DIRECTORY = Path(__file__).resolve().parent
REPO = DIRECTORY.parents[2]
CODE = (
    "launch.py",
    "cf_common.py",
    "cf_adapter.py",
    "cf_worker.py",
    "cf_controller.py",
    "cf_evidence.py",
    "cache-identities.json",
    "required-inputs.json",
    "authorization.json",
    "WINDOWS.md",
)


def package(cache, output):
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip(),
        "Clean reviewed source required",
    )
    require(not output.exists(), "New output only")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO).strip()
    sources = {name: (DIRECTORY / name).relative_to(REPO).as_posix() for name in CODE}
    sources.update(
        {
            "legacy/common.py": "experiments/phase-17-1/common.py",
            "legacy/runtime.py": "experiments/phase-17-1/runtime.py",
            "legacy/fingerprints.py": "experiments/recovery-diagnostic/mechanics.py",
            "legacy-plan.json": "experiments/phase-17-1/plan.json",
            "fixture-identities.json": "experiments/phase-17-1/fixture-identities.json",
            "PROPOSAL.md": "experiments/phase-17-1/subset-confirmation/PLAN.md",
            "LICENSE": "LICENSE",
        }
    )
    cache_manifest = read(DIRECTORY / "cache-identities.json")
    verify(cache, cache_manifest["files"])
    manifest = {
        "attempt": "subset-confirmation-v1",
        "source_commit": commit,
        "source_paths": sources,
        "files": {},
        "publication_authorized": False,
        "phase18_authorized": False,
    }
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, source in sources.items():
            raw = subprocess.check_output(["git", "show", f"{commit}:{source}"], cwd=REPO)
            manifest["files"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            z.writestr(name, raw)
        for name, identity in cache_manifest["files"].items():
            manifest["files"]["cache/" + name] = identity
            z.write(cache / name, "cache/" + name)
        z.writestr("bundle.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    with zipfile.ZipFile(output) as z:
        require(set(z.namelist()) == set(manifest["files"]) | {"bundle.json"}, "Archive members")
        for name, identity in manifest["files"].items():
            raw = z.read(name)
            require(
                len(raw) == identity["bytes"]
                and hashlib.sha256(raw).hexdigest() == identity["sha256"],
                "Readback",
            )
    launcher = output.with_name("subset-confirmation-launch.py")
    require(not launcher.exists(), "Launcher output exists")
    launcher.write_bytes((DIRECTORY / "launch.py").read_bytes())
    receipt = {
        "source_commit": commit,
        "archive": {"name": output.name, **record(output)},
        "launcher": {"name": launcher.name, **record(launcher)},
        "members": len(manifest["files"]) + 1,
        "scope": (
            "Local transfer: controller and fixed existing subset cache; "
            "no weights, final payload or private planning"
        ),
        "hard_wall_seconds": 1800,
        "publication_authorized": False,
    }
    write(output.with_suffix(".receipt.json"), receipt)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(package(args.cache, args.output), indent=2))
