"""Mac saved-evidence review: no returned code execution, model loading or scoring."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from cf_common import read, record, require, safe, sha, verify, write
from cf_evidence import summarize, verify_execution, verify_training


def review(archive, receipt, handoff, destination, completed, guardian):
    require(
        record(archive) == {k: read(receipt)[k] for k in ("bytes", "sha256")}, "Return checksum"
    )
    terminal, outer = read(completed), read(guardian)
    require(terminal["return_receipt_sha256"] == sha(receipt), "Completed receipt binding")
    require(
        0 <= terminal["total_elapsed_seconds"] <= outer["total_wall_seconds"] <= 1800,
        "End-to-end wall limit violated",
    )
    require(not outer.get("deadline_terminated", False), "Guardian timed out")
    require(
        outer["exit_code"] == (0 if terminal["execution_complete"] else 2), "Guardian disposition"
    )
    destination.mkdir(exist_ok=False)
    bundle = destination / "reviewed-bundle"
    bundle.mkdir()
    with zipfile.ZipFile(handoff) as z:
        manifest = json.loads(z.read("bundle.json"))
        require(
            set(z.namelist()) == set(manifest["files"]) | {"bundle.json"}, "Known handoff members"
        )
        for name in z.namelist():
            path = safe(bundle, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(name))
    verify(bundle, manifest["files"])
    root = destination / "run"
    root.mkdir()
    with zipfile.ZipFile(archive) as z:
        returned = json.loads(z.read("return.json"))
        require(
            len(z.namelist()) == len(set(z.namelist()))
            and set(z.namelist()) == set(returned["files"]) | {"return.json"},
            "Return members",
        )
        for name, expected in returned["files"].items():
            path = safe(root, name)
            raw = z.read(name)
            require(
                len(raw) == expected["bytes"]
                and hashlib.sha256(raw).hexdigest() == expected["sha256"],
                "Return member hash",
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    require(read(root / "study-manifest.json") == manifest, "Return source binding")
    require(
        returned["outcome"]["execution_complete"] == terminal["execution_complete"],
        "Outcome mismatch",
    )
    start = read(root / "guardian/launch.json")["started_monotonic"]
    require(
        start <= read(receipt)["packaged_monotonic"] <= start + terminal["total_elapsed_seconds"],
        "Packaging outside original clock",
    )
    report = {
        "included_verified": len(returned["files"]),
        "omitted_not_rehashed": len(returned["omitted_on_Windows"]),
        "optimizer_updates": 0,
        "new_model_scoring": False,
        "accepted_base": False,
    }
    if returned["outcome"]["execution_complete"]:
        report["training"] = verify_training(
            root, bundle=bundle, omitted=returned["omitted_on_Windows"]
        )
        report["execution"] = verify_execution(root, bundle)
        rebuilt = summarize(root)
        require(rebuilt == read(root / "comparisons.json"), "Saved comparison reconstruction")
        report["comparisons_reconstructed"] = True
    else:
        report["disposition"] = "incomplete diagnostic; preserve failure"
    write(destination / "review.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "receipt", "handoff", "destination", "completed", "guardian"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            review(a.archive, a.receipt, a.handoff, a.destination, a.completed, a.guardian),
            indent=2,
        )
    )
