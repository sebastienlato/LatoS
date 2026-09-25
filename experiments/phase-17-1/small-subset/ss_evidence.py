"""Bounded saved-record verification and compact evidence return; no training/scoring."""

import hashlib
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

from ss_common import ARMS, ENDPOINTS, budget, read, record, require, sha, tree_bytes, write


def rows(path):
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), "Torn completed trace")
    return [json.loads(line) for line in raw.splitlines()]


def state_digest(state):
    raw = (
        json.dumps(
            {k: v for k, v in state.items() if k != "state_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    require(hashlib.sha256(raw).hexdigest() == state["state_sha256"], "State digest")


def verify_training(root, *, bundle=None, omitted=None):
    from types import SimpleNamespace

    import torch
    from ss_common import boundary
    from ss_worker import datasets

    bundle = bundle or root / "tools"
    train_data = datasets(bundle)[0]
    dataset_identity = train_data.identity
    sys.path.insert(0, str(bundle / "legacy"))
    import runtime

    mfp = runtime.fingerprints()
    panel = {}
    for arm, (_, passes, steps) in ARMS.items():
        data = rows(root / "jobs" / arm / "updates.jsonl")
        require(len(data) == steps, "Trace length")
        targets, windows = 0, 0
        generator = torch.Generator(device="cpu").manual_seed(160)
        order = torch.randperm(6105, generator=generator)
        cursor = 0
        order_hash = mfp.tensor_digest(order)
        rng_hash = mfp.tensor_digest(generator.get_state())
        rates = read(root / "jobs" / arm / "schedule.json")
        for i, r in enumerate(data, 1):
            if cursor == 6105:
                order = torch.randperm(6105, generator=generator)
                cursor = 0
                order_hash = mfp.tensor_digest(order)
                rng_hash = mfp.tensor_digest(generator.get_state())
            signature = mfp.input_signature(
                SimpleNamespace(
                    stream=SimpleNamespace(cursor=cursor, order=order),
                    train_data=train_data,
                    config=SimpleNamespace(batch_size=2, accumulation_steps=8),
                )
            )
            require(r["input"] == signature, "Input sequence differs from frozen cache/permutation")
            cursor += signature["windows"]
            require(r["state"]["datasets"]["train"] == dataset_identity, "Dataset state identity")
            require(
                r["state"]["sampler"]["order"] == order_hash
                and r["state"]["sampler"]["rng"] == rng_hash,
                "Sampler RNG/permutation",
            )
            m = r["metric"]
            state_digest(r["state"])
            targets += r["input"]["targets"]
            windows += r["input"]["windows"]
            require(
                m["step"] == i
                and m["tokens_seen"] == targets
                and m["windows_seen"] == windows
                and m["targets"] == r["input"]["targets"]
                and m["learning_rate"] == rates[i - 1]
                and math.isfinite(m["loss"]),
                "Update accounting",
            )
            require(m["microbatches"] == (5 if i % 382 == 0 else 8), "Tail accounting")
            epoch, cursor, expected = boundary(i)
            sampler = r["state"]["sampler"]
            require(
                m["epoch"] == epoch
                and windows == expected
                and sampler["cursor"] == cursor
                and sampler["epoch"] == epoch
                and sampler["targets"] == targets
                and sampler["step"] == i,
                "Sampler evidence",
            )
        require(targets == passes * 1943015 and windows == passes * 6105, "Final exposure")
        for step in (0, *range(382, steps + 1, 382)):
            cp = root / "jobs" / arm / f"step-{step:08d}"
            meta = read(cp / "diagnostic.json")
            for name, identity in meta["files"].items():
                path = cp / name
                if path.exists():
                    require(record(path) == identity, "Checkpoint file identity")
                else:
                    require(
                        (omitted or {}).get(path.relative_to(root).as_posix()) == identity,
                        "Absent checkpoint is not inventory-bound",
                    )
            expected = (
                data[step - 1]["state"]
                if step
                else read(root / "jobs" / arm / "initial.json")["state"]
            )
            require(meta["state"] == expected and meta["readback_exact"], "Checkpoint state")
        panel[arm] = data
    require([r["input"] for r in panel["A"]] == [r["input"] for r in panel["B"][:382]], "A/B order")
    require([r["input"] for r in panel["B"]] == [r["input"] for r in panel["C"]], "B/C order")
    require(sum(len(r) for r in panel.values()) == 1910, "Serious budget")
    return {"serious_updates": 1910, "targets": 9715075, "accepted_base": False}


def verify_execution(root, bundle):
    from ss_controller import check_fixture, fixtures

    expected = {
        n.removeprefix("source/src/latos/"): v
        for n, v in read(bundle / "required-inputs.json").items()
        if n.startswith("source/src/latos/") and n.endswith(".py")
    }
    specs = [
        {"id": "prepare", "updates_reserved": 0},
        *fixtures(),
        *[{"id": arm, "updates_reserved": values[2]} for arm, values in ARMS.items()],
        *[{"id": "eval-" + name, "updates_reserved": 0} for name in ("reference", *ENDPOINTS)],
    ]
    require(
        {p.name for p in (root / "jobs").iterdir()} == {s["id"] for s in specs}, "Job allowlist"
    )
    total = 0
    controls = []
    for spec in specs:
        job = root / "jobs" / spec["id"]
        result = read(job / "result.json")
        require(
            result["status"] == "complete"
            and result["optimizer_updates"] == spec["updates_reserved"],
            "Job result",
        )
        total += result["optimizer_updates"]
        source = job / "source/latos"
        require(
            {p.relative_to(source).as_posix(): record(p) for p in source.rglob("*.py")} == expected,
            "Executed source snapshot",
        )
        controls.append(read(job / "controls.json"))
        if spec.get("kind") == "fixture-resumed":
            check_fixture(root, spec)
    require(total == 1962 and all(v == controls[0] for v in controls), "Runtime/work budget")
    require(read(root / "jobs/prepare/guardian-test.json")["passed"] is True, "Guardian preflight")
    return {"physical_updates": total, "source_snapshots": len(specs), "runtime_matches": True}


def summarize(root):
    from latos.evaluation.compare import compare

    results = {}
    for name in ENDPOINTS:
        results[name] = compare(root / "evaluation-reference", root / f"evaluation-{name}", "base")
    contrasts = {}
    for a, b in (("A1", "B1"), ("B1", "B2"), ("A1", "B2"), ("B2", "C2")):
        diagnostic = compare(root / f"evaluation-{a}", root / f"evaluation-{b}", "base")
        contrasts[f"{a}:{b}"] = {
            "paired": diagnostic["paired"],
            "scope": "diagnostic only; not acceptance reference",
        }
    return {
        "primary_reference": "preserved Phase 5",
        "fixed_gate_comparisons": results,
        "descriptive_contrasts": contrasts,
        "accepted_base": False,
        "phase18_authorized": False,
    }


def package(root, deadline):
    require(os.environ.pop("LATOS_SUBSET_PACKAGE", "") == "1", "Only timed controller may package")
    budget(deadline)
    status = read(root / "outcome.json")
    if status["execution_complete"]:
        write(
            root / "verification.json",
            {**verify_training(root), **verify_execution(root, root / "tools")},
        )
        budget(deadline)
        write(root / "comparisons.json", summarize(root))
    paths = {}
    for name in ("launch.json",):
        p = root.parent / name
        if p.is_file():
            paths["guardian/" + name] = p
    excluded = {"events.jsonl", "package.log", "package-events.jsonl"}
    # Controller event stream is closed now and can be included; only active pack logs excluded.
    excluded.remove("events.jsonl")
    for p in root.rglob("*"):
        if p.is_file() and not p.is_relative_to(root / "tools") and p.name not in excluded:
            require(not p.is_symlink(), "Output symlink")
            paths[p.relative_to(root).as_posix()] = p
    included, omitted = {}, {}
    finals = {
        f"jobs/{arm}/step-{step:08d}/model/model.safetensors" for arm, step in ENDPOINTS.values()
    }
    for name, p in paths.items():
        budget(deadline)
        binary = p.suffix == ".safetensors" and name not in finals
        (omitted if binary else included)[name] = record(p)
    manifest = {
        "attempt": "small-subset-v1",
        "files": included,
        "omitted_on_Windows": omitted,
        "outcome": status,
        "accepted_base": False,
        "final_scored": False,
        "bundle_sha256": sha(root / "tools/bundle.json")
        if (root / "tools/bundle.json").exists()
        else None,
    }
    archive = root / "small-subset-return.zip"
    require(
        tree_bytes(root.parent) + 64 * 1024**2 + sum(i["bytes"] for i in included.values())
        < 8 * 1024**3,
        "Return artifact reserve",
    )
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as z:
        for name in included:
            budget(deadline)
            z.write(paths[name], name)
        z.writestr("return.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(archive) as z:
        for name, expected in included.items():
            budget(deadline)
            with z.open(name) as f:
                require(
                    hashlib.file_digest(f, "sha256").hexdigest() == expected["sha256"],
                    "Return readback",
                )
    budget(deadline)
    write(
        root / "return.receipt.json",
        {
            **record(archive),
            "files": len(included) + 1,
            "omitted_files": len(omitted),
            "verified": True,
            "packaged_monotonic": time.monotonic(),
        },
    )


if __name__ == "__main__":
    require(len(sys.argv) == 4 and sys.argv[1] == "package", "Internal timed packaging only")
    package(Path(sys.argv[2]), float(sys.argv[3]))
