"""Independent, read-only review of the pinned Windows return; no model execution.

The original Mac verifier is retained unchanged, including its failed one-output-
ULP cross-host schedule check. Same-Windows schedule equality remains exact here.
Adjacent cosine results are supplementary cross-host arithmetic diagnostics only.
"""

import argparse
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

SOURCE = "0e5b55f516b6dd53a48f53928df2616872ca5f82"
ARCHIVE = "340b57d7f201436e47b575b7a792cb208cb74dfe47cd5f835322d313304c47d9"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def state_check(state):
    core = {k: v for k, v in state.items() if k != "state_sha256"}
    require(digest(canonical(core)) == state["state_sha256"], "State digest mismatch")


def schedule_candidates(step, config):
    """Propagate adjacent libm cosine results through the unchanged float expression.

    This is NOT a loss tolerance or a same-host schedule allowance. Every Windows
    execution must use exactly the same rate at a given step, independently below.
    """
    peak, warm = config["learning_rate"], config["warmup_steps"]
    if step <= warm:
        return {peak * step / warm}
    start = max(1, warm)
    progress = (step - start) / max(1, config["max_steps"] - start)
    cosine = math.cos(math.pi * progress)
    minimum = config["min_lr_ratio"]
    return {
        peak * (minimum + (1 - minimum) * (1 + c) / 2)
        for c in (math.nextafter(cosine, -math.inf), cosine, math.nextafter(cosine, math.inf))
    }


def exact_schedule(observed, reference):
    require(observed == reference, "Same-Windows schedule mismatch")


def compare(reference, replay):
    require(len(reference) == 997 and len(replay) == 485, "Incomplete pair")
    state_steps, excesses, deltas = [], [], []
    for a, b in zip(reference[512:], replay, strict=True):
        require(a["input"] == b["input"], "Replay input mismatch")
        for key in ("step", "targets", "tokens_seen", "windows_seen", "epoch", "microbatches"):
            require(a["metric"][key] == b["metric"][key], "Replay counter mismatch")
        exact_schedule(a["metric"]["learning_rate"], b["metric"]["learning_rate"])
        delta = abs(a["metric"]["loss"] - b["metric"]["loss"])
        require(math.isfinite(delta), "Nonfinite loss difference")
        deltas.append(delta)
        if delta > 1e-5:
            excesses.append({"step": b["metric"]["step"], "delta": delta})
        if a["state"] != b["state"]:
            state_steps.append(b["metric"]["step"])
    return {
        "replay_updates": 485,
        "first_state_difference": state_steps[0] if state_steps else None,
        "state_difference_count": len(state_steps),
        "first_loss_excess": excesses[0] if excesses else None,
        "loss_excess_count": len(excesses),
        "maximum_loss_difference": max(deltas),
        "maximum_loss_difference_step": 513 + deltas.index(max(deltas)) if max(deltas) else None,
        "exact_loss_matches": deltas.count(0),
        "exact_states": not state_steps,
        "loss_gate_passed": not excesses,
    }


def decision(pairs):
    return (
        not pairs["legacy"]["loss_gate_passed"]
        and pairs["deterministic"]["loss_gate_passed"]
        and pairs["deterministic"]["exact_states"]
    )


def review(archive, receipt, root):
    raw = archive.read_bytes()
    require(digest(raw) == ARCHIVE, "Owner-supplied archive checksum mismatch")
    receipt = read(receipt)
    require(receipt["sha256"] == ARCHIVE and receipt["bytes"] == len(raw), "Receipt mismatch")
    record = read(root / "return.json")
    require(record["diagnostic_source_commit"] == SOURCE, "Wrong diagnostic commit")
    with zipfile.ZipFile(archive) as z:
        require(z.read("return.json") == (root / "return.json").read_bytes(), "Manifest mismatch")
        names = z.namelist()
        require(len(names) == len(set(names)), "Duplicate archive path")
        require(set(names) == set(record["files"]) | {"return.json"}, "Archive file set mismatch")
        for name, expected in record["files"].items():
            require(
                not Path(name).is_absolute() and not any(x in name for x in ("..", "\\", ":")),
                "Unsafe path",
            )
            data = z.read(name)
            require(data == (root / name).read_bytes(), "Extracted evidence changed")
            require(
                len(data) == expected["bytes"] and digest(data) == expected["sha256"],
                "Payload hash mismatch",
            )
    all_records = record["files"] | record["omitted_retained_on_Windows"]
    require(
        not set(record["files"]) & set(record["omitted_retained_on_Windows"]), "Inventory overlap"
    )
    require(receipt["included"] == len(record["files"]) == 318, "Included count mismatch")
    require(
        receipt["omitted_retained"] == len(record["omitted_retained_on_Windows"]) == 14,
        "Omission count mismatch",
    )
    bundle = read(root / "bundle/bundle.json")
    require(bundle["source_commit"] == SOURCE, "Bundle commit mismatch")
    for name, expected in bundle["files"].items():
        data = (root / "bundle" / name).read_bytes()
        original = subprocess.check_output(["git", "show", f"{SOURCE}:{expected['source_path']}"])
        require(data == original and digest(data) == expected["sha256"], "Measured source changed")
    plan = read(root / "bundle/plan.json")
    model_config = read(root / "bundle/selected-model.json")
    training_config = {
        "schema_version": 1,
        "sequence_length": 512,
        "max_steps": 997,
        "seed": 160,
        **{
            k: plan[k]
            for k in (
                "batch_size",
                "accumulation_steps",
                "warmup_steps",
                "learning_rate",
                "min_lr_ratio",
                "weight_decay",
                "beta1",
                "beta2",
                "eps",
                "max_grad_norm",
            )
        },
    }
    dataset = read(root / "run/data/dataset.json")
    require(
        dataset == read(root / "run/synthetic-data.json")
        and dataset["files"] == plan["expected_synthetic_files"],
        "Synthetic identity differs",
    )
    for name, sha in dataset["files"].items():
        require(all_records["run/data/" + name]["sha256"] == sha, "Synthetic omission differs")
    require(plan["loss_tolerance"] == 1e-5 and plan["max_physical_updates"] == 2964, "Gate changed")
    preflight = read(root / "bundle/windows-preflight.json")
    cases = list(ET.parse(root / "bundle/windows-preflight.xml").getroot().iter("testcase"))
    require(
        len(cases) == 10
        and not any(c.find(k) is not None for c in cases for k in ("failure", "error", "skipped")),
        "CPU checks failed",
    )
    require(preflight["passed"] and preflight["optimizer_updates"] == 0, "Preflight scope changed")
    require(
        preflight["junit_sha256"] == digest((root / "bundle/windows-preflight.xml").read_bytes())
        and preflight["bundle_sha256"] == digest((root / "bundle/bundle.json").read_bytes()),
        "Preflight binding mismatch",
    )
    shapes = read(root / "bundle/lengths.json")["lengths"]
    sums = [0]
    for length in shapes:
        sums.append(sums[-1] + length - 1)
    require(len(shapes) == 15940 and sums[-1] == 5021949, "Shape exposure changed")
    paths = (
        subprocess.check_output(["git", "ls-tree", "-r", "--name-only", SOURCE, "src/latos"])
        .decode()
        .splitlines()
    )
    package = {
        p.removeprefix("src/latos/"): subprocess.check_output(["git", "show", f"{SOURCE}:{p}"])
        for p in paths
        if p.endswith(".py")
    }
    controls, starts, traces, peaks = (
        {},
        {},
        {},
        {"host_peak_working_set_bytes": 0, "peak_reserved_bytes": 0},
    )
    reference_rates, cross_host = {}, []
    for profile in ("legacy", "deterministic"):
        for mode, count in (("reference", 997), ("resumed", 485)):
            stage = root / "run" / profile / mode
            prefix = stage.relative_to(root).as_posix() + "/"
            inv = read(stage / "inventory.json")
            require(
                inv
                == {
                    n.removeprefix(prefix): v
                    for n, v in all_records.items()
                    if n.startswith(prefix) and n != prefix + "inventory.json"
                },
                "Stage inventory mismatch",
            )
            snapshot = stage / "source/latos"
            require(
                {p.relative_to(snapshot).as_posix(): p.read_bytes() for p in snapshot.rglob("*.py")}
                == package,
                "Executed package mismatch",
            )
            start, result = read(stage / "start.json"), read(stage / "result.json")
            state_check(start["state"])
            require(
                start["state"]["model_config"] == model_config
                and start["state"]["training_config"] == training_config,
                "Model/training contract mismatch",
            )
            for identity in start["state"]["datasets"].values():
                require(
                    identity["payloads"] == dataset["files"]
                    and identity["targets"] == 5021949
                    and identity["windows"] == 15940,
                    "Dataset state identity mismatch",
                )
            if mode == "reference":
                require(
                    not start["state"]["optimizer"]
                    and start["state"]["sampler"]["step"] == 0
                    and start["initialization"]
                    == {
                        "model_seed": 17101,
                        "shuffle_seed": 160,
                        "synthetic_seed": 17102,
                        "source": "fresh random",
                    },
                    "Initialization mismatch",
                )
            starts[profile, mode] = start
            controls[profile, mode] = start["controls"]
            expected_controls = plan["profile_settings"][profile]
            require(
                all(start["controls"][k] == v for k, v in expected_controls.items()),
                "Policy differs",
            )
            require(
                start["source_commit"] == SOURCE
                and start["bundle_sha256"] == preflight["bundle_sha256"],
                "Worker source binding differs",
            )
            env = start["environment"]
            require(
                env["gpu"] == "NVIDIA GeForce RTX 4070 SUPER"
                and env["native_bfloat16"]
                and "616.92" in env["driver_report"]
                and env["capability"] == [8, 9],
                "Hardware differs",
            )
            runtime = env["runtime"]
            require(
                runtime["python"] == plan["python"]
                and runtime["torch"] == plan["torch"]
                and runtime["implementation_sha256"] == plan["legacy_implementation_sha256"],
                "Runtime differs",
            )
            require(
                result["status"] == "complete"
                and result["executed_updates"] == count
                and result["final_step"] == 997
                and result["targets_seen"] == sums[-1],
                "Job incomplete",
            )
            rows = lines(stage / "trace.jsonl")
            require(len(rows) == count, "Trace length mismatch")
            traces[profile, mode] = rows
            for index, row in enumerate(rows, 1 if mode == "reference" else 513):
                metric, state = row["metric"], row["state"]
                state_check(state)
                begin, end = (index - 1) * 16, min(index * 16, len(shapes))
                require(
                    metric["step"] == index
                    and metric["targets"] == sums[end] - sums[begin]
                    and metric["tokens_seen"] == sums[end]
                    and metric["windows_seen"] == end
                    and metric["epoch"] == 0
                    and metric["microbatches"] == math.ceil((end - begin) / 2),
                    "Exposure mismatch",
                )
                require(
                    row["input"]["lengths"] == shapes[begin:end]
                    and row["input"]["targets"] == metric["targets"]
                    and row["input"]["windows"] == end - begin,
                    "Input layout mismatch",
                )
                require(
                    all(
                        math.isfinite(metric[k])
                        for k in ("loss", "learning_rate", "grad_norm_before_clip")
                    ),
                    "Nonfinite metric",
                )
                for key in (
                    "global_rng",
                    "model_config",
                    "training_config",
                    "datasets",
                    "optimizer_parameter_names",
                ):
                    require(state[key] == start["state"][key], "State configuration/RNG changed")
                sampler = state["sampler"]
                require(
                    sampler
                    == {
                        **start["state"]["sampler"],
                        "step": index,
                        "targets": sums[end],
                        "windows": end,
                        "cursor": end,
                        "epoch": 0,
                    },
                    "Sampler state mismatch",
                )
                rate = metric["learning_rate"]
                if profile == "legacy" and mode == "reference":
                    reference_rates[index] = rate
                    config = state["training_config"]
                    require(
                        rate in schedule_candidates(index, config),
                        "Unexplained cross-host schedule",
                    )
                    if index > 375:
                        local = 0.0003 * (
                            0.1 + 0.9 * (1 + math.cos(math.pi * ((index - 375) / 622))) / 2
                        )
                        if rate != local:
                            cross_host.append(
                                {
                                    "step": index,
                                    "windows": rate,
                                    "mac": local,
                                    "output_ulps": abs(rate - local) / math.ulp(local),
                                }
                            )
                else:
                    exact_schedule(rate, reference_rates[index])
            records = [r["resources"] for r in rows] + [result["resources"]]
            for name in ("final.json", "boundary.json"):
                if not (stage / name).exists():
                    require(
                        name == "boundary.json" and mode == "resumed", "Missing checkpoint evidence"
                    )
                    continue
                saved = read(stage / name)
                for key in ("before", "after", "loaded"):
                    state_check(saved[key])
                require(
                    saved["roundtrip_exact"]
                    and saved["before"] == saved["after"] == saved["loaded"],
                    "Checkpoint roundtrip mismatch",
                )
                require(saved["controls"] == start["controls"], "Checkpoint policy mismatch")
                require(
                    saved["before"] == rows[511 if name == "boundary.json" else -1]["state"],
                    "Checkpoint not bound to trace",
                )
                directory = stage / ("split" if name == "boundary.json" else "final-checkpoint")
                metadata = read(directory / "state.json")
                model_metadata = read(directory / "model/metadata.json")
                prefix_cp = directory.relative_to(root).as_posix() + "/"
                require(
                    metadata["runtime"] == runtime
                    and metadata["config"] == training_config
                    and metadata["precision"] == "bfloat16"
                    and metadata["kind"] == "latos-training-single-pass-v3",
                    "Checkpoint runtime/configuration mismatch",
                )
                require(
                    metadata["training_sha256"]
                    == all_records[prefix_cp + "training.safetensors"]["sha256"]
                    and metadata["training_bytes"]
                    == all_records[prefix_cp + "training.safetensors"]["bytes"]
                    and metadata["model_metadata_sha256"]
                    == digest((directory / "model/metadata.json").read_bytes()),
                    "Checkpoint training payload binding differs",
                )
                require(
                    model_metadata["weights_sha256"]
                    == all_records[prefix_cp + "model/model.safetensors"]["sha256"]
                    and model_metadata["weights_bytes"]
                    == all_records[prefix_cp + "model/model.safetensors"]["bytes"]
                    and model_metadata["parameter_count"] == 34087424
                    and read(directory / "model/config.json") == model_config,
                    "Checkpoint model payload binding differs",
                )
                require(
                    metadata["step"] == saved["before"]["sampler"]["step"]
                    and metadata["tokens_seen"] == saved["before"]["sampler"]["targets"]
                    and metadata["windows_seen"] == saved["before"]["sampler"]["windows"]
                    and digest(canonical(metadata["optimizer_groups"]))
                    == saved["before"]["optimizer_groups"],
                    "Checkpoint counter/optimizer group binding differs",
                )
                records.append(saved["resources"])
            for resource in records:
                for key in peaks:
                    require(
                        isinstance(resource[key], int) and resource[key] >= 0,
                        "Missing resource measure",
                    )
                    peaks[key] = max(peaks[key], resource[key])
                require(
                    resource["host_peak_working_set_bytes"] <= plan["max_host_bytes"]
                    and resource["peak_reserved_bytes"] <= env["total_bytes"] * 0.85,
                    "Memory cap exceeded",
                )
    require(
        starts["legacy", "reference"]["state"] == starts["deterministic", "reference"]["state"],
        "Fresh initial states differ",
    )
    pairs = {}
    for profile in ("legacy", "deterministic"):
        require(
            controls[profile, "reference"] == controls[profile, "resumed"],
            "Within-pair policy differs",
        )
        split = read(root / "run" / profile / "reference/boundary.json")
        require(
            split["before"] == starts[profile, "resumed"]["state"],
            "New-process restoration differs",
        )
        pairs[profile] = compare(traces[profile, "reference"], traces[profile, "resumed"])
    baseline = {
        k: v
        for k, v in controls["legacy", "reference"].items()
        if k not in plan["profile_settings"]["legacy"]
    }
    corrected = {
        k: v
        for k, v in controls["deterministic", "reference"].items()
        if k not in plan["profile_settings"]["deterministic"]
    }
    require(baseline == corrected, "Undeclared between-profile policy difference")
    require(
        baseline
        == {
            "SDPA_enabled": dict.fromkeys(("cudnn", "flash", "math", "mem_efficient"), True),
            "TF32": False,
            "bf16_reduced_precision_reduction": True,
            "cpu_threads": 4,
            "cuda_build": "13.0",
            "cudnn_benchmark": False,
            "cudnn_version": 92400,
            "matmul_precision": "highest",
        },
        "Unexpected common controls",
    )
    events = lines(root / "supervisor/recovery-mechanics-results.events.jsonl")
    summary = read(root / "run/summary.json")
    require(
        events[0]["event"] == "start"
        and events[0]["plan"] == plan
        and events[-1]["event"] == "complete"
        and events[-1]["summary"] == summary,
        "Supervisor incomplete",
    )
    jobs = [
        ("prepare", "legacy", 0),
        ("reference", "legacy", 997),
        ("resumed", "legacy", 485),
        ("reference", "deterministic", 997),
        ("resumed", "deterministic", 485),
    ]
    for event in ("job-start", "job-complete"):
        require(
            [
                (e["mode"], e["profile"], e["reserved_updates"])
                for e in events
                if e["event"] == event
            ]
            == jobs,
            "Job sequence/retry mismatch",
        )
    require(
        [(j["mode"], j["profile"], j["reserved_updates"]) for j in summary["jobs"]] == jobs
        and all(j["status"] == "complete" for j in summary["jobs"]),
        "Summary job mismatch",
    )
    require(not list(root.rglob("*failure.json")), "Failure evidence present")
    times = [e["seconds"] for e in events if "seconds" in e]
    require(
        times == sorted(times)
        and 0 < summary["active_seconds"] < 3600
        and max(times) <= summary["active_seconds"],
        "Time bound/accounting mismatch",
    )
    peak_charge = max(e.get("artifact_bytes", 0) for e in events)
    retained_bytes = sum(v["bytes"] for v in all_records.values())
    require(
        peak_charge < 6 * 1024**3 and retained_bytes + len(raw) < 6 * 1024**3,
        "Artifact cap exceeded",
    )
    require(summary["physical_updates"] == 2964, "Update total mismatch")
    for profile, pair in pairs.items():
        reported = summary["pairs"][profile]
        require(
            reported["loss_gate_passed"] == pair["loss_gate_passed"]
            and reported["replay_states_identical"] == pair["exact_states"]
            and len(reported["loss_excesses"]) == pair["loss_excess_count"],
            "Reported comparison differs",
        )
    accepted = decision(pairs)
    require(summary["prospective_correction_demonstrated"] == accepted, "Reported decision differs")
    return {
        "status": "reviewed",
        "archive_sha256": ARCHIVE,
        "diagnostic_source_commit": SOURCE,
        "included_files_verified": 318,
        "omitted_tensor_files_inventory_bound_only": 14,
        "source_snapshots_verified": 4,
        "python_files_per_snapshot": len(package),
        "cpu_checks": {"passed": 10, "skipped": 0, "failed": 0, "optimizer_updates": 0},
        "physical_updates": 2964,
        "physical_targets": sum(r["metric"]["targets"] for rs in traces.values() for r in rs),
        "retries_in_supervisor": 0,
        "pairs": pairs,
        "supervised_seconds": summary["active_seconds"],
        "peak_resources": peaks,
        "peak_supervisor_charged_artifact_bytes": peak_charge,
        "inventory_retained_bytes": retained_bytes,
        "return_archive_bytes": len(raw),
        "cross_host_schedule": {
            "same_Windows_rates_exact_all_jobs": True,
            "differences": cross_host,
            "all_explained_by_adjacent_cosine_results": True,
            "original_Mac_verifier": "failed its one-output-ULP check; retained unchanged",
            "loss_and_same_host_state_gates_unchanged": True,
        },
        "prospective_correction_established": accepted,
        "limitations": "No absent Windows tensors rehashed or CUDA replay performed on Mac. "
        "Combined policy tested on synthetic inputs; historical kernel cause not identified.",
        "phase17_permanently_failed": True,
        "phase17_1_training_authorized": False,
        "phase18_authorized": False,
        "publication_authorized": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.archive, args.receipt, args.root)
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))
