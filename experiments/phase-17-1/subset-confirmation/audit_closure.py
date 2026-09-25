"""Independent final-confirmation metadata/tensor audit; no model execution or new study."""

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from safetensors import safe_open


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def audit(directory, inputs):
    root = directory / "run"
    bundle = directory / "reviewed-bundle"
    manifest = read(root / "study-manifest.json")
    require(
        manifest["source_commit"] == "ba262bdb598d164da322972c1208e7fed55c9335",
        "Wrong executed commit",
    )
    for name, source in manifest["source_paths"].items():
        raw = subprocess.check_output(["git", "show", manifest["source_commit"] + ":" + source])
        require((bundle / name).read_bytes() == raw, "Bundle/Git mismatch")
    auth = read(bundle / "authorization.json")
    require(
        auth["proposal_sha256"] == sha(bundle / "PROPOSAL.md")
        and auth["serious_seed"] == 161
        and auth["hard_wall_seconds"] == 1800
        and auth["last_fixed_subset_experiment"] is True,
        "Authorization/contract mismatch",
    )
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    jobs, times, active = [], {}, None
    last = -1
    for event in events:
        require(event["elapsed"] >= last, "Event clock reversal")
        last = event["elapsed"]
        if event["event"] == "job-start":
            require(active is None, "Concurrent job")
            active = (event["id"], event["elapsed"])
            jobs.append(event["id"])
        elif event["event"] == "child-exit":
            require(active is not None and event["code"] == 0, "Worker failure")
            times[active[0]] = event["elapsed"] - active[1]
            active = None
    require(active is None, "Unfinished child")
    expected = [
        "prepare",
        *[
            f"{runtime}-{fixture}-{mode}"
            for runtime in ("checkout", "wheel")
            for fixture in ("ordinary", "partial-tail")
            for mode in ("reference", "resumed")
        ],
        "D",
        "E",
        "eval-reference",
        "eval-D2",
        "eval-E2",
        "eval-E4",
    ]
    require(jobs == expected, "Prespecified sequence differs")
    milestone = {e["event"]: e["elapsed"] for e in events if e["event"].endswith("-complete")}
    terminal = read(inputs / "completed.json")
    outer = read(inputs / "guardian.json")
    receipt = read(inputs / "return.receipt.json")
    stages = {
        "prepare": milestone["prepare-complete"],
        "train": milestone["training-complete"] - milestone["prepare-complete"],
        "evaluate": milestone["evaluation-complete"] - milestone["training-complete"],
        "packaging_and_outer_closure": outer["total_wall_seconds"]
        - terminal["elapsed_before_package"],
    }
    require(
        all(
            stages[name] <= limit
            for name, limit in [
                ("prepare", 180),
                ("train", 1080),
                ("evaluate", 300),
                ("packaging_and_outer_closure", 180),
            ]
        ),
        "Stage limit exceeded",
    )
    require(
        terminal["execution_complete"]
        and terminal["failure"] is None
        and outer["exit_code"] == 0
        and not outer["deadline_terminated"]
        and outer["total_wall_seconds"] <= 1800,
        "Whole-study disposition",
    )
    require(
        terminal["final_scored"] is False
        and terminal["accepted_base"] is False
        and terminal["base_model_acceptance_run"] is False
        and terminal["phase18_authorized"] is False
        and terminal["further_fixed_subset_exposure_authorized"] is False,
        "Authorization drift",
    )
    require(
        terminal["return_receipt_sha256"] == sha(inputs / "return.receipt.json"), "Receipt binding"
    )
    require(terminal["total_elapsed_seconds"] <= outer["total_wall_seconds"], "Final clock")
    start = read(root / "guardian/launch.json")["started_monotonic"]
    require(
        start + terminal["elapsed_before_package"]
        <= receipt["packaged_monotonic"]
        <= start + terminal["total_elapsed_seconds"],
        "Packaging escaped study clock",
    )
    spec = importlib.util.spec_from_file_location(
        "historical_schedule_audit", "experiments/recovery-diagnostic/review_return.py"
    )
    schedule_audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schedule_audit)
    from latos.training.config import TrainingConfig

    arms, initial, rate_differences, max_ulps = {}, {}, 0, 0
    for arm, passes, steps in [("D", 2, 764), ("E", 4, 1528)]:
        job = root / "jobs" / arm
        begin = read(job / "initial.json")
        initial[arm] = begin["state"]
        config = TrainingConfig(**begin["binding"]["training_config"])
        require(
            config.seed == 161 and config.max_steps == steps and config.warmup_steps == 19,
            "Frozen seed/horizon",
        )
        rates = read(job / "schedule.json")
        rows = [json.loads(line) for line in (job / "updates.jsonl").read_text().splitlines()]
        require(len(rows) == steps and len(rates) == steps, "Missing updates/rates")
        for i, row in enumerate(rows, 1):
            require(
                row["metric"]["learning_rate"] == rates[i - 1], "Same-Windows schedule mismatch"
            )
            require(
                rates[i - 1] in schedule_audit.schedule_candidates(i, config.to_dict()),
                "Cross-host formula mismatch",
            )
            analytic = config.learning_rate_at(i)
            difference = abs(analytic - rates[i - 1])
            rate_differences += difference != 0
            max_ulps = max(max_ulps, difference / max(math.ulp(analytic), math.ulp(rates[i - 1])))
            require(
                math.isfinite(row["metric"]["loss"])
                and math.isfinite(row["metric"]["grad_norm_before_clip"]),
                "Nonfinite metric",
            )
        monitors = {}
        for step in range(0, steps + 1, 382):
            monitor = read(job / f"monitor-{step}.json")
            require(
                monitor["targets"] == 57036
                and monitor["windows"] == 184
                and monitor["split"] == "validation",
                "Changed monitor",
            )
            monitors[str(step)] = monitor["loss"]
        epoch_stats = []
        for epoch in range(passes):
            chunk = rows[epoch * 382 : (epoch + 1) * 382]
            targets = sum(r["metric"]["targets"] for r in chunk)
            require(targets == 1943015, "Pass targets")
            epoch_stats.append(
                {
                    "pass": epoch + 1,
                    "targets": targets,
                    "online_train_nll": sum(
                        r["metric"]["loss"] * r["metric"]["targets"] for r in chunk
                    )
                    / targets,
                    "fraction_clipped": sum(r["metric"]["grad_norm_before_clip"] > 1 for r in chunk)
                    / len(chunk),
                }
            )
        peaks = {k: max(r["resources"][k] for r in rows) for k in rows[0]["resources"]}
        require(
            peaks["host_peak_working_set_bytes"] <= 8 * 1024**3
            and peaks["peak_reserved_bytes"]
            <= 0.85 * begin["binding"]["controls"]["gpu_total_bytes"],
            "Training resources",
        )
        initial_meta = read(job / "step-00000000/model/metadata.json")
        require(
            initial_meta["weights_sha256"]
            != "0aa3c90ddabfdff0f7d785435cf978781ac58ed453c1922bcab8b211e175b174",
            "Reused seed-160 initializer",
        )
        arms[arm] = {
            "passes": passes,
            "updates": steps,
            "targets": passes * 1943015,
            "monitor_loss": monitors,
            "epoch_metrics": epoch_stats,
            "job_seconds": times[arm],
            "resource_peaks": peaks,
            "initial_model_sha256_inventory": initial_meta["weights_sha256"],
        }
    require(
        initial["D"]["model"] == initial["E"]["model"]
        and initial["D"]["sampler"] == initial["E"]["sampler"],
        "Initial pair differs",
    )
    comparisons = read(root / "comparisons.json")
    endpoints = {}
    for name, arm, step in [
        ("reference", None, None),
        ("D2", "D", 764),
        ("E2", "E", 764),
        ("E4", "E", 1528),
    ]:
        folder = root / f"evaluation-{name}"
        summary = read(folder / "summary.json")
        m = read(folder / "manifest.json")
        require(
            m["final_access"] is None
            and m["data"]["split"] == "validation"
            and m["suite_sha256"] == m["protocol"]["suite_sha256"],
            "Final access",
        )
        result = {
            "bpb": summary["matched"]["bits_per_byte"],
            "book_bpb": {n: v["bits_per_byte"] for n, v in summary["matched"]["documents"].items()},
            "repetition": summary["generation"]["repeated_trigram_fraction"],
            "empty": summary["generation"]["empty"],
            "instructions_correct": summary["instructions"]["overall"]["correct"],
            "arc": {
                n: {
                    "correct": v["raw"]["correct"],
                    "total": v["raw"]["total"],
                    "accuracy": v["raw"]["accuracy"],
                    "byte_correct": v["byte_normalized"]["correct"],
                }
                for n, v in summary["external"].items()
            },
            "weights_sha256": m["model"]["weights_sha256"],
        }
        if name != "reference":
            cp = root / f"jobs/{arm}/step-{step:08d}"
            state = read(cp / "diagnostic.json")["state"]
            weights = cp / "model/model.safetensors"
            require(sha(weights) == result["weights_sha256"], "Scored model identity")
            hashes, elements = {}, 0
            with safe_open(weights, framework="numpy") as f:
                for key in f.keys():
                    array = f.get_tensor(key)
                    require(
                        array.dtype == np.float32 and np.isfinite(array).all(),
                        "Nonfinite/wrong-dtype tensor",
                    )
                    layout = {
                        "shape": list(array.shape),
                        "stride": [x // array.itemsize for x in array.strides],
                        "dtype": "torch.float32",
                    }
                    h = hashlib.sha256(canonical(layout))
                    h.update(array.tobytes(order="C"))
                    hashes[key] = h.hexdigest()
                    elements += array.size
            require(
                hashes == state["model"] and elements == 21238272, "Checkpoint tensor fingerprints"
            )
            gates = comparisons["fixed_gate_comparisons"][name]["checks"]
            result.update(
                failed_gates=[k for k, v in gates.items() if not v["pass"]],
                gate_values=gates,
                tensor_elements=elements,
                tensors_verified=len(hashes),
                source_step=step,
            )
        endpoints[name] = result
    contrasts = {}
    for before, after in [("D2", "E2"), ("E2", "E4"), ("D2", "E4")]:
        a, b = endpoints[before], endpoints[after]
        ga, gb = [read(root / f"evaluation-{n}/generation.json") for n in (before, after)]
        require([x["id"] for x in ga] == [x["id"] for x in gb], "Prompt pairing")
        delta = [
            y["observations"]["repeated_trigram_fraction"]
            - x["observations"]["repeated_trigram_fraction"]
            for x, y in zip(ga, gb, strict=True)
        ]
        contrasts[before + ":" + after] = {
            "bpb_change_percent": 100 * (b["bpb"] / a["bpb"] - 1),
            "book_bpb_change_percent": {
                k: 100 * (b["book_bpb"][k] / v - 1) for k, v in a["book_bpb"].items()
            },
            "repetition_change_pp": 100 * (b["repetition"] - a["repetition"]),
            "repetition_prompts_better_tied_worse": [
                sum(x < 0 for x in delta),
                sum(x == 0 for x in delta),
                sum(x > 0 for x in delta),
            ],
            "paired": comparisons["descriptive_contrasts"][before + ":" + after]["paired"],
        }
    prior = read(Path("experiments/phase-17-1/small-subset/results.json"))
    old = prior["endpoints"]["C2"]
    new = endpoints["D2"]
    cross_seed = {
        "scope": "Historical two-pass C2 seed160 vs D2 seed161; no extra model scoring",
        "bpb_change_percent": 100 * (new["bpb"] / old["bpb"] - 1),
        "repetition_change_pp": 100 * (new["repetition"] - old["repetition"]),
        "arc_easy_correct_old_new": [
            old["arc"]["ARC-Easy"]["correct"],
            new["arc"]["ARC-Easy"]["correct"],
        ],
    }
    require(
        comparisons["terminal_endpoint"] == "E4"
        and comparisons["fixed_subset_direction"] == "stop/rethink-data-training"
        and comparisons["further_fixed_subset_exposure_authorized"] is False
        and comparisons["accepted_base"] is False
        and comparisons["base_model_acceptance_run"] is False
        and comparisons["phase18_authorized"] is False,
        "Terminal rule not honored",
    )
    require(len(endpoints["E4"]["failed_gates"]) > 0, "Stopping condition not established")
    previous_ref = Path(
        "outputs/small-subset-corrected-reviewed/return/original/run/evaluation-reference"
    )
    current_ref = read(root / "evaluation-reference/summary.json")
    old_ref = read(previous_ref / "summary.json")
    require(
        {k: v for k, v in current_ref.items() if k != "resources"}
        == {k: v for k, v in old_ref.items() if k != "resources"},
        "Reference quality changed",
    )
    included_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    with __import__("zipfile").ZipFile(inputs / "subset-confirmation-return.zip") as z:
        return_manifest = json.loads(z.read("return.json"))
    omitted = return_manifest["omitted_on_Windows"]
    artifact_upper = (
        included_bytes
        + sum(v["bytes"] for v in omitted.values())
        + sum(p.stat().st_size for p in (directory / "reviewed-bundle").rglob("*") if p.is_file())
        + (inputs / "subset-confirmation-return.zip").stat().st_size
        + 128 * 1024**2
    )
    require(artifact_upper < 8 * 1024**3, "Artifact cap")
    return {
        "source_commit": manifest["source_commit"],
        "scope": "Saved-evidence/tensor audit; no model loading, forward or optimization",
        "total_wall_seconds": outer["total_wall_seconds"],
        "controller_seconds": terminal["total_elapsed_seconds"],
        "stage_seconds": stages,
        "job_seconds": times,
        "artifact_upper_bytes_with_reserve": artifact_upper,
        "omitted_inventory_records_not_rehashed": len(omitted),
        "arms": arms,
        "endpoints": endpoints,
        "contrasts": contrasts,
        "historical_seed_comparison": cross_seed,
        "cross_host_adjacent_cosine_cases": rate_differences,
        "cross_host_max_output_ULPs": max_ulps,
        "same_Windows_rates": "2292 exact matches to recorded schedules",
        "new_optimizer_updates": 0,
        "new_model_scoring": False,
        "accepted_base": False,
        "final_scoring": False,
        "fixed_subset_exposure_direction": (
            "CLOSED: E4 failed fixed gates; no more passes/seeds/exposure-only runs"
        ),
        "phase18_authorized": False,
        "publication_authorized": False,
        "new_experiment_implemented": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.directory, args.inputs)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {
                "verified": True,
                "E4_failed_gates": len(result["endpoints"]["E4"]["failed_gates"]),
                "direction": result["fixed_subset_exposure_direction"],
                "new_optimizer_updates": 0,
            }
        )
    )
