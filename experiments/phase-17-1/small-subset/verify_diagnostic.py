"""Additional saved-evidence audit and scientific arithmetic; never run a model."""

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


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def audit(directory):
    root = directory / "return/original/run"
    original = root.parent
    correction = directory / "return/correction"
    endpoint_steps = {"A1": ("A", 382), "B1": ("B", 382), "B2": ("B", 764), "C2": ("C", 764)}
    original_manifest = read(root / "study-manifest.json")
    for name, path in original_manifest["source_paths"].items():
        saved = root / "tools" / name
        expected = subprocess.check_output(
            ["git", "show", original_manifest["source_commit"] + ":" + path]
        )
        require(saved.read_bytes() == expected, "Executed bundle differs from reviewed Git")
    fix = read(correction / "authorization.json")
    require(digest(correction / "correct.py") == fix["corrector_sha256"], "Correction source")
    require(
        (correction / "correct.py").read_bytes()
        == subprocess.check_output(
            [
                "git",
                "show",
                "0eca1d836f24b2b80ad93730db0719367598bc68:experiments/phase-17-1/small-subset/packaging-correction/correct.py",
            ]
        ),
        "Correction Git binding",
    )
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    times, active = {}, None
    last = -1
    for e in events:
        require(e["elapsed"] >= last, "Nonmonotonic event history")
        last = e["elapsed"]
        if e["event"] == "job-start":
            require(active is None, "Concurrent jobs")
            active = (e["id"], e["elapsed"])
        elif e["event"] == "child-exit":
            require(active is not None and e["code"] == 0, "Original job failure")
            times[active[0]] = e["elapsed"] - active[1]
            active = None
    require(active is None, "Unfinished job")
    milestones = {e["event"]: e["elapsed"] for e in events if e["event"].endswith("-complete")}
    phase_times = {
        "prepare": milestones["prepare-complete"],
        "train": milestones["training-complete"] - milestones["prepare-complete"],
        "evaluate": milestones["evaluation-complete"] - milestones["training-complete"],
    }
    require(
        all(
            phase_times[k] <= cap
            for k, cap in (("prepare", 180), ("train", 1440), ("evaluate", 480))
        ),
        "Original stage budget",
    )
    initial, traces, arms, analytic_ulps, maximum_ulps = {}, {}, {}, 0, 0
    spec = importlib.util.spec_from_file_location(
        "prior_schedule_audit", "experiments/recovery-diagnostic/review_return.py"
    )
    prior_schedule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior_schedule)
    baseline_config = read(Path("configs/training2/selected-model.json"))
    from latos.training.config import TrainingConfig

    for arm, passes, steps in (("A", 1, 382), ("B", 2, 764), ("C", 2, 764)):
        job = root / "jobs" / arm
        start = read(job / "initial.json")
        cfg = {**baseline_config, "n_layers": 4 if arm == "C" else 8}
        training = TrainingConfig(
            sequence_length=512,
            batch_size=2,
            accumulation_steps=8,
            max_steps=steps,
            warmup_steps=19,
            learning_rate=0.0003,
            min_lr_ratio=0.1,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            max_grad_norm=1.0,
            seed=160,
        )
        require(
            start["binding"]["model_config"] == cfg
            and start["binding"]["training_config"] == training.to_dict(),
            "Frozen configuration",
        )
        s = start["state"]
        require(
            s["optimizer"] == {} and s["sampler"]["step"] == 0 and s["sampler"]["targets"] == 0,
            "Fresh state",
        )
        initial[arm] = s
        trace = [json.loads(line) for line in (job / "updates.jsonl").read_text().splitlines()]
        rates = read(job / "schedule.json")
        require(len(rates) == len(trace) == steps, "Schedule length")
        for i, row in enumerate(trace, 1):
            st, metric = row["state"], row["metric"]
            require(
                st["model_config"] == cfg
                and st["training_config"] == training.to_dict()
                and st["datasets"] == s["datasets"]
                and st["global_rng"] == s["global_rng"],
                "State policy drift",
            )
            require(metric["learning_rate"] == rates[i - 1], "Same-Windows schedule mismatch")
            analytic = training.learning_rate_at(i)
            # Use the already established analytic libm-compatibility method.
            # This does not relax exact Windows rate/state/input/loss requirements.
            delta = abs(analytic - rates[i - 1])
            require(
                rates[i - 1] in prior_schedule.schedule_candidates(i, training.to_dict()),
                "Schedule formula incompatible with established cross-host analysis",
            )
            analytic_ulps += delta != 0
            maximum_ulps = max(
                maximum_ulps, delta / max(math.ulp(analytic), math.ulp(rates[i - 1]))
            )
            require(
                math.isfinite(metric["loss"]) and math.isfinite(metric["grad_norm_before_clip"]),
                "Nonfinite metric",
            )
        traces[arm] = trace
        monitors = {
            str(step): read(job / f"monitor-{step}.json")["loss"]
            for step in range(0, steps + 1, 382)
        }
        for step in range(0, steps + 1, 382):
            m = read(job / f"monitor-{step}.json")
            require(
                m["targets"] == 57036 and m["windows"] == 184 and m["split"] == "validation",
                "Monitor changed",
            )
        peaks = {key: max(row["resources"][key] for row in trace) for key in trace[0]["resources"]}
        gpu_total = start["binding"]["controls"]["gpu_total_bytes"]
        require(
            peaks["host_peak_working_set_bytes"] <= 8 * 1024**3
            and peaks["peak_reserved_bytes"] <= 0.85 * gpu_total,
            "Resource gate",
        )
        arms[arm] = {
            "passes": passes,
            "updates": steps,
            "targets": passes * 1943015,
            "parameters": 21238272 if arm == "C" else 34087424,
            "job_seconds": times[arm],
            "monitor_loss": monitors,
            "update_seconds": sum(r["metric"]["seconds"] for r in trace),
            "resource_peaks": peaks,
        }
    require(
        read(root / "jobs/B/schedule.json") == read(root / "jobs/C/schedule.json"),
        "Same-Windows B/C schedules differ",
    )
    require(
        initial["A"]["model"] == initial["B"]["model"]
        and initial["A"]["sampler"] == initial["B"]["sampler"],
        "A/B matched initialization",
    )
    for a, b in zip(traces["A"][:19], traces["B"][:19], strict=True):
        require(
            all(
                a["state"][key] == b["state"][key]
                for key in ("model", "optimizer", "optimizer_groups", "sampler", "global_rng")
            ),
            "A/B common warmup diverges",
        )
    endpoints = {}
    comparisons = read(correction / "comparisons.json")
    for name in ("reference", *endpoint_steps):
        folder = root / f"evaluation-{name}"
        summary, manifest = read(folder / "summary.json"), read(folder / "manifest.json")
        require(
            manifest["final_access"] is None and manifest["data"]["split"] == "validation",
            "Final access",
        )
        stat = {
            "bpb": summary["matched"]["bits_per_byte"],
            "book_bpb": {k: v["bits_per_byte"] for k, v in summary["matched"]["documents"].items()},
            "repetition": summary["generation"]["repeated_trigram_fraction"],
            "empty": summary["generation"]["empty"],
            "arc": {
                k: {
                    "correct": v["raw"]["correct"],
                    "total": v["raw"]["total"],
                    "accuracy": v["raw"]["accuracy"],
                    "byte_correct": v["byte_normalized"]["correct"],
                }
                for k, v in summary["external"].items()
            },
            "instructions": summary["instructions"]["overall"]["correct"],
            "weights_sha256": manifest["model"]["weights_sha256"],
        }
        if name != "reference":
            arm, step = endpoint_steps[name]
            checkpoint = root / f"jobs/{arm}/step-{step:08d}"
            state = read(checkpoint / "diagnostic.json")["state"]
            weights = checkpoint / "model/model.safetensors"
            require(digest(weights) == stat["weights_sha256"], "Scored checkpoint identity")
            elements, hashes = 0, {}
            with safe_open(weights, framework="numpy") as f:
                for key in f.keys():
                    array = f.get_tensor(key)
                    require(
                        array.dtype == np.float32 and np.isfinite(array).all(),
                        "Invalid included weights",
                    )
                    layout = {
                        "shape": list(array.shape),
                        "stride": [n // array.itemsize for n in array.strides],
                        "dtype": "torch.float32",
                    }
                    h = hashlib.sha256(canonical(layout))
                    h.update(array.tobytes(order="C"))
                    hashes[key] = h.hexdigest()
                    elements += array.size
            require(
                hashes == state["model"] and elements == arms[arm]["parameters"],
                "Tensor/trace mismatch",
            )
            checks = comparisons["fixed_gate_comparisons"][name]["checks"]
            stat.update(
                failed_gates=[k for k, v in checks.items() if not v["pass"]],
                bpb_ratio=checks["matched_bpb_ratio"]["value"],
                tensor_elements=elements,
                tensors_verified=len(hashes),
                source_step=step,
            )
        endpoints[name] = stat
    contrasts = {}
    for a, b in (("A1", "B1"), ("B1", "B2"), ("A1", "B2"), ("B2", "C2")):
        left, right = endpoints[a], endpoints[b]
        rows_a, rows_b = [read(root / f"evaluation-{name}/generation.json") for name in (a, b)]
        require([r["id"] for r in rows_a] == [r["id"] for r in rows_b], "Prompt pairing")
        changes = [
            y["observations"]["repeated_trigram_fraction"]
            - x["observations"]["repeated_trigram_fraction"]
            for x, y in zip(rows_a, rows_b, strict=True)
        ]
        contrasts[a + ":" + b] = {
            "bpb_relative_change_percent": 100 * (right["bpb"] / left["bpb"] - 1),
            "book_bpb_relative_change_percent": {
                k: 100 * (right["book_bpb"][k] / v - 1) for k, v in left["book_bpb"].items()
            },
            "repetition_change_percentage_points": 100 * (right["repetition"] - left["repetition"]),
            "repetition_prompts_improved_tied_worse": [
                sum(x < 0 for x in changes),
                sum(x == 0 for x in changes),
                sum(x > 0 for x in changes),
            ],
            "external_paired": comparisons["descriptive_contrasts"][a + ":" + b]["paired"],
        }
    prior_base = Path("outputs/phase17-1-return-reviewed/run/development-reference")
    require(
        {
            k: v
            for k, v in read(root / "evaluation-reference/summary.json").items()
            if k != "resources"
        }
        == {k: v for k, v in read(prior_base / "summary.json").items() if k != "resources"},
        "Reference aggregate drift",
    )
    for group in ("generation", "instructions"):
        now, old = [read(p / f"{group}.json") for p in (root / "evaluation-reference", prior_base)]
        require(
            all(a["generated_ids"] == b["generated_ids"] for a, b in zip(now, old, strict=True)),
            "Reference generated IDs drift",
        )
    before = read(correction / "before.json")
    accounted = (
        sum(v["bytes"] for v in before.values())
        + Path("outputs/small-subset-corrected-return.zip").stat().st_size
        + sum(p.stat().st_size for p in correction.rglob("*") if p.is_file())
        + 128 * 1024**2
    )
    require(accounted < 8 * 1024**3, "Combined artifact cap")
    original_time = read(original / "guardian.json")["total_wall_seconds"]
    auth = read(correction / "authorization.json")
    new_guardian = read(Path("outputs/guardian.json"))
    require(
        auth["prior_debit_seconds"] == max(1273, math.ceil(original_time)) == 1273, "Prior debit"
    )
    require(
        auth["prior_packaging_debit_seconds"]
        == math.ceil(1273 - read(root / "outcome.json")["elapsed_before_package"])
        == 9,
        "Packaging debit",
    )
    projection = (
        3 * times["C"]
        + times["eval-reference"]
        + 3 * times["eval-C2"]
        + phase_times["prepare"]
        + new_guardian["cumulative_packaging_seconds"]
    )
    return {
        "scope": "Second saved-evidence pass; no model construction/forward/optimizer/scoring",
        "original_source_commit": original_manifest["source_commit"],
        "correction_source_commit": "0eca1d836f24b2b80ad93730db0719367598bc68",
        "original_guardian_seconds": original_time,
        "prior_debit_seconds": 1273,
        "prior_packaging_debit_seconds": 9,
        "correction_seconds": new_guardian["correction_wall_seconds"],
        "cumulative_seconds": new_guardian["cumulative_charged_seconds"],
        "cumulative_packaging_seconds": new_guardian["cumulative_packaging_seconds"],
        "stage_seconds": phase_times,
        "job_seconds": times,
        "combined_artifacts_bytes_with_reserve": accounted,
        "arms": arms,
        "endpoints": endpoints,
        "contrasts": contrasts,
        "A_B_common_warmup_exact_updates": 19,
        "cross_host_adjacent_cosine_cases": analytic_ulps,
        "cross_host_maximum_output_ULPs": maximum_ulps,
        "same_Windows_schedule_matches": "all 1910 updates exact",
        "historical_reference_repeat": (
            "Quality aggregates and generated IDs exact; elapsed resource timing excluded"
        ),
        "omitted_records_not_rehashed": 43,
        "new_optimizer_updates": 0,
        "new_model_scoring": False,
        "accepted_base": False,
        "prospective_only": {
            "direction": "Independent-seed exposure confirmation; not implemented or authorized",
            "model_parameters": 21238272,
            "seed": 161,
            "passes": [2, 4],
            "updates": 2292,
            "target_executions": 11658090,
            "same_unique_target_positions": 1943015,
            "central_seconds_from_measured_jobs": projection,
            "target_total_seconds": 1200,
            "proposed_hard_total_seconds": 1800,
            "owner_ceiling_seconds": 2700,
            "paid_compute": 0,
            "full_corpus_rerun": False,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "endpoints": len(result["endpoints"]) - 1,
                "new_optimizer_updates": 0,
            }
        )
    )
