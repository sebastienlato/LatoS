"""Apply the frozen Phase 16 rule and project one full pass without training."""

import argparse
import json
import math
from pathlib import Path


def select(verified, layout):
    train = layout["splits"]["train"]
    dev = layout["splits"]["development"]
    trials = verified["candidates"]
    eligible = []
    estimates = {}
    for candidate in trials:
        if candidate["precision"] != "bfloat16":
            continue
        peer = next(
            c for c in trials if c["layers"] == candidate["layers"] and c["precision"] == "float32"
        )
        rate = candidate["measured_targets_per_second"]
        time = candidate["timings"]
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Invalid throughput")
        checkpoints = list(range(1000, train["updates_at_accumulation8"], 1000)) + [
            train["updates_at_accumulation8"]
        ]
        update_seconds = train["targets_with_EOS"] / rate
        # Full development is larger than the probe; extrapolate target-weighted cost.
        validation_seconds = (
            (len(checkpoints) + 1)
            * max(time["initial_validation"], time["final_validation"])
            * dev["targets_with_EOS"]
            / candidate["final_development"]["targets"]
        )
        checkpoint_seconds = time["initial_checkpoint"] + len(checkpoints) * max(
            time["midpoint_checkpoint"], time["final_checkpoint"]
        )
        central = update_seconds + validation_seconds + checkpoint_seconds
        projected = 2 * central + 300  # Twofold throughput/overhead margin + 5 min preprocessing.
        checks = {
            "both_precision_probes_complete": peer["status"] == candidate["status"] == "measured",
            "numerical_recovery_control": verified["control"]["status"] == "passed",
            "early_loss_below_initialization": candidate["final_development"]["loss"]
            < candidate["initial_development"]["loss"],
            "memory_headroom": all(
                c["resources"]["peak_reserved_bytes"] <= 0.85 * c["total_vram_bytes"]
                for c in (candidate, peer)
            ),
            "inference_targets": all(
                c["inference"]["median_first_token_seconds"] <= 1
                and c["inference"]["median_decode_tokens_per_second"] >= 10
                for c in (candidate, peer)
            ),
            "one_pass_runtime_under_24h": projected < 24 * 3600,
        }
        estimates[str(candidate["layers"])] = {
            "gates": checks,
            "target_update_seconds": update_seconds,
            "full_development_seconds_estimate": validation_seconds,
            "checkpoint_seconds_estimate": checkpoint_seconds,
            "central_training_plus_periodic_overhead_seconds": central,
            "conservative_training_seconds": projected,
            "checkpoint_steps": checkpoints,
            "note": (
                "Projection from short CUDA measurements; not measured full-run "
                "duration. Includes 2x margin plus 300 s preprocessing; final fixed "
                "evaluation separately reserved."
            ),
        }
        if all(checks.values()):
            eligible.append(candidate)
    if not eligible:
        raise ValueError("No eligible configuration; bounded revision needed")
    best = min(c["final_development"]["loss"] for c in eligible)
    band = [c for c in eligible if c["final_development"]["loss"] <= 1.02 * best]
    chosen = min(band, key=lambda c: c["parameters"])
    return {
        "rule": (
            "Smallest eligible model within 2% of lowest final short-probe "
            "loss; no learned-quality ranking"
        ),
        "eligible_layers": [c["layers"] for c in eligible],
        "within_two_percent_layers": [c["layers"] for c in band],
        "relative_probe_loss_to_minimum": {
            str(c["layers"]): c["final_development"]["loss"] / best for c in eligible
        },
        "selected_layers": chosen["layers"],
        "selected_parameters": chosen["parameters"],
        "precision": "bfloat16",
        "estimates": estimates,
        "phase17_started": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verified", type=Path, required=True)
    p.add_argument("--layout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Output exists")
    result = select(json.loads(a.verified.read_text()), json.loads(a.layout.read_text()))
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
