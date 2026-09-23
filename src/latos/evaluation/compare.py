"""Separate, explicit Roadmap 2.0 gates and paired descriptive uncertainty."""

import argparse
import math
import random
from pathlib import Path

from latos.evaluation.inputs import read_json, write_json
from latos.evaluation.scoring import accuracy, lm_summary
from latos.model.storage import file_hash


def verified_run(path):
    path = Path(path)
    inventory = read_json(path / "inventory.json")
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
    if set(inventory) != actual - {"inventory.json"}:
        raise ValueError("Incomplete evidence inventory")
    for name, digest in inventory.items():
        if file_hash(path / name) != digest:
            raise ValueError("Evaluation evidence checksum mismatch")
    manifest, summary = read_json(path / "manifest.json"), read_json(path / "summary.json")
    if manifest["status"] != "complete" or manifest["failures"] or manifest["skips"]:
        raise ValueError("Incomplete evaluation cannot pass a gate")
    inputs = read_json(path / "inputs.json")
    # The captured suite is identified by digest, not by a caller-selected filename.
    suites = [
        p for p in (path / "source").glob("*.json") if file_hash(p) == manifest["suite_sha256"]
    ]
    if len(suites) != 1:
        raise ValueError("Missing or ambiguous captured suite")
    suite = read_json(suites[0])

    def matching_ids(rows, expected):
        ids = [r["id"] for r in rows]
        if ids != [r["id"] for r in expected] or len(set(ids)) != len(ids):
            raise ValueError("Missing, duplicate or reordered evaluation cases")

    for mode in ("legacy", "matched"):
        rows = read_json(path / f"lm-{mode}.json")
        matching_ids(rows, inputs["lm"])
        for row, expected in zip(rows, inputs["lm"], strict=True):
            keys = ("document", "text_sha256") + (("bytes",) if mode == "matched" else ())
            if any(row[key] != expected[key] for key in keys):
                raise ValueError("Language-model case identity mismatch")
        expected = {
            **lm_summary(rows),
            "documents": {
                name: lm_summary([r for r in rows if r["document"] == name])
                for name in sorted({r["document"] for r in rows})
            },
        }
        if summary[mode] != expected:
            raise ValueError("Language-model aggregate disagrees with raw rows")
    instructions = read_json(path / "instructions.json")
    generations = read_json(path / "generation.json")
    from latos.evaluation.behavior import instruction_correct, observations
    from latos.evaluation.runner import summarize_behavior

    for rows, expected_cases in [
        (instructions, suite["instructions"]),
        (generations, suite["generation"]),
    ]:
        matching_ids(rows, expected_cases)
        for row, case in zip(rows, expected_cases, strict=True):
            if (
                row["status"] != "ok"
                or row["prompt"] != case["prompt"]
                or row["category"] != case["category"]
            ):
                raise ValueError("Incomplete or mismatched behavior case")
            if row["observations"] != observations(row["text"], row["stop_reason"]):
                raise ValueError("Generation observations disagree with text")
            if "scorer" in case and (
                row["correct"] != instruction_correct(row["text"], case)
                or row["family"] != case["family"]
            ):
                raise ValueError("Instruction correctness disagrees with output")
    ins_summary, gen_summary = summarize_behavior(instructions, generations, suite)
    if summary["instructions"] != ins_summary or summary["generation"] != gen_summary:
        raise ValueError("Behavior aggregate disagrees with raw rows")
    if set(summary["external"]) != set(inputs["external"]):
        raise ValueError("Missing external benchmark")
    for name, result in summary["external"].items():
        rows = read_json(path / f"{name}.json")
        matching_ids(rows, inputs["external"][name])
        for row, expected in zip(rows, inputs["external"][name], strict=True):
            if any(row[key] != value for key, value in expected.items()) or row["status"] != "ok":
                raise ValueError("External case identity or completeness mismatch")
            scores, sizes = row["log_likelihoods"], row["answer_bytes"]
            if (
                len(scores) != row["choices"]
                or len(sizes) != len(scores)
                or any(not math.isfinite(v) or v > 0 for v in scores)
                or any(type(v) is not int or v <= 0 for v in sizes)
            ):
                raise ValueError("Malformed choice likelihoods")
            normalized = [v / n for v, n in zip(scores, sizes, strict=True)]
            prediction = max(range(len(scores)), key=scores.__getitem__)
            norm_prediction = max(range(len(scores)), key=normalized.__getitem__)
            if (
                row["prediction"] != prediction
                or row["byte_normalized_prediction"] != norm_prediction
                or row["correct"] != (prediction == row["gold"])
                or row["correct_byte_normalized"] != (norm_prediction == row["gold"])
                or row["byte_normalized_log_likelihoods"] != normalized
            ):
                raise ValueError("External prediction disagrees with choice likelihoods")
        expected = {
            "raw": accuracy(rows),
            "byte_normalized": accuracy(rows, "correct_byte_normalized"),
            "uniform_chance": math.fsum(1 / r["choices"] for r in rows) / len(rows),
            "first_choice": sum(r["gold"] == 0 for r in rows) / len(rows),
            "failures": 0,
        }
        if result != expected:
            raise ValueError("External aggregate disagrees with raw rows")
    return manifest, summary


def paired_interval(before, after, *, key="correct", cluster_key=None, samples=2000):
    """Seeded paired bootstrap; correlated templates can be resampled as families."""
    if not before or [r["id"] for r in before] != [r["id"] for r in after]:
        raise ValueError("Paired comparison requires identical ordered case IDs")
    groups = {}
    for a, b in zip(before, after, strict=True):
        group = a[cluster_key] if cluster_key else a["id"]
        groups.setdefault(group, []).append(float(b[key]) - float(a[key]))
    values = list(groups.values())
    rng, draws = random.Random(140), []
    for _ in range(samples):
        chosen = [values[rng.randrange(len(values))] for _ in values]
        draws.append(math.fsum(sum(g) for g in chosen) / sum(map(len, chosen)))
    draws.sort()
    return {
        "delta": math.fsum(sum(g) for g in values) / sum(map(len, values)),
        "bootstrap95": [draws[int(samples * 0.025)], draws[int(samples * 0.975)]],
        "clusters": len(values),
        "samples": samples,
        "seed": 140,
    }


def gate_metrics(reference, candidate, protocol, stage, *, historical=None):
    if stage not in ("base", "assistant"):
        raise ValueError("Unknown quality gate")
    limits = protocol["gates"][stage]
    checks = {}

    def check(name, value, threshold, relation):
        if not math.isfinite(value) or not math.isfinite(threshold):
            raise ValueError("Nonfinite gate metric")
        checks[name] = {
            "value": value,
            "threshold": threshold,
            "relation": relation,
            "pass": value <= threshold if relation == "<=" else value >= threshold,
        }

    check(
        "matched_bpb_ratio",
        candidate["matched"]["bits_per_byte"] / reference["matched"]["bits_per_byte"],
        limits["max_bpb_ratio"],
        "<=",
    )
    if set(reference["matched"]["documents"]) != set(candidate["matched"]["documents"]):
        raise ValueError("Language-model documents differ")
    for name, value in candidate["matched"]["documents"].items():
        check(
            "document_bpb_ratio:" + name,
            value["bits_per_byte"] / reference["matched"]["documents"][name]["bits_per_byte"],
            limits["max_document_bpb_ratio"],
            "<=",
        )
    for name, value in candidate["external"].items():
        check(
            "external_raw_drop:" + name,
            reference["external"][name]["raw"]["accuracy"] - value["raw"]["accuracy"],
            limits["external_max_drop"],
            "<=",
        )
        check(
            "external_byte_drop:" + name,
            reference["external"][name]["byte_normalized"]["accuracy"]
            - value["byte_normalized"]["accuracy"],
            limits["external_max_drop"],
            "<=",
        )
    for name, limit in [
        ("empty", "generation_max_empty"),
        ("repeated_trigram_fraction", "generation_max_repetition"),
    ]:
        check("generation_" + name, candidate["generation"][name], limits[limit], "<=")
    check(
        "generation_repetition_increase",
        candidate["generation"]["repeated_trigram_fraction"]
        - reference["generation"]["repeated_trigram_fraction"],
        limits["generation_max_repetition_increase"],
        "<=",
    )
    if stage == "base":
        arc = candidate["external"]["ARC-Easy"]
        check("arc_easy_min", arc["raw"]["accuracy"], limits["arc_easy_min"], ">=")
        check(
            "arc_easy_gain",
            arc["raw"]["accuracy"] - reference["external"]["ARC-Easy"]["raw"]["accuracy"],
            limits["arc_easy_min_gain"],
            ">=",
        )
        # Strictly above chance, including a descriptive sampling uncertainty buffer.
        check(
            "arc_easy_lower_bound", arc["raw"]["wilson95"][0], arc["uniform_chance"] + 1e-12, ">="
        )
    else:
        if historical is None:
            raise ValueError("Assistant gate requires a fixed historical instruction reference")
        score = candidate["instructions"]["overall"]["accuracy"]
        check("instruction_min", score, limits["instruction_min"], ">=")
        check(
            "instruction_gain",
            score - historical["instructions"]["overall"]["accuracy"],
            limits["instruction_min_gain"],
            ">=",
        )
        for name, result in candidate["instructions"]["categories"].items():
            check("instruction_category:" + name, result["accuracy"], limits["category_min"], ">=")
    return {
        "stage": stage,
        "passed": all(c["pass"] for c in checks.values()),
        "checks": checks,
        "scope": (
            "numeric gates only; contamination, selection and final-access audit also required"
        ),
    }


HISTORICAL_BASE = "f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa"
HISTORICAL_PANEL = {
    HISTORICAL_BASE,
    "62b890214e5544292cc54b725a4e505730fca0d501b3337cdd3f051a622d09ae",
    "faaf9ec7cd3de329ecc49d82fe08a38e6e72ec4bc8965b29a349bb751526ff18",
    "84f70f8bd08829ecc336017b1e39041eca67cfd390755e9e55d9cba1ffb79f94",
    "5ec902472e6e04ca5c97475d44a30818729a2fc5e8592544642767af83ae9ca1",
}


def compare(reference, candidate, stage, historical=None):
    rm, rs = verified_run(reference)
    cm, cs = verified_run(candidate)
    if stage == "base" and rm["model"]["weights_sha256"] != HISTORICAL_BASE:
        raise ValueError("Base gate requires the frozen Phase 5 historical reference")
    for key in ("protocol_sha256", "suite_sha256", "external_sha256", "data"):
        if rm[key] != cm[key]:
            raise ValueError("Incompatible evaluation identities")
    for key in (
        "device",
        "torch",
        "python",
        "threads",
        "batch_size",
        "platform",
        "packages",
        "deterministic_algorithms",
        "cuda_matmul_tf32",
        "cudnn_tf32",
        "cudnn_benchmark",
    ):
        if rm["environment"][key] != cm["environment"][key]:
            raise ValueError("Gate comparisons require the same backend/runtime/batching")
    historical_summary = None
    if historical is not None:
        hm, historical_summary = verified_run(historical)
        if (
            hm["suite_sha256"] != cm["suite_sha256"]
            or hm["protocol_sha256"] != cm["protocol_sha256"]
        ):
            raise ValueError("Historical instruction suite mismatch")
        if hm["environment"] != cm["environment"]:
            raise ValueError("Historical instructions require the same backend/runtime/batching")
        if hm["model"]["weights_sha256"] not in HISTORICAL_PANEL:
            raise ValueError("Instruction reference is outside the frozen historical panel")
    report = gate_metrics(rs, cs, cm["protocol"], stage, historical=historical_summary)
    report.update(
        protocol_sha256=cm["protocol_sha256"],
        reference_weights=rm["model"]["weights_sha256"],
        candidate_weights=cm["model"]["weights_sha256"],
        token_perplexity_comparable=rm["model"]["config"]["tokenizer_sha256"]
        == cm["model"]["config"]["tokenizer_sha256"],
        paired={},
    )
    for name in cs["external"]:
        report["paired"][name] = paired_interval(
            read_json(Path(reference) / f"{name}.json"), read_json(Path(candidate) / f"{name}.json")
        )
    report["paired"]["instructions"] = paired_interval(
        read_json(Path(reference) / "instructions.json"),
        read_json(Path(candidate) / "instructions.json"),
        cluster_key="family",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare fixed evaluations; no composite score")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--stage", choices=("base", "assistant"), required=True)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Comparison output already exists")
    write_json(args.output, compare(args.reference, args.candidate, args.stage, args.historical))


if __name__ == "__main__":
    main()
