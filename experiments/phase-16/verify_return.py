"""Independently verify the compact Windows evidence against the measured Git source."""

import argparse
import json
import math
import statistics
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from latos.model import ModelConfig
from latos.model.storage import file_hash
from latos.training.config import TrainingConfig
from latos.training.data import ShuffleStream
from latos.training.data2 import prepare_probe_dataset
from latos.training.probe import BUNDLE_SHA256, TOKENIZER_SHA256, verify_inputs

SOURCE = "3cb6a47743e698a303deac0ead2664c0c89cf5e5"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_record(path, record):
    require(path.is_file(), f"Missing evidence: {path.name}")
    require(
        path.stat().st_size == record["bytes"] and file_hash(path) == record["sha256"],
        f"Evidence identity mismatch: {path.name}",
    )


def verify(root, package):
    manifest = read(root / "evidence-subset-manifest.json")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(
        actual == set(manifest["files"]) | {"evidence-subset-manifest.json"},
        "Subset file set differs",
    )
    for name, record in manifest["files"].items():
        require(
            not Path(name).is_absolute() and ".." not in Path(name).parts and "\\" not in name,
            "Unsafe path",
        )
        verify_record(root / name, record)
    handoff = read(root / "handoff.json")
    require(handoff["source_commit"] == SOURCE, "Unexpected source commit")
    require(handoff["input_bundle_sha256"] == BUNDLE_SHA256, "Unexpected input identity")
    source_files = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", SOURCE], text=True
    ).splitlines()
    require(
        {n[7:] for n in handoff["files"] if n.startswith("source/")} == set(source_files),
        "Source file inventory differs",
    )
    import hashlib

    for name in source_files:
        data = subprocess.check_output(["git", "show", f"{SOURCE}:{name}"])
        record = handoff["files"]["source/" + name]
        require(
            len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"],
            "Handoff source differs from Git",
        )
    verify_inputs(package)
    require(
        file_hash(root / "accepted-input-bundle.json") == BUNDLE_SHA256,
        "Returned input bundle differs",
    )
    source_checks = 0
    for p in (root / "reviewed-source").rglob("*"):
        if p.is_file():
            name = p.relative_to(root / "reviewed-source").as_posix()
            require(
                p.read_bytes() == subprocess.check_output(["git", "show", f"{SOURCE}:{name}"]),
                "Reviewed source changed",
            )
            source_checks += 1
    originals = read(root / "probe/inventory.json")
    omitted = {}
    for name, record in originals.items():
        p = root / "probe" / name
        if p.exists():
            verify_record(p, record)
        else:
            require(name.endswith(".safetensors"), "Non-tensor original evidence omitted")
            require(
                manifest["excluded_tensor_files"].get(name) == record,
                "Omitted tensor identity differs",
            )
            omitted[name] = record
    require(omitted == manifest["excluded_tensor_files"], "Excluded tensor set differs")
    snapshots = []
    for folder in [
        root / "probe",
        *(p for p in (root / "probe").glob("layers-*") if p.is_dir()),
        root / "probe/control",
    ]:
        if (folder / "inventory.json").exists() and folder != root / "probe":
            for name, record in read(folder / "inventory.json").items():
                outer = (folder.relative_to(root / "probe") / name).as_posix()
                require(originals.get(outer) == record, "Nested inventory differs")
        paths = list((folder / "source/latos").rglob("*.py"))
        require(len(paths) == 60, "Imported source snapshot incomplete")
        for p in paths:
            name = "src/latos/" + p.relative_to(folder / "source/latos").as_posix()
            require(
                p.read_bytes() == subprocess.check_output(["git", "show", f"{SOURCE}:{name}"]),
                "Executed source changed",
            )
        snapshots.append(len(paths))
    tests = {}
    for name in ["development-tests", "wheel-tests-source-cwd"]:
        cases = list(ET.parse(root / f"validation/{name}.xml").iter("testcase"))
        skips = [c for c in cases if c.find("skipped") is not None]
        failures = [
            c for c in cases if c.find("failure") is not None or c.find("error") is not None
        ]
        cuda = [c for c in cases if c.attrib["name"] == "test_cuda_bf16_numerical_control"]
        require(
            len(cases) == 403 and len(skips) == 15 and not failures, "Windows test result mismatch"
        )
        require(len(cuda) == 1 and cuda[0] not in skips, "BF16 test did not execute")
        tests[name] = {
            "passed": len(cases) - len(skips),
            "skipped": len(skips),
            "failed": 0,
            "skip_reasons": [c.find("skipped").attrib.get("message") for c in skips],
        }
    commands = [
        json.loads(line) for line in (root / "validation/commands.jsonl").read_text().splitlines()
    ]
    executions = [c for c in commands if c["name"] == "bounded-cuda-suite"]
    require(
        len(executions) == 1 and executions[0]["returncode"] == 0,
        "Suite command count/result differs",
    )
    summary = read(root / "probe/summary.json")
    jobs = summary["jobs"]
    expected = ["control"] + [
        f"layers-{n}-{p}" for n in (8, 14, 20) for p in ("float32", "bfloat16")
    ]
    require(
        [j["job"] for j in jobs] == expected
        and all(j["returncode"] == 0 and not j["timeout"] for j in jobs),
        "Job count/order/result differs",
    )
    control = read(root / "probe/control/result.json")
    for key, limit in [
        ("loss_absolute_difference", 0.03),
        ("gradient_relative_l2", 0.05),
        ("four_update_weight_relative_l2", 0.03),
        ("accumulation_weight_relative_l2", 0.03),
    ]:
        require(math.isfinite(control[key]) and control[key] <= limit, "Numerical tolerance failed")
    require(
        control["status"] == "passed"
        and control["resume_and_failure_recovery"] == "passed"
        and control["validation_preserves_state"],
        "Control incomplete",
    )
    train, ti = prepare_probe_dataset(
        package / "corpus", package / "tokenizer", "train", 512, max_documents=4096
    )
    dev, di = prepare_probe_dataset(
        package / "corpus", package / "tokenizer", "development", 512, max_documents=128
    )
    require(not set(ti["groups"]) & set(di["groups"]), "Split group overlap")
    rows = []
    for name in expected[1:]:
        folder = root / "probe" / name
        result = read(folder / "result.json")
        run = read(folder / "run.json")
        env = read(folder / "environment.json")
        require(
            "RTX 4070 SUPER" in env["gpu"]
            and env["native_bfloat16"]
            and env["capability"] == [8, 9],
            "Wrong GPU",
        )
        require(
            env["runtime"]["implementation_sha256"]
            == "6467d629600d299dd07ad4d93e3284da5852b0419de0397d3100b16a5ae7069f",
            "Implementation differs",
        )
        require(
            env["runtime"]["cpu_threads"] == 4
            and not env["runtime"]["cuda_matmul_tf32"]
            and not env["runtime"]["cudnn_tf32"]
            and env["runtime"]["matmul_precision"] == "highest",
            "Runtime policy differs",
        )
        model = ModelConfig(**run["model"])
        config = TrainingConfig(**run["training"])
        layers = int(name.split("-")[1])
        precision = name.split("-")[2]
        steps = 8 if precision == "float32" else 64
        require(
            model.to_dict()
            == ModelConfig(
                1, 16384, 512, 512, 8, layers, 1408, 10000.0, 1e-5, TOKENIZER_SHA256
            ).to_dict(),
            "Model differs",
        )
        require(
            config
            == TrainingConfig(
                sequence_length=512,
                batch_size=2,
                accumulation_steps=8,
                max_steps=steps,
                warmup_steps=4,
                learning_rate=0.0003,
                seed=160,
            ),
            "Training differs",
        )
        require(
            result["parameters"] == model.parameter_count and run["precision"] == precision,
            "Count/precision differs",
        )
        require(
            run["datasets"] == {"train": train.identity, "validation": dev.identity},
            "Datasets differ from Mac reconstruction",
        )
        data = read(folder / "data.json")
        require(data == {"train": ti, "development": di}, "Data accounting differs")
        updates = [json.loads(line) for line in (folder / "updates.jsonl").read_text().splitlines()]
        require(len(updates) == steps, "Update count differs")
        stream = ShuffleStream(len(train.windows), 160)
        total = 0
        for i, u in enumerate(updates, 1):
            count = sum(train.target_count(j) for _ in range(8) for j in stream.take(2))
            total += count
            require(
                u["step"] == i
                and u["targets"] == count
                and u["tokens_seen"] == total
                and u["windows_seen"] == 16 * i,
                "Exposure differs",
            )
            require(
                abs(u["learning_rate"] - config.learning_rate_at(i))
                <= math.ulp(config.learning_rate_at(i)),
                "Schedule differs beyond one host-math ULP",
            )
            require(
                all(
                    math.isfinite(u[k])
                    for k in ["loss", "grad_norm_before_clip", "seconds", "targets_per_second"]
                )
                and u["seconds"] > 0,
                "Nonfinite metric",
            )
            require(
                math.isclose(u["targets_per_second"], count / u["seconds"], rel_tol=1e-12),
                "Update throughput differs",
            )
        measured = sum(u["targets"] for u in updates[2:]) / sum(u["seconds"] for u in updates[2:])
        require(
            math.isclose(result["measured_targets_per_second"], measured, rel_tol=1e-12)
            and result["tokens_seen"] == total,
            "Aggregate throughput differs",
        )
        for checkpoint, step in [("initial", 0), ("midpoint", steps // 2), ("final", steps)]:
            state = read(folder / checkpoint / "state.json")
            meta = read(folder / checkpoint / "model/metadata.json")
            require(
                state["step"] == step
                and state["tokens_seen"] == sum(u["targets"] for u in updates[:step])
                and state["cursor"] == step * 16
                and state["epoch"] == 0,
                "Checkpoint progress differs",
            )
            require(
                state["config"] == config.to_dict() and state["datasets"] == run["datasets"],
                "Checkpoint contract differs",
            )
            require(
                meta["tokenizer_sha256"] == TOKENIZER_SHA256
                and meta["parameter_count"] == model.parameter_count,
                "Checkpoint model differs",
            )
            require(
                file_hash(folder / checkpoint / "model/metadata.json")
                == state["model_metadata_sha256"],
                "Model metadata mismatch",
            )
            prefix = f"{name}/{checkpoint}/"
            require(
                omitted[prefix + "training.safetensors"]["sha256"] == state["training_sha256"],
                "Optimizer identity mismatch",
            )
            require(
                omitted[prefix + "model/model.safetensors"]["sha256"] == meta["weights_sha256"],
                "Weights identity mismatch",
            )
        checkpoint_bytes = sum(
            v["bytes"] for k, v in originals.items() if k.startswith(name + "/final/")
        )
        require(checkpoint_bytes == result["checkpoint_bytes"], "Checkpoint byte count differs")
        inf = result["inference"]
        for field in ("first_token_seconds", "decode_tokens_per_second"):
            require(
                statistics.median(x[field] for x in inf["repeats"]) == inf["median_" + field],
                "Inference median differs",
            )
        require(
            result["resources"]["peak_reserved_bytes"]
            >= max(u["peak_reserved_bytes"] for u in updates),
            "Peak below recorded updates",
        )
        require(
            result["resources"]["peak_allocated_bytes"]
            <= result["resources"]["peak_reserved_bytes"]
            < 0.85 * env["total_bytes"],
            "Memory gate failed",
        )
        require(
            inf["median_first_token_seconds"] <= 1 and inf["median_decode_tokens_per_second"] >= 10,
            "Inference gate failed",
        )
        require(
            result["final_development"]["loss"] < result["initial_development"]["loss"],
            "Early loss gate failed",
        )
        require(
            result["final_development"]["targets"] == 57036 and result["status"] == "measured",
            "Development incomplete",
        )
        rows.append(
            {
                "job": name,
                **result,
                "total_vram_bytes": env["total_bytes"],
                "retained_job_bytes": sum(
                    v["bytes"] for k, v in originals.items() if k.startswith(name + "/")
                ),
            }
        )
    return {
        "source_commit": SOURCE,
        "subset_files_verified": len(manifest["files"]),
        "git_source_files_bound": len(source_files),
        "reviewed_source_files_verified": source_checks,
        "source_snapshots_verified": snapshots,
        "omitted_tensor_files": len(omitted),
        "omitted_tensor_bytes": sum(v["bytes"] for v in omitted.values()),
        "tests": tests,
        "control": control,
        "candidates": rows,
        "schedule_cross_host_note": (
            "BF16 steps 23 and 45 differ by exactly one float64 ULP between "
            "Windows and Mac cosine evaluation; same source/config, no CUDA "
            "tolerance changes."
        ),
        "retained_invocation_failures": [
            {"name": c["name"], "returncode": c["returncode"]}
            for c in commands
            if c["returncode"] != 0
        ],
        "scope": (
            "Hashes, source, protocol, exposure, timing aggregates and metadata "
            "independently verified. Tensor payloads remain on Windows; no Mac "
            "model replay or independent recomputation of losses."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output already exists")
    result = verify(args.evidence, args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "subset_files_verified",
                    "git_source_files_bound",
                    "source_snapshots_verified",
                    "omitted_tensor_files",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
