"""Run the fixed equal-update LoRA/full comparison in separate processes."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from latos.data.manifest import canonical_json
from latos.model.storage import file_hash

BASE = "outputs/phase-5-english-pilot/step-00003000/model"
BASE_HASH = "f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa"


def compare(root):
    plans, baselines, summaries, metrics = [], [], [], []
    for method in ("lora", "full"):
        directory = root / method
        plans.append(json.loads((directory / "plan.json").read_text()))
        baselines.append(json.loads((directory / "baseline.json").read_text()))
        summaries.append(json.loads((directory / "summary.json").read_text()))
        metrics.append(
            [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        )
    for method, plan, summary, rows in zip(
        ("lora", "full"), plans, summaries, metrics, strict=True
    ):
        if (
            plan["method"] != method
            or summary["steps"] != plan["config"]["max_steps"]
            or len(rows) != summary["steps"]
            or [row["step"] for row in rows] != list(range(1, summary["steps"] + 1))
            or sum(row["targets"] for row in rows) != summary["tokens_seen"]
            or rows[-1]["windows_seen"] != summary["windows_seen"]
        ):
            raise ValueError("Incomplete or inconsistent comparison run")
    for key in (
        "config",
        "model_config",
        "base_weights_sha256",
        "tokenizer_sha256",
        "chat_contract",
        "datasets",
        "regression",
        "runtime",
        "sampling",
    ):
        if plans[0][key] != plans[1][key]:
            raise ValueError(f"Comparison input mismatch: {key}")
    if baselines[0] != baselines[1]:
        raise ValueError("Starting metrics or samples differ")
    for left, right in zip(metrics[0], metrics[1], strict=True):
        for key in (
            "step",
            "learning_rate",
            "targets",
            "tokens_seen",
            "windows_seen",
            "epoch",
            "sampler_sha256",
        ):
            if left[key] != right[key]:
                raise ValueError(f"Per-update comparison mismatch: {key}")
    result = {
        "equal_inputs_baselines_and_update_exposures": True,
        "runs": {},
        "baseline": baselines[0],
        "protocol_sha256": file_hash(Path(__file__).with_name("PLAN.md")),
    }
    for method, plan, summary in zip(("lora", "full"), plans, summaries, strict=True):
        result["runs"][method] = {"plan": plan, "result": summary}
    result["artifact_bytes"] = {
        name: (root / name).stat().st_size
        for name in (
            "lora/adapter/adapter.safetensors",
            "lora/adapter/metadata.json",
            "lora/merged/model.safetensors",
            "full/model/model.safetensors",
        )
    }
    (root / "comparison.json").write_bytes(canonical_json(result))
    return result


def run(root, device):
    root.mkdir(parents=True, exist_ok=False)
    for method in ("lora", "full"):
        command = [
            sys.executable,
            "-m",
            "latos",
            "adapt",
            "train",
            "--method",
            method,
            "--base",
            BASE,
            "--base-sha256",
            BASE_HASH,
            "--tokenizer-dir",
            "artifacts/tokenizers/english-bpe-v1",
            "--manifest",
            "data/manifests/english-books-v1.json",
            "--corpus-dir",
            "data/processed/english-books-v1",
            "--conversations",
            "data/processed/english-instructions-v1",
            "--config",
            "configs/training/adaptation-pilot.json",
            "--rank",
            "8",
            "--alpha",
            "16",
            "--adapter-seed",
            "91",
            "--device",
            device,
            "--threads",
            "1",
            "--output-dir",
            str(root / method),
        ]
        with (root / f"{method}.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"{method}: completed", flush=True)
    compare(root)
    print("Input, baseline and exposure equality: PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    args = parser.parse_args()
    run(args.output_dir, args.device)
