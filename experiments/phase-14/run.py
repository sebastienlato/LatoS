"""Bounded retrospective evaluation of the five pinned historical dense artifacts."""

import argparse
import subprocess
import sys
from pathlib import Path

from latos.evaluation.inputs import read_json, write_json
from latos.model.storage import file_hash

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    models = read_json(ROOT / "experiments/phase-14/models.json")
    freeze_paths = [
        ROOT / "experiments/phase-14/PLAN.md",
        *sorted((ROOT / "configs/evaluation").glob("*.json")),
    ]
    freeze = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in freeze_paths}
    write_json(args.output_dir / "freeze.json", freeze)
    for name in ["base", "sft", "lora-merged", "full-control", "dpo", "base-repeat"]:
        model = models["base" if name == "base-repeat" else name]
        command = [
            sys.executable,
            "-m",
            "latos.evaluation",
            "run",
            "--model-dir",
            model["path"],
            "--weights-sha256",
            model["weights_sha256"],
            "--tokenizer-dir",
            "artifacts/tokenizers/english-bpe-v1",
            "--device",
            args.device,
            "--output",
            str(args.output_dir / name),
        ]
        print(f"Evaluating {name}", flush=True)
        try:
            with (args.output_dir / f"{name}.log").open("x") as log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1800,
                )
        except Exception as exc:
            write_json(
                args.output_dir / "failure.json",
                {"model": name, "type": type(exc).__name__, "error": str(exc)},
            )
            raise
        summary = read_json(args.output_dir / name / "summary.json")
        print(
            name,
            "BPB",
            summary["matched"]["bits_per_byte"],
            "instructions",
            summary["instructions"]["overall"]["correct"],
            flush=True,
        )
    if freeze != {p.relative_to(ROOT).as_posix(): file_hash(p) for p in freeze_paths}:
        raise ValueError("Frozen evaluation inputs changed during the run")


if __name__ == "__main__":
    main()
