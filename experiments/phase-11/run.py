"""Run fixed unchanged-model tool evaluation, retaining source and every response locally."""

import argparse
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import torch

from latos.data.manifest import canonical_json
from latos.model.storage import bind_tokenizer, file_hash, load_model
from latos.tool_eval import development_cases, evaluate, scripted_control
from latos.tools import PROTOCOL, SYSTEM_PROMPT, ModelResponder

MODELS = {
    "base": (
        "outputs/phase-5-english-pilot/step-00003000/model",
        "f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa",
    ),
    "sft": (
        "outputs/phase-6-instruction-reviewed/step-00000200/model",
        "62b890214e5544292cc54b725a4e505730fca0d501b3337cdd3f051a622d09ae",
    ),
    "dpo": (
        "outputs/phase-10-dpo-reviewed/model",
        "5ec902472e6e04ca5c97475d44a30818729a2fc5e8592544642767af83ae9ca1",
    ),
}
TOKENIZER = "7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab"


def write(path, value):
    path.write_bytes(canonical_json(value))


def run(output: Path, device: str):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    torch.set_num_threads(1)
    source = [
        *Path("src/latos").rglob("*.py"),
        Path("uv.lock"),
        *Path(__file__).parent.glob("*.py"),
        Path("pyproject.toml"),
        Path("tests/test_tools.py"),
        Path(__file__).with_name("PLAN.md"),
    ]
    identities = {}
    for path in source:
        destination = (
            output / "source" / path.relative_to(Path.cwd())
            if path.is_absolute()
            else (output / "source" / path)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        identities[str(path.relative_to(Path.cwd()) if path.is_absolute() else path)] = file_hash(
            path
        )
    write(
        output / "plan.json",
        {
            "source_parent": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_sha256": identities,
            "protocol": PROTOCOL,
            "system_prompt": SYSTEM_PROMPT,
            "cases": development_cases(),
            "models": MODELS,
            "tokenizer_sha256": TOKENIZER,
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "platform": platform.platform(),
                "device": device,
            },
            "context_limit": 256,
            "max_new_tokens": 64,
            "temperature": 0,
            "seed": 0,
            "training_updates": 0,
            "threads": 1,
        },
    )
    control = scripted_control()
    write(output / "scripted.json", control)
    results = {}
    for label, (directory, expected) in MODELS.items():
        path = Path(directory)
        if file_hash(path / "model.safetensors") != expected:
            raise ValueError("Preserved model identity mismatch")
        model = load_model(path, expected_tokenizer_sha256=TOKENIZER).to(device)
        codec = bind_tokenizer(model.config, Path("artifacts/tokenizers/english-bpe-v1"))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        tick = time.perf_counter()
        result = evaluate(ModelResponder(model, codec))
        result["seconds"] = time.perf_counter() - tick
        result["weights_unchanged"] = all(
            torch.equal(value, before[name]) for name, value in model.state_dict().items()
        )
        result["no_gradients"] = all(p.grad is None for p in model.parameters())
        if not result["weights_unchanged"] or not result["no_gradients"]:
            raise ValueError("Evaluation changed model state")
        result["generated_tokens"] = sum(
            e["generated_tokens"] for s in result["sessions"] for e in s["events"]
        )
        write(output / f"{label}.json", result)
        results[label] = {k: v for k, v in result.items() if k not in ("sessions", "scores")}
        print(label, json.dumps(result["metrics"]), flush=True)
        del model, before
    results["scripted_control"] = control["metrics"]
    results["total_seconds"] = time.perf_counter() - start
    try:
        import resource
    except ImportError:
        results["rss_high_water_native_units"] = None
    else:
        results["rss_high_water_native_units"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    results["rss_units"] = "bytes on macOS; KiB on Linux"
    if device == "mps":
        results["mps_boundary_allocated_bytes"] = torch.mps.current_allocated_memory()
        results["mps_boundary_driver_bytes"] = torch.mps.driver_allocated_memory()
    write(output / "results.json", results)
    write(
        output / "artifacts.json",
        {
            str(p.relative_to(output)): {"sha256": file_hash(p), "bytes": p.stat().st_size}
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    run(args.output_dir, args.device)
