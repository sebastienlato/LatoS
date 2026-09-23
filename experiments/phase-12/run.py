"""Fixed 256/512 session-budget ablation; no model, tokenizer or prompt changes."""

import argparse
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import torch

from latos.chat import format_chat
from latos.data.manifest import canonical_json
from latos.model.storage import bind_tokenizer, file_hash, load_model
from latos.tool_eval import evaluate, score_case, scripted_control
from latos.tools import ContextLimit, ModelResponder

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "experiments/phase-11"
CONTEXTS = (256, 512)
MAX_SECONDS = 600
MAX_RSS = 8 * 1024**3


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_bytes(canonical_json(value))


def rss_bytes():
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


class BudgetExceeded(RuntimeError):
    """Abort the entire experiment, not merely one conversation."""


class ObservedResponder:
    def __init__(self, respond, codec, context, deadline):
        self.respond, self.codec = respond, codec
        self.context, self.deadline = context, deadline
        self.attempts = []
        self.exhausted = False

    def __call__(self, messages):
        rss = rss_bytes()
        if time.monotonic() > self.deadline or (rss is not None and rss > MAX_RSS):
            self.exhausted = True
            raise BudgetExceeded("Evaluation resource budget exceeded")
        ids, _ = format_chat(self.codec, messages, max_length=4096, generation_prompt=True)
        attempt = {"prompt_tokens": len(ids), "fits": len(ids) + 64 <= self.context}
        self.attempts.append(attempt)
        try:
            reply = self.respond(messages)
        except ContextLimit:
            attempt["outcome"] = "context_limit"
            raise
        attempt.update(outcome="reply", generated_tokens=reply.generated_tokens)
        return reply


def source_files():
    return sorted(
        [*Path("src/latos").rglob("*.py"), *Path("experiments/phase-12").glob("*.py")]
        + [
            Path("experiments/phase-12/PLAN.md"),
            Path("experiments/phase-11/verify.py"),
            Path("experiments/phase-11/plan.json"),
            Path("experiments/phase-11/samples.json"),
            Path("experiments/phase-11/results.json"),
            Path("tests/test_phase12.py"),
            Path("uv.lock"),
            Path("pyproject.toml"),
        ]
    )


def inventory(root):
    """POSIX-relative keys on every host; exclude only this inventory itself."""
    return {
        p.relative_to(root).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file() and p != root / "artifacts.json"
    }


def write_final_inventory(output):
    """Write the final inventory, including each condition inventory."""
    write(output / "artifacts.json", inventory(output))


def run(output, device):
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from the repository root")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    original = read(PREVIOUS / "plan.json")
    samples = read(PREVIOUS / "samples.json")
    source = source_files()
    plan = {
        **original,
        "source_parent": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {p.as_posix(): file_hash(p) for p in source},
        "contexts": list(CONTEXTS),
        "max_seconds": MAX_SECONDS,
        "max_rss_bytes": MAX_RSS,
        "max_generated_tokens": 24576,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "device": device,
        },
    }
    # Capture actual config/metadata identities in addition to the pinned weight hashes.
    plan["input_files"] = {}
    for directory, expected in original["models"].values():
        path = Path(directory)
        if file_hash(path / "model.safetensors") != expected:
            raise ValueError("Preserved model identity mismatch")
        for p in path.iterdir():
            if p.is_file():
                plan["input_files"][p.as_posix()] = file_hash(p)
    tokenizer = Path("artifacts/tokenizers/english-bpe-v1")
    for p in tokenizer.iterdir():
        if p.is_file():
            plan["input_files"][p.as_posix()] = file_hash(p)
    for context in CONTEXTS:
        root = output / str(context)
        root.mkdir()
        for p in source:
            dest = root / "source" / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
        write(root / "plan.json", {**plan, "context_limit": context})
        write(root / "scripted.json", scripted_control())
    write(output / "plan.json", plan)
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    summaries = {str(c): {"scripted_control": scripted_control()["metrics"]} for c in CONTEXTS}
    for label, (directory, _) in original["models"].items():
        model = load_model(Path(directory), expected_tokenizer_sha256=original["tokenizer_sha256"])
        model = model.to(device)
        if model.config.context_length != 512:
            raise ValueError("This experiment requires the preserved 512-capacity model")
        codec = bind_tokenizer(model.config, tokenizer)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        for context in CONTEXTS:
            observed = ObservedResponder(
                ModelResponder(model, codec, context_limit=context), codec, context, deadline
            )
            tick = time.monotonic()
            result = evaluate(observed)
            # run_session handles RuntimeError; promote budget exhaustion to a run failure.
            if observed.exhausted:
                write(
                    output / "budget_failure.json",
                    {
                        "model": label,
                        "context": context,
                        "partial_result": result,
                        "attempts": observed.attempts,
                    },
                )
                raise BudgetExceeded("Evaluation resource budget exceeded")
            result.update(
                seconds=time.monotonic() - tick,
                attempts=observed.attempts,
                weights_unchanged=all(
                    torch.equal(v, before[k]) for k, v in model.state_dict().items()
                ),
                no_gradients=all(p.grad is None for p in model.parameters()),
                generated_tokens=sum(
                    e["generated_tokens"] for s in result["sessions"] for e in s["events"]
                ),
            )
            write(output / str(context) / f"{label}.json", result)
            if not result["weights_unchanged"] or not result["no_gradients"]:
                raise ValueError("Evaluation changed model state")
            if context == 256:
                expected_scores = [
                    score_case(c, s) for c, s in zip(original["cases"], samples[label], strict=True)
                ]
                if result["sessions"] != samples[label] or result["scores"] != expected_scores:
                    raise ValueError(f"Baseline drift: {label}")
                if result["metrics"] != read(PREVIOUS / "results.json")[label]["metrics"]:
                    raise ValueError("Baseline metric drift")
            summaries[str(context)][label] = {
                k: v for k, v in result.items() if k not in ("sessions", "scores", "attempts")
            }
            print(
                label, context, result["metrics"]["stops"], result["generated_tokens"], flush=True
            )
        del model, before, observed
    for context in CONTEXTS:
        root = output / str(context)
        write(root / "results.json", summaries[str(context)])
        write(root / "artifacts.json", inventory(root))
    result = {
        "conditions": summaries,
        "total_seconds": time.monotonic() - started,
        "rss_high_water_bytes": rss_bytes(),
        "training_updates": 0,
        "baseline_sessions_scores_metrics_match": True,
    }
    if device == "mps":
        torch.mps.synchronize()
        result["mps_boundary_allocated_bytes"] = torch.mps.current_allocated_memory()
        result["mps_boundary_driver_bytes"] = torch.mps.driver_allocated_memory()
    if result["total_seconds"] > MAX_SECONDS or (
        result["rss_high_water_bytes"] is not None and result["rss_high_water_bytes"] > MAX_RSS
    ):
        write(output / "budget_failure.json", result)
        raise BudgetExceeded("Final evaluation resource budget exceeded")
    write(output / "results.json", result)
    write_final_inventory(output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    run(args.output_dir, args.device)
