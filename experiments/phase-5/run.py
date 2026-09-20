"""Measured, single-run English pilot; never opens the reserved test split."""

import argparse
import json
import math
import platform
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch

from latos.data.manifest import canonical_json
from latos.model import ModelConfig, create_model
from latos.model.sampling import generate
from latos.model.storage import bind_tokenizer, file_hash, load_model
from latos.training.checkpoint import runtime_identity, save_checkpoint, source_identity
from latos.training.config import TrainingConfig
from latos.training.data import TokenDataset, prepare_dataset
from latos.training.engine import Trainer, evaluate

PROMPTS = [
    "The morning light fell across the room, and",
    "She opened the letter and discovered",
    "The ocean was quiet until",
    "A careful experiment begins with",
]
SAMPLING = {"max_new_tokens": 64, "temperature": 0.8, "top_k": 40, "seed": 71}


def write(path, value):
    path.write_bytes(canonical_json(value))


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def memory(device):
    result = {}
    try:
        import resource

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["process_peak_rss_bytes"] = raw if platform.system() == "Darwin" else raw * 1024
    except ImportError:
        result["process_peak_rss_bytes"] = None
    if device == "mps":
        result["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
        result["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
    elif device == "cuda":
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    return result


def unigram(train, validation):
    """Add-one train-only frequency baseline, scored on identical held-out targets."""
    counts = Counter(token for window in train.windows for token in window[1:])
    denominator = sum(counts.values()) + train.vocab_size
    total = sum(len(w) - 1 for w in validation.windows)
    loss = (
        sum(
            -math.log((counts[token] + 1) / denominator)
            for window in validation.windows
            for token in window[1:]
        )
        / total
    )
    return {"loss": loss, "perplexity": math.exp(loss), "targets": total, "smoothing": 1}


def samples(model, codec):
    return [
        {
            "prompt": prompt,
            **(result := generate(model, codec.encode(prompt, add_bos=True), **SAMPLING)),
            "continuation": codec.decode(result["generated_ids"]),
        }
        for prompt in PROMPTS
    ]


def run(args):
    config = TrainingConfig.load(args.config)
    model_config = ModelConfig.load(args.model_config)
    if args.device not in ("cpu", "mps", "cuda") or not 1 <= args.threads <= 64:
        raise ValueError("An explicit backend and 1-64 CPU threads are required")
    if args.validate_every < 1 or args.checkpoint_every < 1:
        raise ValueError("Cadences must be positive")
    # Refuse overwrite before touching any artifacts.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "runner.py").write_bytes(Path(__file__).read_bytes())
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    plan = {
        "schema_version": 1,
        "started_utc": datetime.now(UTC).isoformat(),
        "source": source_identity(),
        "runner_sha256": file_hash(Path(__file__)),
        "lock_sha256": file_hash(Path("uv.lock")),
        "runtime": runtime_identity(args.device),
        "config": config.to_dict(),
        "model_config": model_config.to_dict(),
        "benchmark_only": args.benchmark,
        "selection": "last completed configured update; no best-checkpoint selection",
        "validate_every": args.validate_every,
        "checkpoint_every": args.checkpoint_every,
        "prompts": PROMPTS,
        "sampling": SAMPLING,
        "test_split": "reserved; never opened",
        "target_budget_upper_bound": config.max_steps
        * config.batch_size
        * config.accumulation_steps
        * (config.sequence_length - 1),
        "train_diagnostic": "first 128 training windows, same subset before and after",
        "memory_method": (
            "process lifetime peak RSS; accelerator snapshots at update/phase boundaries"
        ),
    }
    write(args.output_dir / "plan.json", plan)
    try:
        codec = bind_tokenizer(model_config, args.tokenizer_dir)
        train, validation = [
            prepare_dataset(
                args.manifest,
                args.corpus_dir,
                codec,
                model_config.tokenizer_sha256,
                config.sequence_length,
                split,
            )
            for split in ("train", "validation")
        ]
        model = create_model(model_config, config.seed)
        if args.initial_snapshot:
            initial = load_model(args.initial_snapshot)
            if initial.config != model_config or any(
                not torch.equal(v, initial.state_dict()[k]) for k, v in model.state_dict().items()
            ):
                raise ValueError("Seeded initialization differs from preserved random snapshot")
            plan["initial_weights_sha256"] = file_hash(args.initial_snapshot / "model.safetensors")
            del initial
        trainer = Trainer(model, config, train, validation, args.device)
        diagnostic = TokenDataset(
            train.windows[:128],
            train.sequence_length,
            train.vocab_size,
            train.tokenizer_sha256,
            "train",
            train.source_sha256,
        )
        plan["datasets"] = trainer.data_identities
        plan["diagnostic_identity"] = diagnostic.identity
        plan["parameter_count"] = model.parameter_count
        write(args.output_dir / "plan.json", plan)
        measurements = [memory(args.device)]
        if not args.benchmark:
            baseline = {
                "validation": trainer.validate(),
                "train_diagnostic": evaluate(model, diagnostic, config.batch_size),
                "unigram_validation": unigram(train, validation),
                "samples": samples(model, codec),
            }
            write(args.output_dir / "baseline.json", baseline)
            save_checkpoint(trainer, args.output_dir / "step-00000000")
            measurements.append(memory(args.device))
        synchronize(args.device)
        training_started = time.perf_counter()
        update_seconds, tokens = 0.0, 0
        with (args.output_dir / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            while trainer.step < config.max_steps:
                synchronize(args.device)
                update_started = time.perf_counter()
                metric = trainer.update()
                synchronize(args.device)
                elapsed = time.perf_counter() - update_started
                metric["synchronized_update_seconds"] = elapsed
                metric["synchronized_targets_per_second"] = metric["targets"] / elapsed
                update_seconds += elapsed
                tokens += metric["targets"]
                if not args.benchmark:
                    if trainer.step % args.validate_every == 0 or trainer.step == config.max_steps:
                        metric["validation"] = trainer.validate()
                    if (
                        trainer.step % args.checkpoint_every == 0
                        or trainer.step == config.max_steps
                    ):
                        save_checkpoint(trainer, args.output_dir / f"step-{trainer.step:08d}")
                metric["memory"] = memory(args.device)
                measurements.append(metric["memory"])
                stream.write(json.dumps(metric, allow_nan=False) + "\n")
                stream.flush()
                if trainer.step % 25 == 0 or trainer.step == config.max_steps:
                    print(json.dumps(metric, allow_nan=False), flush=True)
        synchronize(args.device)
        training_wall = time.perf_counter() - training_started
        result = {
            "status": "ok",
            "steps": trainer.step,
            "tokens_seen": tokens,
            "windows_seen": trainer.windows_seen,
            "corpus_equivalent_exposures": tokens / plan["datasets"]["train"]["targets"],
            "distinct_target_positions_seen": min(tokens, plan["datasets"]["train"]["targets"]),
            "distinct_windows_seen": min(trainer.windows_seen, len(train.windows)),
            "update_seconds": update_seconds,
            "update_targets_per_second": tokens / update_seconds,
            "training_wall_seconds_including_validation_and_checkpoints": training_wall,
            "training_wall_targets_per_second": tokens / training_wall,
            "failures": [],
        }
        if not args.benchmark:
            result["final_validation"] = trainer.validate()
            result["final_train_diagnostic"] = evaluate(model, diagnostic, config.batch_size)
            result["samples"] = samples(model, codec)
            result["selected_checkpoint"] = f"step-{trainer.step:08d}"
        synchronize(args.device)
        measurements.append(memory(args.device))
        result["memory_max_observed"] = {
            k: max(m[k] for m in measurements if m.get(k) is not None)
            for k in measurements[-1]
            if measurements[-1][k] is not None
        }
        result["total_wall_seconds_before_inventory"] = time.perf_counter() - started
        write(args.output_dir / "summary.json", result)
        inventory = {
            p.relative_to(args.output_dir).as_posix(): {
                "sha256": file_hash(p),
                "bytes": p.stat().st_size,
            }
            for p in sorted(args.output_dir.rglob("*"))
            if p.is_file()
        }
        write(args.output_dir / "artifacts.json", inventory)
        return result
    except BaseException as error:
        write(
            args.output_dir / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/english-books-v1.json")
    )
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/processed/english-books-v1"))
    parser.add_argument(
        "--tokenizer-dir", type=Path, default=Path("artifacts/tokenizers/english-bpe-v1")
    )
    parser.add_argument("--model-config", type=Path, default=Path("configs/model/pilot.json"))
    parser.add_argument("--initial-snapshot", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--benchmark", action="store_true")
    run(parser.parse_args())
