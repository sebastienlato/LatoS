"""Measured assistant-only tuning, fixed base comparison, and validation regressions."""

import argparse
import json
import platform
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch

import latos
from latos.chat import CHAT_CONTRACT, format_chat
from latos.data.manifest import canonical_json
from latos.instruction import load_conversations, prepare_conversations
from latos.model.sampling import generate
from latos.model.storage import bind_tokenizer, file_hash, load_model
from latos.training.checkpoint import runtime_identity, save_checkpoint, source_identity
from latos.training.config import TrainingConfig
from latos.training.data import prepare_dataset
from latos.training.engine import Trainer, evaluate

BASE_SHA256 = "f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa"
TOKENIZER_SHA256 = "7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab"
SAMPLING = {"max_new_tokens": 32, "temperature": 0.0, "top_k": 0, "seed": 71}


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


def instruction_checks(model, codec, records, sequence_length):
    samples, families = [], {}
    for record in records:
        messages = record["messages"]
        ids, _ = format_chat(
            codec, messages[:-1], max_length=sequence_length, generation_prompt=True
        )
        if len(ids) + SAMPLING["max_new_tokens"] > sequence_length:
            raise ValueError("Instruction check must fit the declared evaluation context")
        result = generate(model, list(ids), **SAMPLING)
        text = codec.decode(result["generated_ids"])
        expected = messages[-1]["content"]
        correct = text.strip() == expected.strip()
        scores = families.setdefault(record["family"], {"correct": 0, "cases": 0})
        scores["correct"] += int(correct)
        scores["cases"] += 1
        samples.append(
            {
                "id": record["id"],
                "family": record["family"],
                "messages": messages[:-1],
                "expected": expected,
                "continuation": text,
                "correct": correct,
                "prompt_ids": list(ids),
                **result,
            }
        )
    return {
        "correct": sum(s["correct"] for s in samples),
        "cases": len(samples),
        "eos_stops": sum(s["stop_reason"] == "eos" for s in samples),
        "families": families,
        "samples": samples,
    }


def run(args):
    config = TrainingConfig.load(args.config)
    if args.benchmark:
        config = replace(config, max_steps=10, warmup_steps=1)
    if not 1 <= args.threads <= 64 or args.checkpoint_every < 1 or args.validate_every < 1:
        raise ValueError("Invalid thread count or cadence")
    if file_hash(args.base / "model.safetensors") != args.base_sha256:
        raise ValueError("Selected base weight identity mismatch")
    # A new destination is mandatory, including benchmarks and failed attempts.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(args.threads)
    try:
        model = load_model(args.base, expected_tokenizer_sha256=args.tokenizer_sha256)
        codec = bind_tokenizer(model.config, args.tokenizer_dir)
        train, validation = [
            prepare_conversations(
                args.conversations, codec, args.tokenizer_sha256, config.sequence_length, split
            )
            for split in ("train", "validation")
        ]
        regression = prepare_dataset(
            args.manifest,
            args.corpus_dir,
            codec,
            args.tokenizer_sha256,
            config.sequence_length,
            "validation",
        )
        records = load_conversations(args.conversations, "validation")
        trainer = Trainer(model, config, train, validation, args.device)
        plan = {
            "schema_version": 1,
            "started_utc": datetime.now(UTC).isoformat(),
            "source": source_identity(),
            "runtime": runtime_identity(args.device),
            "runner_sha256": file_hash(Path(__file__)),
            "lock_sha256": file_hash(Path("uv.lock")),
            "config": config.to_dict(),
            "model_config": model.config.to_dict(),
            "base_sha256": args.base_sha256,
            "tokenizer_sha256": args.tokenizer_sha256,
            "chat_contract": CHAT_CONTRACT,
            "datasets": trainer.data_identities,
            "regression": regression.identity,
            "sampling": SAMPLING,
            "selection": "final configured update; no validation selection",
            "optimizer": "fresh AdamW; model weights only from selected base",
            "benchmark_only": args.benchmark,
            "parameter_count": model.parameter_count,
            "checkpoint_every": args.checkpoint_every,
            "validate_every": args.validate_every,
            "target_budget_upper_bound": config.max_steps
            * config.batch_size
            * config.accumulation_steps
            * (config.sequence_length - 1),
            "test_policy": "both instruction and English test payloads never opened",
            "memory_method": (
                "process lifetime peak RSS; MPS boundary snapshots, not continuous peak"
            ),
        }
        write(args.output_dir / "plan.json", plan)
        (args.output_dir / "runner.py").write_bytes(Path(__file__).read_bytes())
        package_root = Path(latos.__file__).resolve().parent
        for source in sorted(package_root.rglob("*.py")):
            saved = args.output_dir / "source" / source.relative_to(package_root)
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(source.read_bytes())
        (args.output_dir / "uv.lock").write_bytes(Path("uv.lock").read_bytes())
        (args.output_dir / "conversation-manifest.json").write_bytes(
            (args.conversations / "manifest.json").read_bytes()
        )
        baseline = None
        if not args.benchmark:
            baseline = {
                "assistant_validation": trainer.validate(),
                "english_validation": evaluate(model, regression, config.batch_size),
                "instructions": instruction_checks(model, codec, records, config.sequence_length),
            }
            write(args.output_dir / "baseline.json", baseline)
            save_checkpoint(trainer, args.output_dir / "step-00000000")
        observations = [memory(args.device)]
        synchronize(args.device)
        loop_started = time.perf_counter()
        update_seconds = 0.0
        with (args.output_dir / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            while trainer.step < config.max_steps:
                synchronize(args.device)
                tick = time.perf_counter()
                metric = trainer.update()
                synchronize(args.device)
                elapsed = time.perf_counter() - tick
                metric["synchronized_seconds"] = elapsed
                update_seconds += elapsed
                if not args.benchmark:
                    if trainer.step % args.validate_every == 0 or trainer.step == config.max_steps:
                        metric["assistant_validation"] = trainer.validate()
                    if (
                        trainer.step % args.checkpoint_every == 0
                        or trainer.step == config.max_steps
                    ):
                        save_checkpoint(trainer, args.output_dir / f"step-{trainer.step:08d}")
                metric["memory"] = memory(args.device)
                observations.append(metric["memory"])
                stream.write(json.dumps(metric, allow_nan=False) + "\n")
                stream.flush()
                if trainer.step % 25 == 0 or trainer.step == config.max_steps:
                    print(json.dumps(metric, allow_nan=False), flush=True)
        synchronize(args.device)
        loop_seconds = time.perf_counter() - loop_started
        result = {
            "status": "ok",
            "steps": trainer.step,
            "tokens_seen": trainer.tokens_seen,
            "windows_seen": trainer.windows_seen,
            "distinct_target_positions_seen": min(trainer.tokens_seen, train.identity["targets"]),
            "corpus_equivalent_target_exposures": trainer.tokens_seen / train.identity["targets"],
            "update_seconds": update_seconds,
            "update_targets_per_second": trainer.tokens_seen / update_seconds,
            "loop_seconds_including_validation_checkpoints": loop_seconds,
            "loop_targets_per_second": trainer.tokens_seen / loop_seconds,
            "failures": [],
        }
        if not args.benchmark:
            result.update(
                {
                    "assistant_validation": trainer.validate(),
                    "english_validation": evaluate(model, regression, config.batch_size),
                    "instructions": instruction_checks(
                        model, codec, records, config.sequence_length
                    ),
                    "selected_checkpoint": f"step-{trainer.step:08d}",
                }
            )
            result["english_loss_delta"] = (
                result["english_validation"]["loss"] - baseline["english_validation"]["loss"]
            )
        if file_hash(args.base / "model.safetensors") != args.base_sha256:
            raise ValueError("Base artifact changed during experiment")
        observations.append(memory(args.device))
        result["memory_max_observed"] = {
            key: max(m[key] for m in observations if m.get(key) is not None)
            for key in observations[-1]
            if observations[-1][key] is not None
        }
        result["total_seconds_before_inventory"] = time.perf_counter() - started
        write(args.output_dir / "summary.json", result)
        write(
            args.output_dir / "artifacts.json",
            {
                p.relative_to(args.output_dir).as_posix(): {
                    "bytes": p.stat().st_size,
                    "sha256": file_hash(p),
                }
                for p in sorted(args.output_dir.rglob("*"))
                if p.is_file()
            },
        )
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
    finally:
        torch.set_num_threads(old_threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("outputs/phase-5-english-pilot/step-00003000/model")
    )
    parser.add_argument("--base-sha256", default=BASE_SHA256)
    parser.add_argument("--tokenizer-sha256", default=TOKENIZER_SHA256)
    parser.add_argument(
        "--tokenizer-dir", type=Path, default=Path("artifacts/tokenizers/english-bpe-v1")
    )
    parser.add_argument(
        "--conversations", type=Path, default=Path("data/processed/english-instructions-v1")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/english-books-v1.json")
    )
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/processed/english-books-v1"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training/instruction-pilot.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--benchmark", action="store_true")
    run(parser.parse_args())
