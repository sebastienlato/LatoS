"""Bounded fresh-optimizer adaptation runs with comparable validation-only evidence."""

import json
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from latos.chat import CHAT_CONTRACT, format_chat
from latos.data.manifest import canonical_json, sha256
from latos.instruction import load_conversations, prepare_conversations
from latos.lora import LoRAConfig, LoRAModel, load_adapter, merge_adapter, save_adapter
from latos.model.sampling import generate
from latos.model.storage import bind_tokenizer, file_hash, load_model, save_model
from latos.training.checkpoint import runtime_identity, source_identity
from latos.training.config import TrainingConfig
from latos.training.data import collate, prepare_dataset
from latos.training.engine import Trainer, evaluate


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def memory(device):
    """Process lifetime peak RSS; accelerator boundary snapshots, not independent pools."""
    result = {}
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["process_peak_rss_bytes"] = rss if platform.system() == "Darwin" else rss * 1024
    except ImportError:
        result["process_peak_rss_bytes"] = None
    if device == "mps":
        result["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
        result["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
    return result


def instruction_checks(model, codec, records, sequence_length):
    samples = []
    for record in records:
        messages = record["messages"]
        ids, _ = format_chat(
            codec, messages[:-1], generation_prompt=True, max_length=sequence_length
        )
        if len(ids) + 32 > sequence_length:
            raise ValueError("Instruction generation exceeds evaluation context")
        generated = generate(model, list(ids), max_new_tokens=32, temperature=0.0, top_k=0, seed=71)
        text = codec.decode(generated["generated_ids"])
        samples.append(
            {
                "id": record["id"],
                "family": record["family"],
                "prompt_ids": list(ids),
                "expected": messages[-1]["content"],
                "continuation": text,
                "correct": text.strip() == messages[-1]["content"].strip(),
                **generated,
            }
        )
    return {
        "correct": sum(s["correct"] for s in samples),
        "cases": len(samples),
        "eos_stops": sum(s["stop_reason"] == "eos" for s in samples),
        "samples": samples,
    }


@torch.inference_mode()
def agreement(left, right, dataset, device, *, tolerance):
    """Compare all vocabulary logits at every validation position, including padding."""
    left.eval()
    right.eval()
    maximum = 0.0
    for index in range(len(dataset.windows)):
        ids, _, _ = collate(dataset, [index], device)
        a, b = left(ids).logits, right(ids).logits
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
        maximum = max(maximum, (a - b).abs().max().item())
    return {
        "max_absolute_difference": maximum,
        "atol": tolerance,
        "rtol": tolerance,
        "windows": len(dataset.windows),
    }


def run_adaptation(args):
    config = TrainingConfig.load(args.config)
    lora_config = LoRAConfig(args.rank, args.alpha, args.adapter_seed)
    if args.method not in ("lora", "full") or not 1 <= args.threads <= 64:
        raise ValueError("Invalid method or thread count")
    if file_hash(args.base / "model.safetensors") != args.base_sha256:
        raise ValueError("Selected base weight identity mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    old_threads = torch.get_num_threads()
    started = time.perf_counter()

    def write(name, value):
        (args.output_dir / name).write_bytes(canonical_json(value))

    try:
        torch.set_num_threads(args.threads)
        base = load_model(args.base)
        codec = bind_tokenizer(base.config, args.tokenizer_dir)
        token_hash = base.config.tokenizer_sha256
        train, validation = [
            prepare_conversations(
                args.conversations, codec, token_hash, config.sequence_length, split
            )
            for split in ("train", "validation")
        ]
        regression = prepare_dataset(
            args.manifest, args.corpus_dir, codec, token_hash, config.sequence_length, "validation"
        )
        records = load_conversations(args.conversations, "validation")
        model = LoRAModel(base, lora_config) if args.method == "lora" else base
        # Release the caller's base copy; LoRAModel owns independent frozen storage.
        del base
        trainer = Trainer(model, config, train, validation, args.device)
        device = trainer.device
        plan = {
            "schema_version": 1,
            "started_utc": datetime.now(UTC).isoformat(),
            "source": source_identity(),
            "runtime": runtime_identity(device),
            "method": args.method,
            "config": config.to_dict(),
            "lora": asdict(lora_config),
            "model_config": model.config.to_dict(),
            "base_weights_sha256": args.base_sha256,
            "tokenizer_sha256": token_hash,
            "chat_contract": CHAT_CONTRACT,
            "datasets": trainer.data_identities,
            "regression": regression.identity,
            "selection": "fixed final update; no validation selection or hyperparameter search",
            "sampling": {"max_new_tokens": 32, "temperature": 0.0, "top_k": 0, "seed": 71},
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "base_parameters": model.config.parameter_count,
            "test_policy": "neither reserved test payload is opened",
            "optimizer": "fresh AdamW; no optimizer resume or historical replay",
            "lock_sha256": file_hash(Path("uv.lock")),
            "memory_method": "process lifetime RSS; MPS boundary snapshots, not continuous peaks",
        }
        write("plan.json", plan)
        for path in sorted(Path(__file__).parent.rglob("*.py")):
            target = args.output_dir / "source" / path.relative_to(Path(__file__).parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        (args.output_dir / "uv.lock").write_bytes(Path("uv.lock").read_bytes())

        def score():
            return {
                "assistant_validation": trainer.validate(),
                "english_validation": evaluate(model, regression, config.batch_size),
                "instructions": instruction_checks(model, codec, records, config.sequence_length),
            }

        write("baseline.json", score())
        observed = [memory(device)]
        seconds = 0.0
        with (args.output_dir / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            while trainer.step < config.max_steps:
                synchronize(device)
                tick = time.perf_counter()
                metric = trainer.update()
                synchronize(device)
                metric["synchronized_seconds"] = time.perf_counter() - tick
                seconds += metric["synchronized_seconds"]
                metric["sampler_sha256"] = sha256(
                    canonical_json(
                        {
                            "order": trainer.stream.order.tolist(),
                            "epoch": trainer.stream.epoch,
                            "cursor": trainer.stream.cursor,
                        }
                    )
                )
                metric["memory"] = memory(device)
                observed.append(metric["memory"])
                stream.write(json.dumps(metric, allow_nan=False) + "\n")
                stream.flush()
        result = {
            "steps": trainer.step,
            "tokens_seen": trainer.tokens_seen,
            "windows_seen": trainer.windows_seen,
            "update_seconds": seconds,
            "update_targets_per_second": trainer.tokens_seen / seconds,
            "trainable_parameters": plan["trainable_parameters"],
            "optimizer_tensor_bytes": sum(
                t.numel() * t.element_size()
                for slots in trainer.optimizer.state.values()
                for t in slots.values()
                if isinstance(t, torch.Tensor)
            ),
            "training_memory_max_observed": {
                key: max(m[key] for m in observed)
                for key in observed[0]
                if observed[0][key] is not None
            },
        }
        if args.method == "lora":
            model.validate_adapter()
            result["frozen_base_unchanged"] = True
            save_adapter(model, args.output_dir / "adapter")
            # Reload the exact base from disk, bind adapter, then compare on the same backend.
            restored = load_adapter(load_model(args.base), args.output_dir / "adapter").to(device)
            result["round_trip"] = agreement(model, restored, validation, device, tolerance=0.0)
            merged = merge_adapter(restored)
            save_model(merged, args.output_dir / "merged")
            merged = load_model(args.output_dir / "merged").to(device)
            result["merge"] = agreement(model, merged, validation, device, tolerance=1e-4)
            del restored, merged
        else:
            save_model(model, args.output_dir / "model")
        result.update(score())
        if file_hash(args.base / "model.safetensors") != args.base_sha256:
            raise ValueError("Selected base artifact changed")
        result["total_seconds_before_inventory"] = time.perf_counter() - started
        write("summary.json", result)
        write(
            "artifacts.json",
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
            "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
    finally:
        torch.set_num_threads(old_threads)
