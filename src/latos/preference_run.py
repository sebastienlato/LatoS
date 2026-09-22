"""Fresh-optimizer DPO experiment, fixed reference and validation-only comparisons."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from latos.adaptation import agreement, instruction_checks, memory, synchronize
from latos.chat import CHAT_CONTRACT
from latos.data.manifest import canonical_json
from latos.doctor import available_backends, select_device
from latos.instruction import load_conversations, prepare_conversations
from latos.model.storage import bind_tokenizer, file_hash, load_model, save_model
from latos.preferences import (
    DPOConfig,
    dpo_loss,
    pair_logps,
    preference_records,
    prepare_preferences,
    score_preferences,
)
from latos.training.checkpoint import runtime_identity, source_identity
from latos.training.data import ShuffleStream, prepare_dataset
from latos.training.engine import evaluate


def run_preferences(args):
    config = DPOConfig.load(args.config)
    if type(args.threads) is not int or not 1 <= args.threads <= 64:
        raise ValueError("Invalid thread count")
    if file_hash(args.base / "model.safetensors") != args.base_sha256:
        raise ValueError("Selected SFT weight identity mismatch")
    device = select_device(args.device, available_backends())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    old_threads, started = torch.get_num_threads(), time.perf_counter()

    def write(name, value):
        (args.output_dir / name).write_bytes(canonical_json(value))

    try:
        torch.set_num_threads(args.threads)
        policy = load_model(args.base).to(device)
        reference = load_model(args.base).to(device).requires_grad_(False).eval()
        if config.sequence_length > policy.config.context_length:
            raise ValueError("DPO sequence length exceeds model capacity")
        codec = bind_tokenizer(policy.config, args.tokenizer_dir)
        token_hash = policy.config.tokenizer_sha256
        records = {s: preference_records(args.conversations, s) for s in ("train", "validation")}
        # Check actual payload identities too, not merely declared manifest groups.
        for key in ("id", "source_group"):
            if {r[key] for r in records["train"]} & {r[key] for r in records["validation"]}:
                raise ValueError("Preference split identity overlap")
        prompts = [{canonical_json(r["prompt"]) for r in records[s]} for s in records]
        if prompts[0] & prompts[1]:
            raise ValueError("Preference prompt overlap")
        datasets = {
            s: prepare_preferences(r, codec, token_hash, config.sequence_length, s)
            for s, r in records.items()
        }
        validation = prepare_conversations(
            args.conversations, codec, token_hash, config.sequence_length, "validation"
        )
        regression = prepare_dataset(
            args.manifest, args.corpus_dir, codec, token_hash, config.sequence_length, "validation"
        )
        conversations = load_conversations(args.conversations, "validation")
        # Keep a CPU snapshot for exact checks without allocating optimizer state for reference.
        frozen = {name: t.detach().cpu().clone() for name, t in reference.state_dict().items()}
        for name, tensor in policy.state_dict().items():
            if not torch.equal(tensor.detach().cpu(), frozen[name]):
                raise ValueError("Policy must start equal to the fixed reference")
        optimizer = torch.optim.AdamW(
            policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        if {id(p) for g in optimizer.param_groups for p in g["params"]} & {
            id(p) for p in reference.parameters()
        }:
            raise ValueError("Reference present in optimizer")
        plan = {
            "schema_version": 1,
            "started_utc": datetime.now(UTC).isoformat(),
            "source": source_identity(),
            "runtime": runtime_identity(device),
            "config": config.to_dict(),
            "model_config": policy.config.to_dict(),
            "base_weights_sha256": args.base_sha256,
            "base_config_sha256": file_hash(args.base / "config.json"),
            "tokenizer_sha256": token_hash,
            "chat_contract": CHAT_CONTRACT,
            "datasets": {s: d.identity for s, d in datasets.items()},
            "assistant_validation": validation.identity,
            "english_validation": regression.identity,
            "selection": "fixed final update; no validation selection or search",
            "sampling": {"max_new_tokens": 32, "temperature": 0, "top_k": 0, "seed": 71},
            "optimizer": "fresh AdamW; model-only output, no DPO optimizer resume",
            "reference": "separate immutable SFT copy; no gradients or optimizer slots",
            "test_policy": "neither reserved test payload opened by the runner",
            "lock_sha256": file_hash(Path("uv.lock")),
            "parameter_count": policy.parameter_count,
        }
        plan["experiment_files"] = {}
        for attribute, name in (("protocol", "protocol.md"), ("runner", "runner.py")):
            path = getattr(args, attribute, None)
            if path is not None:
                (args.output_dir / name).write_bytes(path.read_bytes())
                plan["experiment_files"][name] = file_hash(args.output_dir / name)
        write("plan.json", plan)
        write("preferences.json", records)
        for path in sorted(Path(__file__).parent.rglob("*.py")):
            target = args.output_dir / "source" / path.relative_to(Path(__file__).parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        (args.output_dir / "uv.lock").write_bytes(Path("uv.lock").read_bytes())
        (args.output_dir / "config.json").write_bytes(args.config.read_bytes())

        def score():
            return {
                "preferences": {
                    s: score_preferences(policy, reference, d, config, device)
                    for s, d in datasets.items()
                },
                "assistant_validation": evaluate(policy, validation, config.batch_size),
                "english_validation": evaluate(policy, regression, config.batch_size),
                "instructions": instruction_checks(
                    policy, codec, conversations, config.sequence_length
                ),
            }

        baseline = score()
        write("baseline.json", baseline)
        stream = ShuffleStream(len(records["train"]), config.seed)
        targets_seen, pairs_seen, seconds = 0, 0, 0.0
        observations = [memory(device)]
        with (args.output_dir / "metrics.jsonl").open("x", encoding="utf-8") as log:
            for step in range(1, config.steps + 1):
                synchronize(device)
                tick = time.perf_counter()
                indices = stream.take(config.batch_size)
                policy.train()
                p, targets = pair_logps(policy, datasets["train"], indices, device)
                with torch.no_grad():
                    r, _ = pair_logps(reference, datasets["train"], indices, device)
                loss, _ = dpo_loss(p, r, config.beta)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), config.clip_norm, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                synchronize(device)
                elapsed = time.perf_counter() - tick
                seconds += elapsed
                targets_seen += targets
                pairs_seen += len(indices)
                observations.append(memory(device))
                row = {
                    "step": step,
                    "indices": indices,
                    "epoch": stream.epoch,
                    "loss": loss.item(),
                    "grad_norm": norm.item(),
                    "targets": targets,
                    "targets_seen": targets_seen,
                    "pairs_seen": pairs_seen,
                    "synchronized_seconds": elapsed,
                }
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
        for name, tensor in reference.state_dict().items():
            if not torch.equal(tensor.detach().cpu(), frozen[name]):
                raise ValueError("Reference weights changed")
        if any(p.grad is not None or p.requires_grad for p in reference.parameters()):
            raise ValueError("Reference gradient state changed")
        if not any(
            not torch.equal(t.detach().cpu(), frozen[n]) for n, t in policy.state_dict().items()
        ):
            raise ValueError("DPO did not update policy weights")
        save_model(policy, args.output_dir / "model")
        restored = load_model(args.output_dir / "model").to(device)
        round_trip = agreement(policy, restored, validation, device, tolerance=0.0)
        del restored
        result = {
            "steps": config.steps,
            "pairs_seen": pairs_seen,
            "targets_seen": targets_seen,
            "update_seconds": seconds,
            "pairs_per_update_second": pairs_seen / seconds,
            "frozen_reference_unchanged": True,
            "reference_gradients_and_optimizer_slots": False,
            "policy_updated": True,
            "round_trip": round_trip,
            "training_memory_max_observed": {
                k: max(m[k] for m in observations)
                for k in observations[0]
                if observations[0][k] is not None
            },
            **score(),
        }
        if file_hash(args.base / "model.safetensors") != args.base_sha256:
            raise ValueError("SFT artifact changed")
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
