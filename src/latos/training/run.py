"""CLI orchestration with immutable step checkpoints and segment metrics."""

import json

import torch

from latos.data.manifest import canonical_json
from latos.model import ModelConfig, create_model
from latos.model.storage import bind_tokenizer
from latos.training.checkpoint import (
    load_checkpoint,
    runtime_identity,
    save_checkpoint,
    source_identity,
)
from latos.training.config import TrainingConfig
from latos.training.data import prepare_dataset
from latos.training.engine import Trainer


def run_training(args) -> dict:
    if args.output_dir.exists():
        raise ValueError("Training output already exists; choose a new destination")
    for name in ("threads", "validate_every", "checkpoint_every"):
        if not 1 <= getattr(args, name) <= 1_000_000:
            raise ValueError(f"{name} must be positive and bounded")
    if args.threads > 64:
        raise ValueError("CPU threads exceed the limit of 64")
    config = TrainingConfig.load(args.config)
    if args.resume:
        model_config = ModelConfig.load(args.resume / "model/config.json")
    else:
        model_config = ModelConfig.load(args.model_config)
    codec = bind_tokenizer(model_config, args.tokenizer_dir)
    datasets = [
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
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(args.threads)
        trainer = (
            load_checkpoint(args.resume, *datasets, args.device, expected_config=config)
            if args.resume
            else Trainer(create_model(model_config, config.seed), config, *datasets, args.device)
        )
        stop = args.stop_after if args.stop_after is not None else config.max_steps
        if not trainer.step < stop <= config.max_steps:
            raise ValueError("stop-after must exceed the saved step and not exceed max_steps")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report = {
            "schema_version": 1,
            "runtime": runtime_identity(args.device),
            "source": source_identity(),
            "config": config.to_dict(),
            "model_config": model_config.to_dict(),
            "datasets": trainer.data_identities,
            "start_step": trainer.step,
            "initial_validation": trainer.validate(),
            "selection": "last completed update; no best-checkpoint selection",
        }
        (args.output_dir / "run.json").write_bytes(canonical_json(report))
        # An initial recovery point exists even if the first update fails.
        last = args.output_dir / f"step-{trainer.step:08d}"
        save_checkpoint(trainer, last)
        with (args.output_dir / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            while trainer.step < stop:
                metric = trainer.update()
                if trainer.step % args.validate_every == 0 or trainer.step == stop:
                    metric["validation"] = trainer.validate()
                if trainer.step % args.checkpoint_every == 0 or trainer.step == stop:
                    last = args.output_dir / f"step-{trainer.step:08d}"
                    save_checkpoint(trainer, last)
                stream.write(json.dumps(metric, allow_nan=False) + "\n")
                stream.flush()
        report.update(
            {
                "status": "ok",
                "end_step": trainer.step,
                "tokens_seen": trainer.tokens_seen,
                "final_validation": trainer.validate(),
                "checkpoint": last.name,
            }
        )
        (args.output_dir / "summary.json").write_bytes(canonical_json(report))
        return report
    finally:
        torch.set_num_threads(old_threads)
