"""Offline Phase 4 CPU acceptance; all generated data/weights go to a fresh directory."""

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import torch

from latos.data.acquire import acquire
from latos.data.manifest import canonical_json
from latos.data.prepare import prepare
from latos.doctor import diagnose
from latos.model import ModelConfig, create_model
from latos.model.storage import file_hash
from latos.tokenization import LatoTokenizer
from latos.tokenization.training import train
from latos.training import Trainer, TrainingConfig, evaluate
from latos.training.checkpoint import (
    load_checkpoint,
    runtime_identity,
    save_checkpoint,
    source_identity,
)
from latos.training.data import prepare_dataset


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal(a, b)
    else:
        assert left == right


def peak_rss_bytes():
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if platform.system() == "Darwin" else 1024
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(1)
    config = TrainingConfig.load(root / "configs/training/fixture.json")
    manifest = root / "data/fixtures/tiny/manifest.json"
    started = time.perf_counter()
    acquire(manifest, output / "raw")
    prepare(manifest, output / "raw", output / "corpus")
    metadata = train(
        manifest, output / "corpus", root / "configs/tokenizer/debug.json", output / "tokenizer"
    )
    codec = LatoTokenizer.load(output / "tokenizer")
    model_config = ModelConfig(
        1, codec.vocab_size, 64, 64, 4, 2, 192, 10000.0, 1e-5, metadata["tokenizer_sha256"]
    )
    (output / "model-config.json").write_bytes(canonical_json(model_config.to_dict()))
    # Absence of the test split is deliberate evidence that training never reads it.
    (output / "corpus/test.jsonl").unlink()
    datasets = [
        prepare_dataset(
            manifest,
            output / "corpus",
            codec,
            metadata["tokenizer_sha256"],
            config.sequence_length,
            split,
        )
        for split in ("train", "validation")
    ]
    trainer = Trainer(create_model(model_config, config.seed), config, *datasets)
    initial = {"train": evaluate(trainer.model, datasets[0]), "validation": trainer.validate()}
    cut = 97
    metrics = []
    with (output / "metrics.jsonl").open("x", encoding="utf-8") as stream:
        for _ in range(config.max_steps):
            metric = trainer.update()
            metrics.append(metric)
            stream.write(json.dumps(metric) + "\n")
            if trainer.step == cut:
                save_checkpoint(trainer, output / "interrupted")
    state = copy.deepcopy(trainer.model.state_dict())
    optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    sampler_rng = trainer.stream.generator.get_state().clone()
    final = {"train": evaluate(trainer.model, datasets[0]), "validation": trainer.validate()}
    assert_equal(state, trainer.model.state_dict())
    assert_equal(optimizer, trainer.optimizer.state_dict())
    assert_equal(rng, torch.get_rng_state())
    assert_equal(sampler_rng, trainer.stream.generator.get_state())
    assert all(p.grad is None for p in trainer.model.parameters())
    assert trainer.model.training
    # Fixed before executing the run; held-out quality is not an acceptance criterion.
    assert final["train"]["loss"] < 0.25
    assert final["train"]["loss"] < initial["train"]["loss"] * 0.1
    final_metadata = save_checkpoint(trainer, output / "final")
    resumed = load_checkpoint(output / "interrupted", *datasets)
    for expected in metrics[cut:]:
        actual = resumed.update()
        for key in expected.keys() - {"seconds", "targets_per_second"}:
            assert expected[key] == actual[key], key
    assert_equal(trainer.model.state_dict(), resumed.model.state_dict())
    assert_equal(trainer.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_equal(trainer.stream.generator.get_state(), resumed.stream.generator.get_state())
    assert_equal(trainer.stream.order, resumed.stream.order)
    assert trainer.stream.cursor == resumed.stream.cursor
    summary = {
        "schema_version": 1,
        "status": "pass",
        "source": source_identity(),
        "peak_process_rss_bytes": peak_rss_bytes(),
        "runtime": runtime_identity("cpu"),
        "environment": diagnose("cpu"),
        "config": config.to_dict(),
        "model_config": model_config.to_dict(),
        "parameter_count": trainer.model.parameter_count,
        "datasets": trainer.data_identities,
        "initial": initial,
        "final": final,
        "tokens_seen": trainer.tokens_seen,
        "windows_seen": trainer.windows_seen,
        "elapsed_seconds_including_resume": time.perf_counter() - started,
        "training_seconds": sum(m["seconds"] for m in metrics),
        "training_targets_per_second": trainer.tokens_seen / sum(m["seconds"] for m in metrics),
        "overfit_criteria": "train loss < 0.25 and < 10% of initial after exactly 400 updates",
        "resume": {
            "cut_step": cut,
            "final_step": trainer.step,
            "compared_updates": config.max_steps - cut,
            "weights_bitwise_equal": True,
            "optimizer_bitwise_equal": True,
            "sampler_bitwise_equal": True,
            "non_timing_metrics_equal": True,
        },
        "validation": {
            "weights_unchanged": True,
            "optimizer_unchanged": True,
            "rng_unchanged": True,
            "gradients_unchanged": True,
            "mode_restored": True,
        },
        "test_split": "removed before training; never read by the engine",
        "final_model_metadata_sha256": final_metadata["model_metadata_sha256"],
        "final_model_tensor_sha256": file_hash(output / "final/model/model.safetensors"),
        "final_training_tensor_sha256": final_metadata["training_sha256"],
        "quality_limit": "Fixture memorization only; no language quality claim",
    }
    (output / "acceptance.json").write_bytes(canonical_json(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
