"""Pilot evidence must remain usable without test text and after execution failure."""

import importlib.util
import json
import math
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from latos.data.acquire import acquire
from latos.data.manifest import canonical_json
from latos.data.prepare import prepare
from latos.model import ModelConfig
from latos.model.storage import file_hash
from latos.tokenization.training import train
from latos.training.config import TrainingConfig
from latos.training.data import TokenDataset

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pilot", ROOT / "experiments/phase-5/run.py")
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def test_unigram_uses_only_training_counts_and_scores_each_target():
    training = TokenDataset(((1, 4, 4, 2),), 16, 260, "a" * 64, "train", "b" * 64)
    validation = replace(training, windows=((1, 4, 5, 2),), split="validation")
    result = PILOT.unigram(training, validation)
    expected = -(math.log(3 / 263) + math.log(1 / 263) + math.log(2 / 263)) / 3
    assert result["targets"] == 3
    assert result["loss"] == pytest.approx(expected)
    assert result["perplexity"] == pytest.approx(math.exp(expected))


@pytest.fixture
def pilot_args(tmp_path):
    manifest = ROOT / "data/fixtures/tiny/manifest.json"
    acquire(manifest, tmp_path / "raw")
    prepare(manifest, tmp_path / "raw", tmp_path / "corpus")
    metadata = train(
        manifest, tmp_path / "corpus", ROOT / "configs/tokenizer/debug.json", tmp_path / "codec"
    )
    # Both baseline evaluation and execution must succeed when test text is absent.
    (tmp_path / "corpus/test.jsonl").unlink()
    model = ModelConfig(
        1,
        metadata["vocab_size"],
        128,
        16,
        2,
        1,
        32,
        10000.0,
        1e-5,
        metadata["tokenizer_sha256"],
    )
    (tmp_path / "model.json").write_bytes(canonical_json(model.to_dict()))
    config = TrainingConfig(sequence_length=32, max_steps=2, warmup_steps=0)
    (tmp_path / "config.json").write_bytes(canonical_json(config.to_dict()))
    old = torch.get_num_threads()
    yield Namespace(
        manifest=manifest,
        corpus_dir=tmp_path / "corpus",
        tokenizer_dir=tmp_path / "codec",
        model_config=tmp_path / "model.json",
        config=tmp_path / "config.json",
        output_dir=tmp_path / "run",
        device="cpu",
        threads=1,
        benchmark=False,
        initial_snapshot=None,
        validate_every=1,
        checkpoint_every=1,
    )
    torch.set_num_threads(old)


def test_pilot_records_baselines_checkpoints_exposure_and_refuses_overwrite(pilot_args):
    result = PILOT.run(pilot_args)
    out = pilot_args.output_dir
    baseline = json.loads((out / "baseline.json").read_text())
    metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert baseline["validation"]["targets"] == result["final_validation"]["targets"]
    assert baseline["unigram_validation"]["targets"] == result["final_validation"]["targets"]
    assert result["tokens_seen"] == sum(m["targets"] for m in metrics)
    assert result["selected_checkpoint"] == "step-00000002"
    assert [s["prompt"] for s in baseline["samples"]] == [s["prompt"] for s in result["samples"]]
    assert (out / "step-00000000/state.json").exists()
    assert (out / "step-00000002/training.safetensors").exists()
    inventory = json.loads((out / "artifacts.json").read_text())
    assert all(file_hash(out / p) == entry["sha256"] for p, entry in inventory.items())
    before = (out / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        PILOT.run(pilot_args)
    assert (out / "summary.json").read_bytes() == before


def test_failed_update_preserves_initial_recovery_point(pilot_args, monkeypatch):
    def fail(self):
        raise RuntimeError("injected update failure")

    monkeypatch.setattr(PILOT.Trainer, "update", fail)
    with pytest.raises(RuntimeError, match="injected update failure"):
        PILOT.run(pilot_args)
    out = pilot_args.output_dir
    assert (out / "step-00000000/state.json").exists()
    assert json.loads((out / "failure.json").read_text())["message"] == "injected update failure"
    assert not (out / "summary.json").exists()
