"""SFT experiment integration with a tiny CPU base and unavailable test files."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from latos.data.acquire import acquire
from latos.data.manifest import canonical_json
from latos.data.prepare import prepare
from latos.model import ModelConfig, create_model
from latos.model.storage import file_hash, save_model
from latos.tokenization.training import train
from latos.training.config import TrainingConfig

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(
        f"sft_{name}", ROOT / f"experiments/phase-6/{name}.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


RUN, DATA, VERIFY = module("run"), module("data"), module("verify")


@pytest.fixture
def args(tmp_path):
    manifest = ROOT / "data/fixtures/tiny/manifest.json"
    acquire(manifest, tmp_path / "raw")
    prepare(manifest, tmp_path / "raw", tmp_path / "corpus")
    meta = train(
        manifest, tmp_path / "corpus", ROOT / "configs/tokenizer/debug.json", tmp_path / "codec"
    )
    (tmp_path / "corpus/test.jsonl").unlink()
    model_config = ModelConfig(
        1, meta["vocab_size"], 256, 16, 2, 1, 32, 10000.0, 1e-5, meta["tokenizer_sha256"]
    )
    save_model(create_model(model_config, 17), tmp_path / "base")
    config = TrainingConfig(sequence_length=256, max_steps=2, warmup_steps=0)
    (tmp_path / "config.json").write_bytes(canonical_json(config.to_dict()))
    DATA.materialize(tmp_path / "conversations")
    (tmp_path / "conversations/test.jsonl").unlink()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield Namespace(
        manifest=manifest,
        corpus_dir=tmp_path / "corpus",
        tokenizer_dir=tmp_path / "codec",
        base=tmp_path / "base",
        base_sha256=file_hash(tmp_path / "base/model.safetensors"),
        tokenizer_sha256=meta["tokenizer_sha256"],
        conversations=tmp_path / "conversations",
        config=tmp_path / "config.json",
        output_dir=tmp_path / "run",
        device="cpu",
        threads=1,
        benchmark=False,
        validate_every=1,
        checkpoint_every=1,
    )
    torch.set_num_threads(old_threads)


def test_complete_sft_experiment_and_independent_replay(args):
    result = RUN.run(args)
    assert result["steps"] == 2
    assert result["instructions"]["cases"] == 32
    baseline = json.loads((args.output_dir / "baseline.json").read_text())
    assert baseline["assistant_validation"]["targets"] == result["assistant_validation"]["targets"]
    assert baseline["english_validation"]["targets"] == result["english_validation"]["targets"]
    assert file_hash(args.base / "model.safetensors") == args.base_sha256
    checkpoint = args.output_dir / "step-00000000/state.json"
    state = json.loads(checkpoint.read_text())
    assert state["step"] == state["tokens_seen"] == 0
    assert state["datasets"]["train"]["objective"] == "explicit-assistant-targets-v1"
    verification = VERIFY.verify(
        Namespace(
            **vars(args),
            run_dir=args.output_dir,
            output=args.output_dir.parent / "verification.json",
        )
    )
    assert verification["initial_weights_equal_selected_base"]
    assert verification["recovery"]["max_weight_absolute_difference"] == 0
    assert verification["targets_accounted"] == result["tokens_seen"]
    before = (args.output_dir / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        RUN.run(args)
    assert (args.output_dir / "summary.json").read_bytes() == before


def test_failure_preserves_new_optimizer_checkpoint(args, monkeypatch):
    def fail(self):
        raise RuntimeError("injected SFT update failure")

    monkeypatch.setattr(RUN.Trainer, "update", fail)
    with pytest.raises(RuntimeError, match="injected"):
        RUN.run(args)
    assert (args.output_dir / "step-00000000/state.json").is_file()
    assert not (args.output_dir / "summary.json").exists()
    failure = json.loads((args.output_dir / "failure.json").read_text())
    assert failure["type"] == "RuntimeError"


def test_wrong_base_rejected_before_output_creation(args):
    args.base_sha256 = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        RUN.run(args)
    assert not args.output_dir.exists()
