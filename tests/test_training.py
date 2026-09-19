"""Numerical training contracts and recovery across batches, epochs, and processes."""

import copy
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from latos.data.acquire import acquire
from latos.data.manifest import canonical_json
from latos.data.prepare import prepare
from latos.model import ModelConfig, create_model
from latos.model.storage import file_hash
from latos.tokenization import LatoTokenizer
from latos.tokenization.training import train
from latos.training import Trainer, TrainingConfig, evaluate
from latos.training.checkpoint import load_checkpoint, save_checkpoint
from latos.training.data import TokenDataset, collate, prepare_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def model_config():
    return ModelConfig(1, 260, 16, 16, 2, 1, 32, 10000.0, 1e-5, "a" * 64)


@pytest.fixture
def datasets():
    rows = ((1, 8, 9, 10, 2), (1, 11, 2), (1, 12, 13, 14, 15, 2), (1, 16, 2), (1, 17, 18, 2))
    training = TokenDataset(rows, 16, 260, "a" * 64, "train", "b" * 64)
    return training, replace(
        training, windows=((1, 19, 20, 2), (1, 21, 2)), split="validation", source_sha256="c" * 64
    )


def make_trainer(model_config, datasets, **kwargs):
    return Trainer(
        create_model(model_config, 17),
        TrainingConfig(sequence_length=16, max_steps=12, warmup_steps=2, **kwargs),
        *datasets,
    )


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_tree(a, b)
    else:
        assert left == right


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_steps", True),
        ("batch_size", 0),
        ("warmup_steps", 100),
        ("learning_rate", float("nan")),
        ("beta2", 1.0),
        ("seed", -1),
        ("sequence_length", 1),
        ("accumulation_steps", 0),
    ],
)
def test_config_rejects_invalid(field, value):
    with pytest.raises(ValueError):
        TrainingConfig(**{field: value})


def test_schedule_endpoints():
    config = TrainingConfig(max_steps=10, warmup_steps=2, learning_rate=0.01, min_lr_ratio=0.1)
    assert config.learning_rate_at(1) == 0.005
    assert config.learning_rate_at(2) == 0.01
    assert config.learning_rate_at(10) == pytest.approx(0.001)
    assert all(config.learning_rate_at(i) > config.learning_rate_at(i + 1) for i in range(2, 10))
    assert replace(config, warmup_steps=0).learning_rate_at(1) == 0.01
    assert replace(config, warmup_steps=0, max_steps=1).learning_rate_at(1) == 0.01
    with pytest.raises(ValueError):
        config.learning_rate_at(0)


def test_padding_and_alignment(model_config, datasets):
    ids, labels, count = collate(datasets[0], [0, 1], "cpu")
    assert count == 6
    assert ids.tolist() == [[1, 8, 9, 10, 2], [1, 11, 2, 0, 0]]
    assert labels.tolist() == [[-100, 8, 9, 10, 2], [-100, 11, 2, -100, -100]]
    model = create_model(model_config)
    batch_loss = model(ids, labels).loss
    individual = (
        sum(
            model(torch.tensor([row]), torch.tensor([row])).loss * (len(row) - 1)
            for row in datasets[0].windows[:2]
        )
        / count
    )
    torch.testing.assert_close(batch_loss, individual)
    ids[1, 3:] = 50
    torch.testing.assert_close(model(ids, labels).loss, batch_loss)


def test_accumulation_matches_token_weighted_large_batch(model_config, datasets):
    # Unequal lengths exercise token weighting; clipping is active in both paths.
    accumulated = make_trainer(
        model_config, datasets, batch_size=1, accumulation_steps=3, max_grad_norm=0.1
    )
    combined = make_trainer(
        model_config, datasets, batch_size=3, accumulation_steps=1, max_grad_norm=0.1
    )
    a, b = accumulated.update(), combined.update()
    assert a["targets"] == b["targets"]
    assert a["loss"] == pytest.approx(b["loss"], abs=1e-6)
    assert a["grad_norm_before_clip"] > 0.1
    for x, y in zip(accumulated.model.parameters(), combined.model.parameters(), strict=True):
        torch.testing.assert_close(x, y, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("cut", [0, 1, 3, 7, 12])
def test_resume_exact_cpu(model_config, datasets, tmp_path, cut):
    reference = make_trainer(model_config, datasets)
    uninterrupted = [reference.update() for _ in range(12)]
    interrupted = make_trainer(model_config, datasets)
    for _ in range(cut):
        interrupted.update()
    save_checkpoint(interrupted, tmp_path / "saved")
    restored = load_checkpoint(tmp_path / "saved", *datasets)
    for expected in uninterrupted[cut:]:
        actual = restored.update()
        for key in expected.keys() - {"seconds", "targets_per_second"}:
            assert expected[key] == actual[key]
    equal_tree(reference.model.state_dict(), restored.model.state_dict())
    equal_tree(reference.optimizer.state_dict(), restored.optimizer.state_dict())
    assert reference.tokens_seen == restored.tokens_seen
    equal_tree(reference.stream.generator.get_state(), restored.stream.generator.get_state())
    equal_tree(reference.stream.order, restored.stream.order)
    assert reference.stream.cursor == restored.stream.cursor
    with pytest.raises(ValueError, match="complete"):
        restored.update()


def test_validation_preserves_all_state(model_config, datasets):
    trainer = make_trainer(model_config, datasets)
    trainer.update()
    trainer.model.blocks[0].eval()  # Preserve heterogeneous module modes as well.
    for p in trainer.model.parameters():
        p.grad = torch.ones_like(p)
    weights = copy.deepcopy(trainer.model.state_dict())
    optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    gradients = [p.grad.clone() for p in trainer.model.parameters()]
    modes = [m.training for m in trainer.model.modules()]
    rng, shuffle = torch.get_rng_state().clone(), trainer.stream.generator.get_state().clone()
    metrics = trainer.validate()
    assert metrics["targets"] == 5 and metrics["loss"] > 0
    equal_tree(weights, trainer.model.state_dict())
    equal_tree(optimizer, trainer.optimizer.state_dict())
    equal_tree(gradients, [p.grad for p in trainer.model.parameters()])
    equal_tree(rng, torch.get_rng_state())
    equal_tree(shuffle, trainer.stream.generator.get_state())
    assert modes == [m.training for m in trainer.model.modules()]
    assert trainer.step == 1
    other = evaluate(trainer.model, datasets[1], batch_size=1)
    assert other["loss"] == pytest.approx(metrics["loss"], abs=1e-6)


def test_validation_restores_modes_after_failure(model_config, datasets, monkeypatch):
    trainer = make_trainer(model_config, datasets)

    def fail(*args):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(trainer.model, "forward", fail)
    with pytest.raises(RuntimeError):
        trainer.validate()
    assert trainer.model.training


def test_failure_blocks_continuation_and_checkpoint(model_config, datasets, tmp_path):
    trainer = make_trainer(model_config, datasets)
    with torch.no_grad():
        trainer.model.embedding.weight[8, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        trainer.update()
    with pytest.raises(ValueError, match="reload"):
        trainer.update()
    with pytest.raises(ValueError, match="completed update"):
        save_checkpoint(trainer, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_checkpoint_rejects_changes_and_never_overwrites(model_config, datasets, tmp_path):
    trainer = make_trainer(model_config, datasets)
    trainer.update()
    path = tmp_path / "saved"
    save_checkpoint(trainer, path)
    before = file_hash(path / "state.json")
    with pytest.raises(ValueError, match="already exists"):
        save_checkpoint(trainer, path)
    assert file_hash(path / "state.json") == before
    with pytest.raises(ValueError, match="dataset identity"):
        load_checkpoint(path, replace(datasets[0], source_sha256="d" * 64), datasets[1])
    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(path, *datasets, expected_config=replace(trainer.config, max_steps=13))
    state = json.loads((path / "state.json").read_text())
    state["tokens_seen"] += 1
    (path / "state.json").write_bytes(canonical_json(state))
    with pytest.raises(ValueError, match="token exposure"):
        load_checkpoint(path, *datasets)
    state["tokens_seen"] -= 1
    state["runtime"]["cpu_threads"] += 1
    (path / "state.json").write_bytes(canonical_json(state))
    with pytest.raises(ValueError, match="same implementation"):
        load_checkpoint(path, *datasets)
    state["runtime"]["cpu_threads"] -= 1
    (path / "state.json").write_bytes(canonical_json(state))
    with (path / "training.safetensors").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(path, *datasets)


@pytest.fixture(scope="module")
def fixture_corpus(tmp_path_factory):
    path = tmp_path_factory.mktemp("training-fixture")
    manifest = ROOT / "data/fixtures/tiny/manifest.json"
    acquire(manifest, path / "raw")
    prepare(manifest, path / "raw", path / "corpus")
    metadata = train(
        manifest, path / "corpus", ROOT / "configs/tokenizer/debug.json", path / "codec"
    )
    codec = LatoTokenizer.load(path / "codec")
    config = ModelConfig(
        1, codec.vocab_size, 64, 32, 4, 1, 96, 10000.0, 1e-5, metadata["tokenizer_sha256"]
    )
    return path, manifest, codec, config


def test_window_seams_and_reserved_test(fixture_corpus):
    from latos.tokenization.corpus import read_split

    path, manifest, codec, config = fixture_corpus
    # The training loader must not open test.jsonl.
    (path / "corpus/test.jsonl").unlink(missing_ok=True)
    data = prepare_dataset(manifest, path / "corpus", codec, config.tokenizer_sha256, 16, "train")
    texts, _ = read_split(manifest, path / "corpus", "train")
    expected = []
    for text in texts:
        ids = codec.encode(text, add_bos=True, add_eos=True)
        expected.extend(zip(ids[:-1], ids[1:], strict=True))
    observed = [tuple(pair) for row in data.windows for pair in zip(row[:-1], row[1:], strict=True)]
    assert observed == expected
    assert all(not (a == 2 and b == 1) for a, b in observed)
    with pytest.raises(ValueError, match="reserved test"):
        prepare_dataset(manifest, path / "corpus", codec, config.tokenizer_sha256, 16, "test")


def test_overfit_tiny_sequences(model_config, datasets):
    config = TrainingConfig(
        sequence_length=16,
        max_steps=100,
        warmup_steps=3,
        batch_size=5,
        accumulation_steps=1,
        learning_rate=0.02,
    )
    trainer = Trainer(create_model(model_config, 17), config, *datasets)
    before = evaluate(trainer.model, datasets[0])["loss"]
    for _ in range(config.max_steps):
        trainer.update()
    after = evaluate(trainer.model, datasets[0])["loss"]
    assert after < 0.6 and after < before * 0.12


def test_cli_resume_in_separate_processes(fixture_corpus, tmp_path):
    path, manifest, codec, model_config = fixture_corpus
    (tmp_path / "model.json").write_bytes(canonical_json(model_config.to_dict()))
    config = TrainingConfig(sequence_length=64, max_steps=6, warmup_steps=1)
    (tmp_path / "train.json").write_bytes(canonical_json(config.to_dict()))
    base = [
        "latos",
        "train",
        "--manifest",
        str(manifest),
        "--corpus-dir",
        str(path / "corpus"),
        "--tokenizer-dir",
        str(path / "codec"),
        "--config",
        str(tmp_path / "train.json"),
        "--checkpoint-every",
        "3",
    ]

    def call(extra):
        result = subprocess.run(base + extra, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)

    call(["--model-config", str(tmp_path / "model.json"), "--output-dir", str(tmp_path / "full")])
    call(
        [
            "--model-config",
            str(tmp_path / "model.json"),
            "--output-dir",
            str(tmp_path / "part"),
            "--stop-after",
            "3",
        ]
    )
    result = call(
        [
            "--resume",
            str(tmp_path / "part/step-00000003"),
            "--output-dir",
            str(tmp_path / "resumed"),
        ]
    )
    assert result["end_step"] == 6 and result["start_step"] == 3
    for name in ("model/model.safetensors", "training.safetensors", "state.json"):
        assert file_hash(tmp_path / "full/step-00000006" / name) == file_hash(
            tmp_path / "resumed/step-00000006" / name
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS hardware unavailable")
def test_mps_update_validation_resume(model_config, datasets, tmp_path):
    trainer = Trainer(
        create_model(model_config, 17), TrainingConfig(sequence_length=16), *datasets, device="mps"
    )
    trainer.update()
    before = copy.deepcopy(trainer.model.state_dict())
    assert trainer.validate()["loss"] > 0
    equal_tree(before, trainer.model.state_dict())
    save_checkpoint(trainer, tmp_path / "mps")
    resumed = load_checkpoint(tmp_path / "mps", *datasets, device="mps")
    trainer.update()
    resumed.update()
    for left, right in zip(trainer.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


def test_clip_is_applied_before_optimizer(model_config, datasets, monkeypatch):
    trainer = make_trainer(model_config, datasets, max_grad_norm=0.01)
    original = trainer.optimizer.step
    observed = []

    def capture():
        observed.append(
            torch.stack([p.grad.square().sum() for p in trainer.model.parameters()])
            .sum()
            .sqrt()
            .item()
        )
        return original()

    monkeypatch.setattr(trainer.optimizer, "step", capture)
    metric = trainer.update()
    assert metric["grad_norm_before_clip"] > 0.01
    assert observed[0] == pytest.approx(0.01, abs=1e-7)


def test_adamw_decay_grouping_and_single_tied_embedding(model_config, datasets):
    trainer = make_trainer(model_config, datasets)
    parameters = list(trainer.model.parameters())
    grouped = [p for group in trainer.optimizer.param_groups for p in group["params"]]
    assert len(grouped) == len(parameters) == len({id(p) for p in grouped})
    before = [p.detach().clone() for p in parameters]
    for p in parameters:
        p.grad = torch.zeros_like(p)
    trainer.optimizer.step()
    for old, new in zip(before, parameters, strict=True):
        factor = (
            1 - trainer.config.learning_rate * trainer.config.weight_decay if new.ndim >= 2 else 1
        )
        torch.testing.assert_close(new, old * factor, atol=0, rtol=0)


def test_save_failure_cleans_staging_and_keeps_previous(
    model_config, datasets, tmp_path, monkeypatch
):
    trainer = make_trainer(model_config, datasets)
    save_checkpoint(trainer, tmp_path / "previous")
    before = file_hash(tmp_path / "previous/model/model.safetensors")

    def fail(*args, **kwargs):
        raise RuntimeError("readback failure")

    monkeypatch.setattr("latos.training.checkpoint.load_checkpoint", fail)
    with pytest.raises(RuntimeError, match="readback"):
        save_checkpoint(trainer, tmp_path / "next")
    assert list(tmp_path.iterdir()) == [tmp_path / "previous"]
    assert file_hash(tmp_path / "previous/model/model.safetensors") == before


def test_bad_optimizer_shape_rejected_after_checksum(model_config, datasets, tmp_path):
    from safetensors.torch import load_file, save_file

    trainer = make_trainer(model_config, datasets)
    trainer.update()
    save_checkpoint(trainer, tmp_path / "saved")
    tensor_path = tmp_path / "saved/training.safetensors"
    tensors = load_file(str(tensor_path))
    tensors["optimizer.0.exp_avg"] = torch.zeros(2)
    save_file(tensors, str(tensor_path))
    state_path = tmp_path / "saved/state.json"
    state = json.loads(state_path.read_text())
    state.update(training_sha256=file_hash(tensor_path), training_bytes=tensor_path.stat().st_size)
    state_path.write_bytes(canonical_json(state))
    with pytest.raises(ValueError, match="Invalid optimizer tensor"):
        load_checkpoint(tmp_path / "saved", *datasets)


def test_split_and_context_mismatches_rejected(model_config, datasets):
    with pytest.raises(ValueError, match="separate"):
        make_trainer(model_config, datasets[::-1])
    with pytest.raises(ValueError, match="sequence length"):
        Trainer(create_model(model_config), TrainingConfig(sequence_length=8), *datasets)
    with pytest.raises(ValueError, match="tokenizer"):
        make_trainer(replace(model_config, tokenizer_sha256="f" * 64), datasets)
