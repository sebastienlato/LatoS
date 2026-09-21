"""Independent numerical, freezing, persistence and inference checks for attention LoRA."""

import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import load_file, save_file

from latos.inference.cache import KVCache
from latos.lora import LoRAConfig, LoRALinear, LoRAModel, load_adapter, merge_adapter, save_adapter
from latos.model import ModelConfig, create_model
from latos.model.storage import file_hash, load_model, save_model
from latos.training import Trainer, TrainingConfig, evaluate
from latos.training.checkpoint import save_checkpoint
from latos.training.data import TokenDataset


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def base():
    return create_model(ModelConfig(1, 260, 32, 16, 2, 2, 32, 10000.0, 1e-5, "a" * 64), 7)


def datasets():
    train = TokenDataset(
        ((1, 8, 9, 2), (1, 11, 12, 2), (1, 13, 2)),
        16,
        260,
        "a" * 64,
        "train",
        "b" * 64,
        ((False, False, True, True), (False, False, True, True), (False, True, True)),
    )
    return train, replace(train, split="validation", source_sha256="c" * 64)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rank", True),
        ("rank", 0),
        ("rank", 65),
        ("alpha", float("nan")),
        ("alpha", 0),
        ("alpha", True),
        ("seed", -1),
    ],
)
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        LoRAConfig(**{field: value})


def test_linear_equation_and_gradients():
    base = torch.nn.Linear(3, 5, bias=False).requires_grad_(False)
    layer = LoRALinear(base, LoRAConfig(2, 6), torch.Generator().manual_seed(7))
    with torch.no_grad():
        layer.b.copy_(torch.arange(10).reshape(5, 2) / 10)
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
    expected = x @ (base.weight + 3 * (layer.b @ layer.a)).T
    actual = layer(x)
    torch.testing.assert_close(actual, expected)
    grads = torch.autograd.grad(actual.square().sum(), (x, layer.a, layer.b))
    ref = torch.autograd.grad(expected.square().sum(), (x, layer.a, layer.b))
    for a, b in zip(grads, ref, strict=True):
        torch.testing.assert_close(a, b)
    assert base.weight.grad is None


def test_zero_init_freezing_optimizer_and_read_only_validation(base):
    original = {k: v.clone() for k, v in base.state_dict().items()}
    rng = torch.get_rng_state().clone()
    model = LoRAModel(base, LoRAConfig(2, 4))
    assert torch.equal(torch.get_rng_state(), rng)
    ids = torch.tensor([[1, 8, 9, 2]])
    assert torch.equal(base(ids).logits, model(ids).logits)
    model(ids, ids).loss.backward()
    assert all(
        p.grad is None for n, p in model.named_parameters() if n not in model.adapter_state()
    )
    assert any(p.grad.abs().sum() > 0 for n, p in model.adapter_state().items() if n.endswith(".b"))
    assert all(
        p.grad.abs().sum() == 0 for n, p in model.adapter_state().items() if n.endswith(".a")
    )
    train, val = datasets()
    trainer = Trainer(
        model, TrainingConfig(sequence_length=16, max_steps=4, warmup_steps=0), train, val
    )
    optimizer_ids = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    assert optimizer_ids == {id(p) for p in model.adapter_state().values()}
    for _ in range(4):
        trainer.update()
        model.validate_adapter()
    assert all(torch.equal(original[k], v) for k, v in model.base_state().items())
    assert all(torch.equal(original[k], v) for k, v in base.state_dict().items())
    assert not torch.equal(base(ids).logits, model(ids).logits)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.train()
    evaluate(model, val)
    assert model.training
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_trained_roundtrip_merge_cache_and_no_mutation(base, tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    model = LoRAModel(base, LoRAConfig(2, 4))
    train, val = datasets()
    trainer = Trainer(
        model, TrainingConfig(sequence_length=16, max_steps=3, warmup_steps=0), train, val, device
    )
    for _ in range(3):
        trainer.update()
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    save_adapter(model, tmp_path / "adapter")
    restored = load_adapter(base, tmp_path / "adapter").to(device)
    merged = merge_adapter(model)
    save_model(merged, tmp_path / "merged")
    merged = load_model(tmp_path / "merged").to(device)
    assert type(merged) is type(base)
    ids = (torch.arange(64).reshape(2, 32) % 260).to(device)
    model.eval()
    tolerance = 1e-4 if device == "mps" else 1e-5
    with torch.inference_mode():
        expected = model(ids).logits
        assert torch.equal(expected, restored(ids).logits)
        torch.testing.assert_close(expected, merged(ids).logits, atol=tolerance, rtol=tolerance)
        for tested in (model, restored, merged):
            cache = KVCache(tested)
            cached = torch.cat([cache.append(ids[:, :17]), cache.append(ids[:, 17:])], dim=1)
            torch.testing.assert_close(tested(ids).logits, cached, atol=tolerance, rtol=tolerance)
    assert all(torch.equal(before[k], v.cpu()) for k, v in model.state_dict().items())
    assert not any(isinstance(m, LoRALinear) for m in merged.modules())
    with pytest.raises(ValueError, match="already exists"):
        save_adapter(model, tmp_path / "adapter")
    with pytest.raises(ValueError):
        save_model(model, tmp_path / "unmerged")
    with pytest.raises(ValueError):
        save_checkpoint(trainer, tmp_path / "unsupported-resume")
    assert not (tmp_path / "unsupported-resume").exists()


def test_wrong_base_config_and_frozen_mutation_rejected(base, tmp_path):
    model = LoRAModel(base, LoRAConfig(2, 4))
    save_adapter(model, tmp_path / "adapter")
    with pytest.raises(ValueError, match="identity"):
        load_adapter(create_model(base.config, 99), tmp_path / "adapter")
    other = create_model(replace(base.config, tokenizer_sha256="b" * 64), 7)
    with pytest.raises(ValueError, match="identity"):
        load_adapter(other, tmp_path / "adapter")
    with pytest.raises(ValueError):
        LoRAModel(model)
    with pytest.raises(ValueError):
        LoRAModel(base, LoRAConfig(32))
    with torch.no_grad():
        model.embedding.weight[0, 0] += 1
    with pytest.raises(ValueError, match="identity changed"):
        merge_adapter(model)
    with pytest.raises(ValueError, match="identity changed"):
        save_adapter(model, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("corruption", ["bytes", "extra", "missing", "shape", "dtype", "nan"])
def test_corrupt_adapter_rejected(base, tmp_path, corruption):
    model = LoRAModel(base, LoRAConfig(2, 4))
    save_adapter(model, tmp_path / "adapter")
    path = tmp_path / "adapter/adapter.safetensors"
    if corruption == "bytes":
        path.write_bytes(path.read_bytes()[:-1])
    else:
        tensors = load_file(str(path))
        key = next(iter(tensors))
        if corruption == "extra":
            tensors["unexpected"] = torch.zeros(1)
        if corruption == "missing":
            tensors.pop(key)
        if corruption == "shape":
            tensors[key] = tensors[key][:1]
        if corruption == "dtype":
            tensors[key] = tensors[key].double()
        if corruption == "nan":
            tensors[key].fill_(float("nan"))
        save_file(tensors, str(path))
        meta_path = tmp_path / "adapter/metadata.json"
        meta = json.loads(meta_path.read_text())
        meta.update(weights_bytes=path.stat().st_size, weights_sha256=file_hash(path))
        meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_adapter(base, tmp_path / "adapter")


def test_trainable_base_or_frozen_adapter_rejected(base):
    model = LoRAModel(base, LoRAConfig(2, 4))
    model.embedding.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        model.validate_adapter()
    model.embedding.weight.requires_grad_(False)
    next(iter(model.adapter_state().values())).requires_grad_(False)
    with pytest.raises(ValueError, match="only adapters"):
        model.validate_adapter()


def test_cpu_repeat_matches_parameters_and_metrics(base):
    train, val = datasets()
    runs = [
        Trainer(
            LoRAModel(base, LoRAConfig(2, 4)),
            TrainingConfig(sequence_length=16, max_steps=5, warmup_steps=0),
            train,
            val,
        )
        for _ in range(2)
    ]
    for _ in range(5):
        left, right = [t.update() for t in runs]
        for key in left.keys() - {"seconds", "targets_per_second"}:
            assert left[key] == right[key]
    assert all(
        torch.equal(v, runs[1].model.state_dict()[k]) for k, v in runs[0].model.state_dict().items()
    )
