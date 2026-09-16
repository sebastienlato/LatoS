"""Independent numerical references, causal behavior, objective alignment, and snapshots."""

import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from latos.data.manifest import canonical_json
from latos.model import ModelConfig, create_model
from latos.model.network import CausalAttention, RMSNorm, SwiGLU, causal_loss, rotary
from latos.model.sampling import generate, sample_next
from latos.model.storage import bind_tokenizer, file_hash, load_model, save_model


@pytest.fixture
def config():
    return ModelConfig(1, 260, 16, 16, 2, 2, 32, 10000.0, 1e-5, "0" * 64)


def test_rmsnorm_against_scalar_equation_and_gradient():
    module = RMSNorm(4, 1e-5).double()
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float64, requires_grad=True)
    expected = torch.tensor(
        [[v / math.sqrt((1 + 4 + 9 + 16) / 4 + 1e-5) for v in (1, -2, 3, -4)]], dtype=torch.float64
    )
    torch.testing.assert_close(module(x), expected, atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(module, (x,))


def reference_rotation(x, theta):
    # Explicit scalar angles provide an independent reference for the vectorized operation.
    rows = []
    for time in range(x.shape[-2]):
        columns = []
        for pair in range(x.shape[-1] // 2):
            angle = time / theta ** (2 * pair / x.shape[-1])
            a, b = x[..., time, 2 * pair], x[..., time, 2 * pair + 1]
            columns.extend(
                [
                    a * math.cos(angle) - b * math.sin(angle),
                    a * math.sin(angle) + b * math.cos(angle),
                ]
            )
        rows.append(torch.stack(columns, dim=-1))
    return torch.stack(rows, dim=-2)


def test_rotary_equation_norm_and_gradients():
    q = torch.randn(2, 2, 5, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn_like(q)
    rotated, rotated_k = rotary(q, k, 10000.0)
    torch.testing.assert_close(rotated, reference_rotation(q, 10000.0), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(rotated_k, reference_rotation(k, 10000.0), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(rotated.square().sum(-1), q.square().sum(-1))
    torch.testing.assert_close(rotated[..., 0, :], q[..., 0, :])
    assert torch.autograd.gradcheck(lambda x: rotary(x, k, 10000.0)[0], (q,))


def reference_attention(module, x):
    batch, length, width = x.shape
    qkv = x @ module.qkv.weight.T
    q, k, v = [
        part.reshape(batch, length, module.heads, module.head_dim).transpose(1, 2)
        for part in qkv.split(width, dim=-1)
    ]
    q, k = reference_rotation(q, module.theta), reference_rotation(k, module.theta)
    rows = []
    for time in range(length):
        scores = (q[:, :, time : time + 1] @ k[:, :, : time + 1].transpose(-1, -2)) / math.sqrt(
            module.head_dim
        )
        rows.append(scores.softmax(-1) @ v[:, :, : time + 1])
    result = torch.cat(rows, dim=2).transpose(1, 2).reshape(batch, length, width)
    return result @ module.output.weight.T


def test_sdpa_matches_explicit_causal_attention_and_gradients(config):
    module = CausalAttention(config).double()
    x = torch.randn(2, 5, config.d_model, dtype=torch.float64, requires_grad=True)
    actual = module(x)
    expected = reference_attention(module, x)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    variables = (x, module.qkv.weight, module.output.weight)
    left = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
    right = torch.autograd.grad(expected.square().sum(), variables)
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


def test_swiglu_equation(config):
    module = SwiGLU(config).double()
    x = torch.randn(2, 3, config.d_model, dtype=torch.float64)
    gate = x @ module.gate.weight.T
    expected = ((gate * gate.sigmoid()) * (x @ module.up.weight.T)) @ module.down.weight.T
    torch.testing.assert_close(module(x), expected, atol=1e-12, rtol=1e-12)


def test_model_shapes_causality_batch_independence_and_gradients(config):
    model = create_model(config, seed=29)
    ids = torch.arange(14).reshape(2, 7)
    result = model(ids, labels=ids)
    assert result.logits.shape == (2, 7, 260)
    assert result.loss is not None and torch.isfinite(result.loss)
    changed = ids.clone()
    changed[:, 4:] += 37
    torch.testing.assert_close(
        result.logits[:, :4], model(changed).logits[:, :4], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(model(ids[:1]).logits, result.logits[:1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(model(ids[:, :4]).logits, result.logits[:, :4], atol=1e-6, rtol=1e-6)
    assert model(ids[:, :1]).logits.shape == (2, 1, 260)
    result.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.grad.abs().sum().item() for p in model.parameters()) > 0


def test_loss_manual_alignment_mask_and_gradient():
    logits = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.0, 3.0, 1.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]], requires_grad=True
    )
    labels = torch.tensor([[2, 0, -100, 1]])
    expected = (
        (torch.logsumexp(logits[0, 0], 0) - logits[0, 0, 0])
        + (torch.logsumexp(logits[0, 2], 0) - logits[0, 2, 1])
    ) / 2
    loss = causal_loss(logits, labels)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.equal(logits.grad[0, 1], torch.zeros(3))
    assert torch.equal(logits.grad[0, 3], torch.zeros(3))


def test_loss_rejects_no_targets_and_invalid_ids():
    logits = torch.zeros(1, 3, 5)
    with pytest.raises(ValueError, match="unmasked"):
        causal_loss(logits, torch.full((1, 3), -100))
    with pytest.raises(ValueError, match="invalid target"):
        causal_loss(logits, torch.tensor([[0, 1, 5]]))
    with pytest.raises(ValueError, match="time>=2"):
        causal_loss(logits[:, :1], torch.tensor([[0]]))


def test_parameter_formula_and_weight_tying(config):
    model = create_model(config)
    expected = 260 * 16 + 2 * (4 * 16 * 16 + 3 * 16 * 32 + 2 * 16) + 16
    assert model.parameter_count == config.parameter_count == expected
    names = dict(model.named_parameters())
    assert "embedding.weight" in names and not any("head" in name for name in names)
    hidden = []
    handle = model.final_norm.register_forward_hook(lambda _, __, output: hidden.append(output))
    result = model(torch.tensor([[1, 4, 5]]))
    handle.remove()
    torch.testing.assert_close(result.logits, hidden[0] @ model.embedding.weight.T)


def test_seeded_initialization_isolated_and_float32(config):
    before = torch.get_rng_state().clone()
    model = create_model(config, seed=17)
    assert torch.equal(before, torch.get_rng_state())
    assert all(
        torch.equal(p, q)
        for p, q in zip(model.parameters(), create_model(config, 17).parameters(), strict=True)
    )
    assert not torch.equal(model.embedding.weight, create_model(config, 18).embedding.weight)
    default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        other = create_model(config, 17)
        assert all(p.dtype == torch.float32 for p in other.parameters())
        assert torch.equal(other.embedding.weight, model.embedding.weight)
    finally:
        torch.set_default_dtype(default)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("d_model", 15),
        ("n_heads", 3),
        ("n_layers", 0),
        ("context_length", 0),
        ("vocab_size", True),
        ("tokenizer_sha256", "bad"),
        ("rope_theta", float("nan")),
        ("norm_eps", 0),
    ],
)
def test_invalid_configuration(config, field, value):
    with pytest.raises(ValueError):
        replace(config, **{field: value})


@pytest.mark.parametrize(
    "ids",
    [
        torch.tensor([], dtype=torch.long).reshape(1, 0),
        torch.zeros(1, 17, dtype=torch.long),
        torch.tensor([[260]]),
        torch.tensor([[-1]]),
        torch.tensor([[1.0]]),
        torch.tensor([1, 2]),
    ],
)
def test_invalid_model_input(config, ids):
    with pytest.raises(ValueError):
        create_model(config)(ids)


def test_snapshot_exact_round_trip_and_corruption(config, tmp_path):
    model = create_model(config, seed=8).eval()
    ids = torch.tensor([[1, 4, 7, 10]])
    metadata = save_model(model, tmp_path / "model")
    assert metadata["parameter_count"] == model.parameter_count
    loaded = load_model(tmp_path / "model", expected_tokenizer_sha256="0" * 64)
    assert not loaded.training
    assert torch.equal(model(ids).logits, loaded(ids).logits)
    with pytest.raises(ValueError, match="already exists"):
        save_model(model, tmp_path / "model")
    with pytest.raises(ValueError, match="tokenizer identity"):
        load_model(tmp_path / "model", expected_tokenizer_sha256="1" * 64)
    weights = tmp_path / "model/model.safetensors"
    original = weights.read_bytes()
    weights.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path / "model")


def test_snapshot_rejects_wrong_shape_with_updated_hash(config, tmp_path):
    directory = tmp_path / "model"
    save_model(create_model(config), directory)
    path = directory / "model.safetensors"
    tensors = load_file(str(path))
    tensors["embedding.weight"] = tensors["embedding.weight"][:-1]
    save_file(tensors, str(path))
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata.update(weights_sha256=file_hash(path), weights_bytes=path.stat().st_size)
    (directory / "metadata.json").write_bytes(canonical_json(metadata))
    with pytest.raises(ValueError, match="Invalid model tensor"):
        load_model(directory)


def test_nonfinite_weights_not_saved(config, tmp_path):
    model = create_model(config)
    with torch.no_grad():
        model.embedding.weight[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite float32"):
        save_model(model, tmp_path / "model")
    assert not (tmp_path / "model").exists()


def test_sampling_greedy_topk_and_no_input_mutation():
    logits = torch.zeros(260)
    logits[0] = 100
    logits[10] = 10
    original = logits.clone()
    generator = torch.Generator().manual_seed(7)
    assert sample_next(logits, 0.0, 0, generator) == 10
    assert sample_next(logits, 1.0, 1, generator) == 10
    assert sample_next(logits, 5e-324, 0, generator) == 10
    assert torch.equal(logits, original)
    with pytest.raises(ValueError):
        sample_next(logits * torch.nan, 1.0, 0, generator)


def test_causal_gradient_cannot_reach_future_inputs(config):
    model = create_model(config)
    captured = []

    def capture(_, __, output):
        output.retain_grad()
        captured.append(output)

    hook = model.embedding.register_forward_hook(capture)
    model(torch.tensor([[1, 4, 5, 6, 7, 8]])).logits[:, 2].sum().backward()
    hook.remove()
    assert captured[0].grad[:, :3].abs().sum() > 0
    assert torch.equal(captured[0].grad[:, 3:], torch.zeros_like(captured[0].grad[:, 3:]))


def test_generation_restores_mode_after_failure(config, monkeypatch):
    model = create_model(config).train()

    def broken(*args):
        raise RuntimeError("Sampling failure")

    monkeypatch.setattr("latos.model.sampling.sample_next", broken)
    with pytest.raises(RuntimeError, match="Sampling failure"):
        generate(model, [1, 4])
    assert model.training


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS hardware unavailable")
def test_mps_sampling_transfers_before_float64_conversion():
    logits = torch.linspace(-1, 1, 260, dtype=torch.float32)
    cpu = sample_next(logits, 0.8, 20, torch.Generator().manual_seed(71))
    mps = sample_next(logits.to("mps"), 0.8, 20, torch.Generator().manual_seed(71))
    assert mps == cpu


def test_generation_seed_context_eos_and_mode_restore(config, monkeypatch):
    model = create_model(config)
    before = torch.get_rng_state().clone()
    first = generate(model, [1, 4], max_new_tokens=4, temperature=0.8, top_k=8, seed=19)
    assert first == generate(model, [1, 4], max_new_tokens=4, temperature=0.8, top_k=8, seed=19)
    assert torch.equal(before, torch.get_rng_state()) and model.training
    assert not set(first["generated_ids"]) & {0, 1, 3}
    assert generate(model, [1] * 16, max_new_tokens=1)["stop_reason"] == "context_limit"
    assert generate(model, [1, 2])["stop_reason"] == "eos"
    assert generate(model, [1], max_new_tokens=0)["generated_ids"] == []
    monkeypatch.setattr("latos.model.sampling.sample_next", lambda *args: 2)
    assert generate(model, [1, 4])["generated_ids"] == [2]
    with pytest.raises(ValueError, match="no silent truncation"):
        generate(model, [1] * 17)


def test_cli_model_check():
    config_path = Path(__file__).resolve().parents[1] / "configs/model/debug.json"
    result = subprocess.run(
        ["latos", "model", "check", "--config", str(config_path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "ok" and report["parameter_count"] == 631104
    assert report["optimizer_steps"] == 0


def test_tokenizer_binding_verifies_hash_and_vocabulary(config, tmp_path):
    from latos.data.acquire import acquire
    from latos.data.prepare import prepare
    from latos.tokenization.training import train

    root = Path(__file__).resolve().parents[1]
    manifest = root / "data/fixtures/tiny/manifest.json"
    acquire(manifest, tmp_path / "raw")
    prepare(manifest, tmp_path / "raw", tmp_path / "corpus")
    metadata = train(
        manifest, tmp_path / "corpus", root / "configs/tokenizer/debug.json", tmp_path / "tokenizer"
    )
    matching = replace(
        config, vocab_size=metadata["vocab_size"], tokenizer_sha256=metadata["tokenizer_sha256"]
    )
    assert bind_tokenizer(matching, tmp_path / "tokenizer").vocab_size == metadata["vocab_size"]
    with pytest.raises(ValueError, match="vocabulary size"):
        bind_tokenizer(
            replace(matching, vocab_size=matching.vocab_size + 1), tmp_path / "tokenizer"
        )
    with pytest.raises(ValueError, match="checksum"):
        bind_tokenizer(replace(matching, tokenizer_sha256="f" * 64), tmp_path / "tokenizer")
