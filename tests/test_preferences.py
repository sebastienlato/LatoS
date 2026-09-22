"""DPO mathematics, final-response masking, immutable reference and offline runner."""

import json
import math
from dataclasses import asdict

import pytest
import torch
from test_phase6 import args as tiny_args

from latos.chat import format_chat
from latos.cli import main
from latos.model.storage import bind_tokenizer, file_hash, load_model
from latos.preference_run import run_preferences
from latos.preferences import (
    DPOConfig,
    dpo_loss,
    pair_logps,
    preference_records,
    prepare_preferences,
    score_preferences,
    sequence_logps,
)
from latos.training.data import collate

args = tiny_args


def test_dpo_scalar_reference_and_gradient():
    p = torch.tensor([[-2.0, -4.0], [-6.0, -3.0]], dtype=torch.float64, requires_grad=True)
    r = torch.tensor([[-3.0, -3.0], [-5.0, -4.0]], dtype=torch.float64, requires_grad=True)
    loss, margins = dpo_loss(p, r, 0.2)
    expected = sum(math.log1p(math.exp(-x)) for x in (0.4, -0.4)) / 2
    assert loss.item() == pytest.approx(expected, abs=1e-14)
    loss.backward()
    expected_grad = torch.tensor(
        [[-0.1 / (1 + math.exp(x)), 0.1 / (1 + math.exp(x))] for x in (0.4, -0.4)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(p.grad, expected_grad, atol=1e-14, rtol=1e-14)
    assert r.grad is None
    assert margins.tolist() == pytest.approx([0.4, -0.4])
    same, margin = dpo_loss(p, p, 0.2)
    assert same.item() == pytest.approx(math.log(2))
    assert torch.equal(margin, torch.zeros_like(margin))
    better = p.detach().clone()
    better[:, 0] += 1
    assert dpo_loss(better, r, 0.2)[0] < loss
    extreme, _ = dpo_loss(
        torch.tensor([[10000.0, -10000.0], [-10000.0, 10000.0]]), torch.zeros(2, 2), 0.1
    )
    assert torch.isfinite(extreme)


def test_logps_independent_shift_sum_padding_gradient():
    logits = torch.arange(30, dtype=torch.float64).reshape(2, 5, 3) / 11
    logits.requires_grad_()
    labels = torch.tensor([[-100, -100, 1, 2, -100], [-100, 0, -100, -100, -100]])
    result = sequence_logps(logits, labels)
    expected = []
    for row in range(2):
        total = 0.0
        for t in range(1, 5):
            y = labels[row, t].item()
            if y != -100:
                x = logits[row, t - 1].detach().tolist()
                total += x[y] - math.log(sum(math.exp(v) for v in x))
        expected.append(total)
    assert result.tolist() == pytest.approx(expected, abs=1e-14)
    result.sum().backward()
    assert torch.equal(logits.grad[0, [0, 3, 4]], torch.zeros(3, 3, dtype=torch.float64))
    assert not torch.equal(logits.grad[0, 1], torch.zeros(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="target"):
        sequence_logps(logits, torch.full_like(labels, -100))
    labels[0, 2] = 3
    with pytest.raises(ValueError, match="target"):
        sequence_logps(logits, labels)


@pytest.mark.parametrize("beta", [0, -1, float("nan"), float("inf"), True])
def test_invalid_beta(beta):
    with pytest.raises(ValueError):
        dpo_loss(torch.zeros(1, 2), torch.zeros(1, 2), beta)


@pytest.mark.parametrize(
    "change",
    [
        {"steps": True},
        {"steps": 1001},
        {"seed": -1},
        {"learning_rate": float("nan")},
        {"clip_norm": 0},
    ],
)
def test_invalid_config(change):
    with pytest.raises(ValueError):
        DPOConfig(**change)


def test_preferences_splits_masking_and_padding(args):
    model = load_model(args.base)
    codec = bind_tokenizer(model.config, args.tokenizer_dir)
    records = {s: preference_records(args.conversations, s) for s in ("train", "validation")}
    assert [len(records[s]) for s in records] == [192, 32]
    assert not {r["source_group"] for r in records["train"]} & {
        r["source_group"] for r in records["validation"]
    }
    with pytest.raises(ValueError, match="reserved"):
        preference_records(args.conversations, "test")
    recall = next(r for r in records["train"] if r["family"] == "recall")
    data = prepare_preferences([recall], codec, args.tokenizer_sha256, 256, "train")
    prefix, _ = format_chat(codec, recall["prompt"], max_length=256, generation_prompt=True)
    for row, mask, side in zip(
        data.windows, data.target_masks, ("chosen", "rejected"), strict=True
    ):
        assert row[: len(prefix)] == prefix
        assert not any(mask[: len(prefix)])
        assert all(mask[len(prefix) :]) and row[-1] == 2
        assert sum(mask) == len(codec.encode(recall[side])) + 1
    ids, labels, targets = collate(data, [0, 1], "cpu")
    assert targets == sum(map(sum, data.target_masks))
    assert (labels[ids == 0] == -100).all()
    batched, _ = pair_logps(model, data, [0], "cpu")
    single = []
    for i in (0, 1):
        x, y, _ = collate(data, [i], "cpu")
        single.append(sequence_logps(model(x).logits, y)[0])
    torch.testing.assert_close(batched[0], torch.stack(single), atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="context"):
        prepare_preferences([recall], codec, args.tokenizer_sha256, 2, "train")
    reference = load_model(args.base).requires_grad_(False).eval()
    before = {n: t.clone() for n, t in model.state_dict().items()}
    model.train()
    score = score_preferences(model, reference, data, DPOConfig(), "cpu")
    assert score["relative_ties"] == 1 and score["relative_wins"] == 0
    assert model.training and not reference.training
    assert all(torch.equal(before[n], t) for n, t in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def configure(args):
    args.config.write_text(json.dumps(asdict(DPOConfig(steps=2))))
    return args


def test_complete_run_repetition_and_cli(args, capsys):
    configure(args)
    result = run_preferences(args)
    assert result["steps"] == 2 and result["pairs_seen"] == 8
    assert result["instructions"]["cases"] == 32
    assert result["policy_updated"] and result["frozen_reference_unchanged"]
    assert not result["reference_gradients_and_optimizer_slots"]
    assert result["round_trip"]["max_absolute_difference"] == 0
    assert file_hash(args.base / "model.safetensors") == args.base_sha256
    baseline = json.loads((args.output_dir / "baseline.json").read_text())
    assert baseline["preferences"]["train"]["relative_ties"] == 192
    assert baseline["preferences"]["train"]["loss"] == pytest.approx(math.log(2))
    first = args.output_dir
    with pytest.raises(FileExistsError):
        run_preferences(args)
    args.output_dir = first.parent / "repeat"
    cli = ["preferences"]
    for name in (
        "base",
        "base_sha256",
        "tokenizer_dir",
        "conversations",
        "manifest",
        "corpus_dir",
        "config",
        "output_dir",
        "device",
        "threads",
    ):
        cli.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    assert main(cli) == 0
    assert main(cli) == 1
    capsys.readouterr()
    left = load_model(first / "model")
    right = load_model(args.output_dir / "model")
    assert all(torch.equal(t, right.state_dict()[n]) for n, t in left.state_dict().items())
    assert (
        json.loads((first / "summary.json").read_text())["preferences"]
        == json.loads((args.output_dir / "summary.json").read_text())["preferences"]
    )


def test_wrong_baseline_and_failure_retention(args, monkeypatch):
    configure(args)
    original_hash = args.base_sha256
    args.base_sha256 = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        run_preferences(args)
    assert not args.output_dir.exists()
    args.base_sha256 = original_hash

    def fail(*args, **kwargs):
        raise RuntimeError("injected DPO update failure")

    monkeypatch.setattr(torch.optim.AdamW, "step", fail)
    with pytest.raises(RuntimeError, match="injected"):
        run_preferences(args)
    assert (args.output_dir / "baseline.json").exists()
    assert (args.output_dir / "failure.json").exists()
    assert not (args.output_dir / "summary.json").exists()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_tiny_mps_dpo(args):
    configure(args)
    args.device = "mps"
    result = run_preferences(args)
    assert result["frozen_reference_unchanged"] and result["policy_updated"]
    assert result["round_trip"]["max_absolute_difference"] == 0


def test_invalid_objective_inputs():
    for p, r in (
        (torch.zeros(0, 2), torch.zeros(0, 2)),
        (torch.zeros(2), torch.zeros(2)),
        (torch.zeros(1, 2), torch.zeros(2, 2)),
        (torch.tensor([[float("nan"), 0.0]]), torch.zeros(1, 2)),
        (torch.zeros(1, 2), torch.tensor([[0.0, float("inf")]])),
    ):
        with pytest.raises(ValueError):
            dpo_loss(p, r, 0.1)
    with pytest.raises(ValueError, match="Nonfinite"):
        sequence_logps(torch.full((1, 2, 4), float("nan")), torch.tensor([[-100, 2]]))


def test_independent_verifier_rejects_tampered_run(args, tmp_path):
    import runpy
    from argparse import Namespace
    from pathlib import Path

    configure(args)
    run_preferences(args)
    verify = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "experiments/phase-10/verify.py")
    )["verify"]
    options = Namespace(
        run_dir=args.output_dir, base=args.base, tokenizer_dir=args.tokenizer_dir, device="cpu"
    )
    result = verify(options)
    assert result["complete_updates"] == 2 and result["pairs_accounted"] == 8
    summary = args.output_dir / "summary.json"
    summary.write_text("{}")
    with pytest.raises(ValueError, match="Artifact changed"):
        verify(options)
