"""Offline integration: equal-budget adaptation without either test payload."""

import json
import runpy
from pathlib import Path

import pytest
from test_phase6 import args as tiny_args

from latos.adaptation import run_adaptation
from latos.cli import main

args = tiny_args


def test_equal_budget_lora_full_and_merge_cli(args, capsys):
    args.output_dir = args.output_dir.parent / "lora"
    args.rank, args.alpha, args.adapter_seed = 2, 4.0, 91
    args.method = "lora"
    lora = run_adaptation(args)
    lora_dir = args.output_dir
    args.method = "full"
    args.output_dir = args.output_dir.parent / "full"
    full = run_adaptation(args)
    assert lora["tokens_seen"] == full["tokens_seen"]
    assert lora["windows_seen"] == full["windows_seen"]
    assert lora["trainable_parameters"] < full["trainable_parameters"]
    assert lora["optimizer_tensor_bytes"] < full["optimizer_tensor_bytes"]
    assert lora["round_trip"]["max_absolute_difference"] == 0
    assert lora["frozen_base_unchanged"]
    assert lora["merge"]["atol"] == lora["merge"]["rtol"] == 1e-4
    assert json.loads((lora_dir / "baseline.json").read_text()) == json.loads(
        (args.output_dir / "baseline.json").read_text()
    )
    assert (
        main(
            [
                "adapt",
                "merge",
                "--base",
                str(args.base),
                "--adapter",
                str(lora_dir / "adapter"),
                "--output-dir",
                str(lora_dir / "cli-merged"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "adapt",
                "merge",
                "--base",
                str(args.base),
                "--adapter",
                str(lora_dir / "adapter"),
                "--output-dir",
                str(lora_dir / "cli-merged"),
            ]
        )
        == 1
    )
    with pytest.raises(FileExistsError):
        run_adaptation(args)
    compare = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "experiments/phase-9/run.py")
    )["compare"]
    assert compare(args.output_dir.parent)["equal_inputs_baselines_and_update_exposures"]
    summary = args.output_dir / "summary.json"
    changed = json.loads(summary.read_text())
    changed["steps"] = 1
    summary.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Incomplete"):
        compare(args.output_dir.parent)


def test_failed_run_retains_evidence(args, monkeypatch):
    from latos.adaptation import Trainer

    args.rank, args.alpha, args.adapter_seed, args.method = 2, 4.0, 91, "lora"

    def fail(self):
        raise RuntimeError("injected adaptation failure")

    monkeypatch.setattr(Trainer, "update", fail)
    with pytest.raises(RuntimeError, match="injected"):
        run_adaptation(args)
    assert (args.output_dir / "baseline.json").exists()
    assert (args.output_dir / "plan.json").exists()
    assert (args.output_dir / "failure.json").exists()
    assert not (args.output_dir / "summary.json").exists()


def test_wrong_base_before_output(args):
    args.rank, args.alpha, args.adapter_seed, args.method = 2, 4.0, 91, "lora"
    args.base_sha256 = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        run_adaptation(args)
    assert not args.output_dir.exists()
