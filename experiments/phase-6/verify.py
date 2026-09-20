"""Independently rehash, rescore and replay a completed SFT run on its recorded runtime."""

import argparse
import json
from pathlib import Path

import torch

from latos.chat import format_chat
from latos.data.manifest import canonical_json
from latos.instruction import load_conversations, prepare_conversations
from latos.model.sampling import generate
from latos.model.storage import bind_tokenizer, file_hash, load_model
from latos.training.checkpoint import load_checkpoint
from latos.training.data import prepare_dataset
from latos.training.engine import evaluate


def verify(args):
    root = args.run_dir
    inventory = json.loads((root / "artifacts.json").read_text())
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual_files != set(inventory) | {"artifacts.json"}:
        raise ValueError("Run inventory file set mismatch")
    for name, entry in inventory.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Inventory paths must stay inside the run")
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Inventory artifact escapes the run")
        if path.stat().st_size != entry["bytes"] or file_hash(path) != entry["sha256"]:
            raise ValueError(f"Run artifact mismatch: {name}")
    plan = json.loads((root / "plan.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    baseline = json.loads((root / "baseline.json").read_text())
    device = plan["runtime"]["device"]
    torch.set_num_threads(plan["runtime"]["cpu_threads"])
    model = load_model(args.base, expected_tokenizer_sha256=plan["tokenizer_sha256"])
    if file_hash(args.base / "model.safetensors") != plan["base_sha256"]:
        raise ValueError("Preserved base mismatch")
    initial = load_model(root / "step-00000000/model")
    if not all(torch.equal(v, initial.state_dict()[k]) for k, v in model.state_dict().items()):
        raise ValueError("SFT initialization differs from selected base")
    codec = bind_tokenizer(model.config, args.tokenizer_dir)
    datasets = [
        prepare_conversations(
            args.conversations,
            codec,
            plan["tokenizer_sha256"],
            plan["config"]["sequence_length"],
            split,
        )
        for split in ("train", "validation")
    ]
    regression = prepare_dataset(
        args.manifest,
        args.corpus_dir,
        codec,
        plan["tokenizer_sha256"],
        plan["config"]["sequence_length"],
        "validation",
    )
    if {"train": datasets[0].identity, "validation": datasets[1].identity} != plan["datasets"]:
        raise ValueError("Instruction data identity mismatch")
    if regression.identity != plan["regression"]:
        raise ValueError("Regression data identity mismatch")
    final = load_checkpoint(root / summary["selected_checkpoint"], *datasets, device)
    if final.step != plan["config"]["max_steps"] or final.tokens_seen != summary["tokens_seen"]:
        raise ValueError("Final checkpoint counters mismatch")
    metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    if [m["step"] for m in metrics] != list(range(1, final.step + 1)):
        raise ValueError("Missing update metrics")
    if sum(m["targets"] for m in metrics) != final.tokens_seen:
        raise ValueError("Target exposure mismatch")
    records = load_conversations(args.conversations, "validation")
    measured = {}
    for name, candidate, recorded in (
        ("base", model.to(device), baseline),
        ("sft", final.model, summary),
    ):
        measured[name] = {}
        for label, dataset in (
            ("assistant_validation", datasets[1]),
            ("english_validation", regression),
        ):
            score = evaluate(candidate, dataset, plan["config"]["batch_size"])
            delta = abs(score["loss"] - recorded[label]["loss"])
            if delta > 1e-5 or score["targets"] != recorded[label]["targets"]:
                raise ValueError("Recomputed validation does not match")
            measured[name][label] = {**score, "absolute_loss_difference": delta}
        checks = recorded["instructions"]
        if len(checks["samples"]) != len(records):
            raise ValueError("Instruction sample count mismatch")
        correct, eos, families = 0, 0, {}
        for record, saved in zip(records, checks["samples"], strict=True):
            messages = record["messages"]
            prompt, _ = format_chat(
                codec,
                messages[:-1],
                max_length=plan["config"]["sequence_length"],
                generation_prompt=True,
            )
            generated = generate(candidate, list(prompt), **plan["sampling"])
            text = codec.decode(generated["generated_ids"])
            match = text.strip() == messages[-1]["content"].strip()
            if (
                saved["id"] != record["id"]
                or saved["prompt_ids"] != list(prompt)
                or saved["generated_ids"] != generated["generated_ids"]
                or saved["continuation"] != text
                or saved["correct"] != match
                or saved["stop_reason"] != generated["stop_reason"]
            ):
                raise ValueError("Instruction sample replay differs")
            correct += int(match)
            eos += int(generated["stop_reason"] == "eos")
            family = families.setdefault(record["family"], {"correct": 0, "cases": 0})
            family["correct"] += int(match)
            family["cases"] += 1
        if (
            checks["correct"] != correct
            or checks["eos_stops"] != eos
            or checks["cases"] != len(records)
            or checks["families"] != families
        ):
            raise ValueError("Instruction aggregate scores differ")
        measured[name]["instruction_samples_replayed"] = len(records)
    checkpoints = sorted(root.glob("step-*/state.json"))
    if len(checkpoints) < 3:
        raise ValueError("Verification needs an intermediate recovery checkpoint")
    replay = load_checkpoint(checkpoints[-2].parent, *datasets, device)
    replay_start = replay.step
    while replay.step < final.step:
        result = replay.update()
        if result["tokens_seen"] != metrics[replay.step - 1]["tokens_seen"]:
            raise ValueError("Replay target counter mismatch")
    maximum = 0.0
    for key, expected in final.model.state_dict().items():
        actual = replay.model.state_dict()[key]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        maximum = max(maximum, (actual - expected).abs().max().item())
    report = {
        "status": "ok",
        "inventory_files_verified": len(inventory),
        "inventory_sha256": file_hash(root / "artifacts.json"),
        "initial_weights_equal_selected_base": True,
        "recomputed": measured,
        "updates_accounted": len(metrics),
        "targets_accounted": final.tokens_seen,
        "recovery": {
            "device": device,
            "from_step": replay_start,
            "to_step": final.step,
            "max_weight_absolute_difference": maximum,
            "atol": 1e-6,
            "rtol": 1e-5,
        },
        "scope": "Same local runtime replay and scalar validation; no cross-device guarantee",
    }
    with args.output.open("xb") as stream:
        stream.write(canonical_json(report))
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--base", type=Path, default=Path("outputs/phase-5-english-pilot/step-00003000/model")
    )
    parser.add_argument(
        "--tokenizer-dir", type=Path, default=Path("artifacts/tokenizers/english-bpe-v1")
    )
    parser.add_argument(
        "--conversations", type=Path, default=Path("data/processed/english-instructions-v1")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/english-books-v1.json")
    )
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/processed/english-books-v1"))
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args())
