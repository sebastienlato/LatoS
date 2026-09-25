"""Pass-tail adapter and diagnostic checkpoints; never change native historical code."""

import gc
from dataclasses import replace

from cf_common import accumulation, boundary, read, record, require, write


def advance_pass(trainer):
    import torch

    if trainer.stream.cursor == trainer.stream.size:
        require(trainer.step % 382 == 0, "Unexpected pass transition")
        trainer.stream.epoch += 1
        trainer.stream.order = torch.randperm(
            trainer.stream.size, generator=trainer.stream.generator
        )
        trainer.stream.cursor = 0


def update(trainer):
    """Only accumulation count changes for the tail; restore full config before evidence."""
    original = trainer.config
    count = accumulation(
        trainer.stream.size, trainer.stream.cursor, original.batch_size, original.accumulation_steps
    )
    trainer.config = replace(original, accumulation_steps=count)
    try:
        metric = trainer.update()
    finally:
        trainer.config = original
    require(metric["microbatches"] == count, "Native accumulation mismatch")
    return metric


def check_progress(trainer):
    epoch, cursor, windows = boundary(trainer.step)
    require(
        (trainer.stream.epoch, trainer.stream.cursor, trainer.windows_seen)
        == (epoch, cursor, windows),
        "Sampler boundary mismatch",
    )
    if trainer.step % 382 == 0:
        require(trainer.tokens_seen == trainer.step // 382 * 1943015, "Exposure mismatch")


def save_checkpoint(trainer, destination, binding, m):
    """No resume API: persist complete model/optimizer/sampler with exact tensor readback.

    The native loader assumes unflushed epochs; these distinct diagnostic artifacts
    must never be passed to that loader. Evaluation loads only the native model part.
    """
    import torch
    from safetensors.torch import load_file, save_file

    from latos.model.storage import save_model

    require(not destination.exists(), "Immutable checkpoint")
    before = m.snapshot(trainer)
    check_progress(trainer)
    destination.mkdir()
    save_model(trainer.model, destination / "model")
    state = trainer.optimizer.state_dict()
    tensors = {
        "shuffle_order": trainer.stream.order,
        "shuffle_rng": trainer.stream.generator.get_state(),
    }
    tensors.update(
        {
            f"optimizer.{i}.{key}": t.detach().cpu().contiguous()
            for i, slots in state["state"].items()
            for key, t in slots.items()
        }
    )
    require(all(torch.isfinite(t).all().item() for t in tensors.values()), "Nonfinite checkpoint")
    save_file(tensors, str(destination / "diagnostic-state.safetensors"))
    restored = load_file(str(destination / "diagnostic-state.safetensors"))
    require(set(restored) == set(tensors), "State keys")
    require(all(torch.equal(restored[k], t.cpu()) for k, t in tensors.items()), "State readback")
    model = load_file(str(destination / "model/model.safetensors"))
    require(set(model) == set(before["model"]), "Model keys")
    require({k: m.tensor_digest(t) for k, t in model.items()} == before["model"], "Model readback")
    require(m.snapshot(trainer) == before, "Checkpoint changed live state")
    write(
        destination / "diagnostic.json",
        {
            "kind": "subset-confirmation-no-resume-v1",
            "binding": binding,
            "state": before,
            "optimizer_groups": state["param_groups"],
            "readback_exact": True,
            "files": {
                p.relative_to(destination).as_posix(): record(p)
                for p in destination.rglob("*")
                if p.is_file()
            },
        },
    )
    del restored, model, tensors
    gc.collect()
    return read(destination / "diagnostic.json")["state"]["state_sha256"]
