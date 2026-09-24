"""Synthetic inputs and complete state fingerprints; no corpus/evaluation readers."""

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from latos.data.manifest import canonical_json
from latos.model.storage import file_hash
from latos.training.config import TrainingConfig


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical_json(value))


def byte_hash(value):
    return hashlib.sha256(value).hexdigest()


class Windows:
    def __init__(self, directory):
        self.tokens = np.memmap(directory / "tokens.u32", dtype="<u4", mode="r")
        self.offsets = np.memmap(directory / "offsets.u64", dtype="<u8", mode="r")

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return tuple(self.tokens[int(self.offsets[index]) : int(self.offsets[index + 1])].tolist())


class SyntheticDataset:
    target_masks = None

    def __init__(self, directory, descriptor, *, dummy=False):
        metadata = read(directory / "dataset.json")
        if metadata["descriptor"] != descriptor:
            raise ValueError("Synthetic data descriptor differs")
        for name in ("tokens.u32", "offsets.u64"):
            if file_hash(directory / name) != metadata["files"][name]:
                raise ValueError("Synthetic cache checksum mismatch")
        self.windows = Windows(directory)
        self.sequence_length = 512
        self.vocab_size = descriptor["vocab_size"]
        self.tokenizer_sha256 = descriptor["tokenizer_sha256"]
        self.split = "validation" if dummy else "train"
        self.source_sha256 = byte_hash(canonical_json(descriptor))
        lengths = np.diff(self.windows.offsets.astype(np.int64))
        if (
            int(self.windows.offsets[0]) != 0
            or int(self.windows.offsets[-1]) != len(self.windows.tokens)
            or len(lengths) != descriptor["windows"]
            or not np.all((lengths >= 2) & (lengths <= 512))
        ):
            raise ValueError("Invalid synthetic offsets")
        if np.any(
            (self.windows.tokens < 1)
            | (self.windows.tokens == 3)
            | (self.windows.tokens >= self.vocab_size)
        ):
            raise ValueError("Invalid synthetic token IDs")
        self.identity = {
            "schema_version": 1,
            "boundary_policy": "original-synthetic-variable-windows-v1",
            "sequence_length": 512,
            "vocab_size": self.vocab_size,
            "tokenizer_sha256": self.tokenizer_sha256,
            "split": self.split,
            "source_sha256": self.source_sha256,
            "windows": len(lengths),
            "targets": int((lengths - 1).sum()),
            "payloads": metadata["files"],
            "scope": "synthetic token IDs; dummy validation identity is never scored",
        }

    def target_count(self, index):
        return int(self.windows.offsets[index + 1]) - int(self.windows.offsets[index]) - 1


def descriptor(lengths_path, model, plan):
    return {
        "lengths_sha256": file_hash(lengths_path),
        "windows": read(lengths_path)["windows"],
        "vocab_size": model.vocab_size,
        "tokenizer_sha256": model.tokenizer_sha256,
        "synthetic_seed": plan["synthetic_seed"],
        "shuffle_seed": plan["shuffle_seed"],
        "generator": "64 fixed 32-ID motifs with 25% common noise and 5% "
        "full-vocabulary noise; BOS/EOS per synthetic window",
    }


def make_cache(lengths_path, directory, model, plan):
    directory.mkdir(parents=True, exist_ok=False)
    profile = read(lengths_path)
    shape_order = np.asarray(profile["lengths"], dtype=np.int64)
    if len(shape_order) != profile["windows"] or not np.all(
        (shape_order >= 2) & (shape_order <= 512)
    ):
        raise ValueError("Invalid shape profile")
    permutation = torch.randperm(
        len(shape_order), generator=torch.Generator().manual_seed(plan["shuffle_seed"])
    ).numpy()
    lengths = np.empty_like(shape_order)
    lengths[permutation] = shape_order  # The trainer shuffle reproduces the recorded length order.
    rng = np.random.default_rng(plan["synthetic_seed"])
    motifs = rng.integers(4, min(516, model.vocab_size), size=(64, 32), dtype=np.uint32)
    offsets = [0]
    with (directory / "tokens.u32").open("xb") as stream:
        for index, length in enumerate(lengths.tolist()):
            row = np.resize(motifs[index % 64], length).astype("<u4")
            noise = rng.random(length)
            common = rng.integers(4, min(4096, model.vocab_size), size=length, dtype=np.uint32)
            rare = rng.integers(4, model.vocab_size, size=length, dtype=np.uint32)
            row[noise < 0.25] = common[noise < 0.25]
            row[noise > 0.95] = rare[noise > 0.95]
            row[0], row[-1] = 1, 2
            stream.write(row.tobytes())
            offsets.append(offsets[-1] + length)
    (directory / "offsets.u64").write_bytes(np.asarray(offsets, dtype="<u8").tobytes())
    meta = {
        "descriptor": descriptor(lengths_path, model, plan),
        "files": {name: file_hash(directory / name) for name in ("tokens.u32", "offsets.u64")},
    }
    write_once(directory / "dataset.json", meta)
    return meta


def training_config(plan):
    keys = (
        "batch_size",
        "accumulation_steps",
        "warmup_steps",
        "learning_rate",
        "min_lr_ratio",
        "weight_decay",
        "beta1",
        "beta2",
        "eps",
        "max_grad_norm",
    )
    return TrainingConfig(
        sequence_length=512,
        max_steps=plan["steps"],
        seed=plan["shuffle_seed"],
        **{key: plan[key] for key in keys},
    )


def tensor_digest(tensor):
    layout = {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
    }
    work = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(canonical_json(layout))
    h.update(work.numpy().tobytes())
    return h.hexdigest()


def rng_digest(cuda=False):
    np_state = np.random.get_state()
    metadata = {
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[2], np_state[3], np_state[4]],
    }
    h = hashlib.sha256(canonical_json(metadata))
    h.update(np_state[1].tobytes())
    h.update(torch.get_rng_state().numpy().tobytes())
    if cuda:
        for value in torch.cuda.get_rng_state_all():
            h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def snapshot(trainer):
    if not trainer.ready or any(p.grad is not None for p in trainer.model.parameters()):
        raise ValueError("Fingerprint only complete, cleared updates")
    model = {name: tensor_digest(t) for name, t in sorted(trainer.model.state_dict().items())}
    state = trainer.optimizer.state_dict()
    optimizer = {
        f"{index}.{key}": tensor_digest(t)
        for index, slots in sorted(state["state"].items())
        for key, t in sorted(slots.items())
    }
    sampler = {
        "step": trainer.step,
        "targets": trainer.tokens_seen,
        "windows": trainer.windows_seen,
        "epoch": trainer.stream.epoch,
        "cursor": trainer.stream.cursor,
        "order": tensor_digest(trainer.stream.order),
        "rng": tensor_digest(trainer.stream.generator.get_state()),
    }
    parameter_names = {id(p): name for name, p in trainer.model.named_parameters()}
    index_names = {
        str(index): parameter_names[id(parameter)]
        for saved, live in zip(state["param_groups"], trainer.optimizer.param_groups, strict=True)
        for index, parameter in zip(saved["params"], live["params"], strict=True)
    }
    result = {
        "optimizer_parameter_names": index_names,
        "model": model,
        "optimizer": optimizer,
        "optimizer_groups": byte_hash(canonical_json(state["param_groups"])),
        "sampler": sampler,
        "global_rng": rng_digest(trainer.device == "cuda"),
        "model_config": trainer.model.config.to_dict(),
        "training_config": trainer.config.to_dict(),
        "datasets": trainer.data_identities,
    }
    result["state_sha256"] = byte_hash(canonical_json(result))
    return result


def input_signature(trainer):
    begin = trainer.stream.cursor
    end = min(
        len(trainer.train_data.windows),
        begin + trainer.config.batch_size * trainer.config.accumulation_steps,
    )
    indices = trainer.stream.order[begin:end].tolist()
    h = hashlib.sha256()
    for index in indices:
        h.update(canonical_json(trainer.train_data.windows[index]))
    return {
        "sha256": h.hexdigest(),
        "windows": len(indices),
        "targets": sum(trainer.train_data.target_count(i) for i in indices),
        "lengths": [len(trainer.train_data.windows[i]) for i in indices],
    }


def compare_pair(reference, resumed, boundary_step, tolerance):
    if len(reference) != boundary_step + len(resumed):
        raise ValueError("Pair update counts disagree")
    losses, states = [], []
    first_tensors = None
    for step, (a, b) in enumerate(
        zip(reference[boundary_step:], resumed, strict=True), boundary_step + 1
    ):
        if a["metric"]["step"] != step or b["metric"]["step"] != step or a["input"] != b["input"]:
            raise ValueError("Pair steps/input identity differs")
        for key in (
            "targets",
            "tokens_seen",
            "windows_seen",
            "epoch",
            "microbatches",
            "learning_rate",
        ):
            if a["metric"][key] != b["metric"][key]:
                raise ValueError("Pair counter/schedule differs")
        if not all(math.isfinite(r["metric"]["loss"]) for r in (a, b)):
            raise ValueError("Nonfinite comparison loss")
        delta = abs(a["metric"]["loss"] - b["metric"]["loss"])
        if delta > tolerance:
            losses.append({"step": step, "delta": delta})
        if a["state"]["state_sha256"] != b["state"]["state_sha256"]:
            states.append(step)
            if first_tensors is None:
                first_tensors = {
                    "step": step,
                    "model": [
                        k
                        for k in a["state"]["model"]
                        if a["state"]["model"][k] != b["state"]["model"].get(k)
                    ],
                    "optimizer": [
                        k
                        for k in a["state"]["optimizer"]
                        if a["state"]["optimizer"][k] != b["state"]["optimizer"].get(k)
                    ],
                }
    return {
        "loss_gate_passed": not losses,
        "loss_excesses": losses,
        "prefix": "shared exact saved state; prefix not optimized twice",
        "replay_states_identical": not states,
        "first_state_difference": first_tensors,
        "different_replay_updates": states,
        "updates": len(reference),
        "replay_updates": len(resumed),
        "tolerance": tolerance,
        "scope": "Synthetic same-profile reproducibility only; no learned-quality evaluation",
    }
