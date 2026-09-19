"""Atomic, non-overwriting tensor-only checkpoints with strict resume identities."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from latos import __version__
from latos.data.manifest import canonical_json
from latos.model.storage import file_hash, load_model, save_model
from latos.training.config import TrainingConfig
from latos.training.data import TokenDataset
from latos.training.engine import Trainer


def implementation_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def source_identity() -> dict:
    """Record Git identity when running from a checkout; never store filenames."""
    root = Path(__file__).resolve().parents[3]
    unknown = {"commit": None, "dirty": None}
    if not (root / ".git").exists():
        return unknown
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except OSError, subprocess.CalledProcessError:
        return unknown
    return {"commit": commit, "dirty": dirty}


def runtime_identity(device: str) -> dict:
    return {
        "latos": __version__,
        "implementation_sha256": implementation_hash(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "device": device,
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        "mps_fast_math": os.environ.get("PYTORCH_MPS_FAST_MATH", "0"),
        "cuda_device": torch.cuda.get_device_name() if device == "cuda" else None,
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def save_checkpoint(trainer: Trainer, directory: Path) -> dict:
    if directory.exists():
        raise ValueError("Checkpoint output already exists; choose a new destination")
    if not trainer.ready or any(p.grad is not None for p in trainer.model.parameters()):
        raise ValueError("Checkpoint requires a completed update with cleared gradients")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".latos-training-", dir=directory.parent))
    try:
        save_model(trainer.model, staging / "model")
        state = trainer.optimizer.state_dict()
        tensors = {
            "shuffle_rng": trainer.stream.generator.get_state(),
            "shuffle_order": trainer.stream.order,
        }
        for index, slots in state["state"].items():
            for name, tensor in slots.items():
                tensors[f"optimizer.{index}.{name}"] = tensor.detach().cpu().contiguous()
        save_file(tensors, str(staging / "training.safetensors"))
        metadata = {
            "schema_version": 1,
            "kind": "latos-training-float32-v1",
            "runtime": runtime_identity(trainer.device),
            "source": source_identity(),
            "config": trainer.config.to_dict(),
            "datasets": trainer.data_identities,
            "step": trainer.step,
            "tokens_seen": trainer.tokens_seen,
            "windows_seen": trainer.windows_seen,
            "epoch": trainer.stream.epoch,
            "cursor": trainer.stream.cursor,
            "optimizer_groups": state["param_groups"],
            "training_sha256": file_hash(staging / "training.safetensors"),
            "training_bytes": (staging / "training.safetensors").stat().st_size,
            "model_metadata_sha256": file_hash(staging / "model" / "metadata.json"),
        }
        (staging / "state.json").write_bytes(canonical_json(metadata))
        # Read-back also validates optimizer slots, counters, permutation, and RNG state.
        load_checkpoint(
            staging,
            trainer.train_data,
            trainer.validation_data,
            trainer.device,
            expected_config=trainer.config,
        )
        staging.rename(directory)
        return metadata
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_checkpoint(
    directory: Path,
    train: TokenDataset,
    validation: TokenDataset,
    device: str = "cpu",
    *,
    expected_config: TrainingConfig | None = None,
) -> Trainer:
    metadata = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != 1 or metadata["kind"] != "latos-training-float32-v1":
        raise ValueError("Unsupported training checkpoint")
    if metadata["runtime"] != runtime_identity(device):
        raise ValueError("Resume requires the same implementation, runtime, backend, and settings")
    config = TrainingConfig(**metadata["config"])
    if expected_config is not None and config != expected_config:
        raise ValueError("Resume training configuration mismatch")
    if metadata["datasets"] != {"train": train.identity, "validation": validation.identity}:
        raise ValueError("Resume dataset identity mismatch")
    if file_hash(directory / "model" / "metadata.json") != metadata["model_metadata_sha256"]:
        raise ValueError("Checkpoint model metadata checksum mismatch")
    model = load_model(directory / "model", expected_tokenizer_sha256=train.tokenizer_sha256)
    trainer = Trainer(model, config, train, validation, device)
    step, epoch, cursor = metadata["step"], metadata["epoch"], metadata["cursor"]
    for key in ("step", "tokens_seen", "windows_seen", "epoch", "cursor"):
        if type(metadata[key]) is not int or metadata[key] < 0:
            raise ValueError("Invalid training checkpoint counter")
    size = len(train.windows)
    if step > config.max_steps or cursor > size:
        raise ValueError("Checkpoint progress exceeds configured limits")
    # Derive sampler progress from update count, including short epoch-tail batches.
    batches_per_epoch = (size + config.batch_size - 1) // config.batch_size
    batches = step * config.accumulation_steps
    expected_epoch = (batches - 1) // batches_per_epoch if batches else 0
    expected_cursor = (
        min(size, ((batches - 1) % batches_per_epoch + 1) * config.batch_size) if batches else 0
    )
    if (
        epoch != expected_epoch
        or cursor != expected_cursor
        or metadata["windows_seen"] != epoch * size + cursor
    ):
        raise ValueError("Checkpoint sampler counters are inconsistent")
    path = directory / "training.safetensors"
    if (
        path.stat().st_size != metadata["training_bytes"]
        or path.stat().st_size > model.parameter_count * 12 + size * 8 + 1_000_000
        or file_hash(path) != metadata["training_sha256"]
    ):
        raise ValueError("Checkpoint training tensor size or checksum mismatch")
    tensors = load_file(str(path), device="cpu")
    order, rng = tensors["shuffle_order"], tensors["shuffle_rng"]
    if (
        order.dtype != torch.int64
        or order.shape != (size,)
        or not torch.equal(order.sort().values, torch.arange(size))
    ):
        raise ValueError("Invalid checkpoint shuffle permutation")
    if rng.dtype != torch.uint8 or rng.shape != trainer.stream.generator.get_state().shape:
        raise ValueError("Invalid checkpoint shuffle RNG")
    expected_tokens = epoch * sum(len(w) - 1 for w in train.windows) + sum(
        len(train.windows[i]) - 1 for i in order[:cursor].tolist()
    )
    if metadata["tokens_seen"] != expected_tokens:
        raise ValueError("Checkpoint token exposure counter is inconsistent")
    state = trainer.optimizer.state_dict()
    groups = state["param_groups"]
    lr = config.learning_rate_at(step) if step else config.learning_rate
    for group in groups:
        group["lr"] = lr
    if canonical_json(groups) != canonical_json(metadata["optimizer_groups"]):
        raise ValueError("Checkpoint optimizer settings mismatch")
    expected_keys = {"shuffle_order", "shuffle_rng"}
    if step:
        for saved_group, live_group in zip(groups, trainer.optimizer.param_groups, strict=True):
            for index, parameter in zip(saved_group["params"], live_group["params"], strict=True):
                slots = {}
                for name in ("step", "exp_avg", "exp_avg_sq"):
                    key = f"optimizer.{index}.{name}"
                    expected_keys.add(key)
                    tensor = tensors[key]
                    shape = () if name == "step" else parameter.shape
                    if (
                        tensor.shape != shape
                        or tensor.dtype != torch.float32
                        or not torch.isfinite(tensor).all().item()
                    ):
                        raise ValueError("Invalid optimizer tensor")
                    if name == "step" and tensor.item() != step:
                        raise ValueError("Optimizer step differs from training progress")
                    if name == "exp_avg_sq" and (tensor < 0).any().item():
                        raise ValueError("Optimizer second moment must be nonnegative")
                    slots[name] = tensor
                state["state"][index] = slots
    if set(tensors) != expected_keys:
        raise ValueError("Unexpected checkpoint tensor names")
    trainer.optimizer.load_state_dict(state)
    trainer.stream.order = order
    trainer.stream.generator.set_state(rng)
    trainer.stream.epoch, trainer.stream.cursor = epoch, cursor
    trainer.step = step
    trainer.tokens_seen = metadata["tokens_seen"]
    trainer.windows_seen = metadata["windows_seen"]
    return trainer
