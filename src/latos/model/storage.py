"""Model-only float32 tensor snapshots; configuration and tokenizer identities are verified."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import safetensors
import torch
from safetensors.torch import load_file, save_file

from latos import __version__
from latos.data.manifest import canonical_json, sha256
from latos.model.config import ModelConfig
from latos.model.network import LatoModel, create_model
from latos.tokenization import LatoTokenizer


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_tokenizer(config: ModelConfig, directory: Path) -> LatoTokenizer:
    codec = LatoTokenizer.load(directory, expected_sha256=config.tokenizer_sha256)
    if codec.vocab_size != config.vocab_size:
        raise ValueError("Model vocabulary size does not match its tokenizer")
    return codec


def save_model(model: LatoModel, directory: Path) -> dict:
    if directory.exists():
        raise ValueError("Model output already exists; choose a new destination")
    tensors = {
        name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
    }
    if any(
        t.dtype != torch.float32 or not torch.isfinite(t).all().item() for t in tensors.values()
    ):
        raise ValueError("Model snapshots require finite float32 weights")
    if model.parameter_count != model.config.parameter_count:
        raise ValueError("Model parameters do not match its configuration")
    config_data = canonical_json(model.config.to_dict())
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".latos-model-", dir=directory.parent))
    try:
        save_file(tensors, str(staging / "model.safetensors"))
        (staging / "config.json").write_bytes(config_data)
        metadata = {
            "schema_version": 1,
            "kind": "latos-dense-v1-model-only",
            "latos_version": __version__,
            "torch_version": str(torch.__version__),
            "safetensors_version": safetensors.__version__,
            "dtype": "float32",
            "config_sha256": sha256(config_data),
            "weights_sha256": file_hash(staging / "model.safetensors"),
            "weights_bytes": (staging / "model.safetensors").stat().st_size,
            "parameter_count": model.parameter_count,
            "tokenizer_sha256": model.config.tokenizer_sha256,
            "training_state": "not included; no optimizer, scheduler, or run history",
        }
        (staging / "metadata.json").write_bytes(canonical_json(metadata))
        # Verify the complete tensor names/shapes/values before publishing the directory.
        restored = load_model(staging, expected_tokenizer_sha256=model.config.tokenizer_sha256)
        for name, value in restored.state_dict().items():
            if not torch.equal(value, tensors[name]):
                raise ValueError("Model weights changed during save/load")
        staging.rename(directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return metadata


def load_model(directory: Path, *, expected_tokenizer_sha256: str | None = None) -> LatoModel:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    config_data = (directory / "config.json").read_bytes()
    if metadata["schema_version"] != 1 or metadata["kind"] != "latos-dense-v1-model-only":
        raise ValueError("Unsupported model snapshot")
    if sha256(config_data) != metadata["config_sha256"]:
        raise ValueError("Model configuration checksum mismatch")
    config = ModelConfig(**json.loads(config_data))
    if metadata["tokenizer_sha256"] != config.tokenizer_sha256 or (
        expected_tokenizer_sha256 is not None
        and config.tokenizer_sha256 != expected_tokenizer_sha256
    ):
        raise ValueError("Model tokenizer identity mismatch")
    if metadata["parameter_count"] != config.parameter_count or metadata["dtype"] != "float32":
        raise ValueError("Model parameter count or dtype metadata mismatch")
    path = directory / "model.safetensors"
    size = path.stat().st_size
    if size != metadata["weights_bytes"] or size > config.parameter_count * 4 + 100_000:
        raise ValueError("Model tensor file size mismatch or budget exceeded")
    if file_hash(path) != metadata["weights_sha256"]:
        raise ValueError("Model weights checksum mismatch")
    try:
        tensors = load_file(str(path), device="cpu")
    except safetensors.SafetensorError as exc:
        raise ValueError("Invalid model tensor file") from exc
    model = create_model(config)
    expected = model.state_dict()
    if set(tensors) != set(expected):
        raise ValueError("Model tensor names do not match configuration")
    for name, tensor in tensors.items():
        if (
            tensor.shape != expected[name].shape
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"Invalid model tensor: {name}")
    model.load_state_dict(tensors, strict=True)
    return model.eval()
