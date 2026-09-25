"""Prospective execution policy and mandatory checkpoint records around unchanged native APIs."""

import gc
import importlib.util
import os
import platform
import random
import secrets
from pathlib import Path

from common import hash_file, read, record, require, verify_records, write_once


def fingerprints():
    path = Path(__file__).with_name("fingerprints.py")
    if not path.exists():
        path = path.parent.parent / "recovery-diagnostic/mechanics.py"
    spec = importlib.util.spec_from_file_location("phase171_fingerprints", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup(plan, output):
    require(os.name == "nt", "Actual Windows RTX execution required")
    p = plan["numerical_policy"]
    require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == p["CUBLAS_WORKSPACE_CONFIG"]
        and os.environ.get("PYTHONHASHSEED") == "0",
        "Process policy not set before imports",
    )
    for key in (
        "CUBLASLT_WORKSPACE_SIZE",
        "CUDNN_ERRATA_JSON_FILE",
        "NVIDIA_TF32_OVERRIDE",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "TORCH_CUDNN_V8_API_DISABLED",
        "CUDNN_CONV_WSCAP_DBG",
        "TORCH_CUBLAS_WORKSPACE_CACHE",
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
    ):
        require(not os.environ.get(key), f"Undeclared execution override: {key}")
    import numpy as np
    import safetensors
    import torch

    from latos.training.checkpoint import implementation_hash
    from latos.training.probe import cuda_environment

    require(
        platform.python_version() == p["python"]
        and str(torch.__version__) == p["torch"]
        and np.__version__ == p["numpy"]
        and safetensors.__version__ == p["safetensors"]
        and platform.platform() == p["platform"],
        "Pinned runtime differs",
    )
    require(
        implementation_hash() == plan["source"]["original_core_implementation_sha256"],
        "Core changed",
    )
    torch.set_num_threads(p["cpu_threads"])
    if torch.get_num_interop_threads() != p["interop_threads"]:
        torch.set_num_interop_threads(p["interop_threads"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    for name in ("flash", "mem_efficient", "math", "cudnn"):
        getattr(torch.backends.cuda, "enable_" + name + "_sdp")(True)
    environment = cuda_environment(output, min_free_bytes=0)
    require(
        environment["gpu"] == p["GPU"]
        and environment["capability"] == p["compute_capability"]
        and p["driver"] in (environment["driver_report"] or "")
        and environment["cuda_build"] == p["cuda_build"]
        and torch.backends.cudnn.version() == p["cudnn_version"],
        "Measured GPU/libraries differ",
    )
    check_controls(plan)
    return {
        "numerical_policy": p,
        "runtime": environment["runtime"],
        "gpu_total_bytes": environment["total_bytes"],
        "driver_report": environment["driver_report"],
    }


def check_controls(plan):
    import torch

    p = plan["numerical_policy"]
    observed = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "SDPA_enabled": {
            name: getattr(torch.backends.cuda, name + "_sdp_enabled")()
            for name in ("flash", "mem_efficient", "math", "cudnn")
        },
    }
    require(all(observed[k] == p[k] for k in observed), "Numerical controls changed")


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def native_inventory(directory):
    return {
        p.relative_to(directory).as_posix(): record(p)
        for p in directory.rglob("*")
        if p.is_file() and p.name != "policy.json"
    }


def save_bound(trainer, destination, binding, m):
    from latos.training.checkpoint import load_checkpoint, save_checkpoint

    require(not destination.exists(), "Checkpoint already exists")
    # Preserve any staging directory left by an interruption; never reuse it.
    pending = destination.with_name(destination.name + "." + secrets.token_hex(8) + ".pending")
    pending.mkdir(parents=True, exist_ok=False)
    payload = pending / "payload"
    before = m.snapshot(trainer)
    save_checkpoint(trainer, payload)
    restored = load_checkpoint(
        payload,
        trainer.train_data,
        trainer.validation_data,
        trainer.device,
        expected_config=trainer.config,
        expected_precision=trainer.precision,
        expected_single_pass=True,
    )
    after, loaded = m.snapshot(trainer), m.snapshot(restored)
    require(before == after == loaded, "Native checkpoint roundtrip changed full state")
    del restored
    gc.collect()
    policy = {
        "schema_version": 1,
        "binding": binding,
        "state": before,
        "native_files": native_inventory(payload),
        "roundtrip_exact": True,
    }
    write_once(payload / "policy.json", policy)
    verify_policy(payload, binding)
    payload.rename(destination)
    pending.rmdir()  # Only this now-empty staging directory is removed.
    return {
        "checkpoint": destination.name,
        "policy_sha256": hash_file(destination / "policy.json"),
        "state_sha256": before["state_sha256"],
        "roundtrip_exact": True,
    }


def verify_policy(directory, binding):
    saved = read(directory / "policy.json")
    require(
        saved["binding"] == binding and saved["roundtrip_exact"] is True,
        "Checkpoint policy/attempt/source/runtime mismatch",
    )
    require(
        native_inventory(directory) == saved["native_files"], "Incomplete or changed checkpoint"
    )
    verify_records(directory, saved["native_files"])
    return saved


def load_bound(directory, datasets, config, binding, m, device="cuda", precision="bfloat16"):
    from latos.training.checkpoint import load_checkpoint

    policy = verify_policy(directory, binding)
    trainer = load_checkpoint(
        directory,
        *datasets,
        device,
        expected_config=config,
        expected_precision=precision,
        expected_single_pass=True,
    )
    require(m.snapshot(trainer) == policy["state"], "Restored full state/RNG differs")
    return trainer


def fixture(plan, name):
    from dataclasses import replace

    from latos.model import ModelConfig
    from latos.training.config import TrainingConfig
    from latos.training.data import TokenDataset

    f = plan["adapter_preflight"]
    pair = next(p for p in f["pairs"] if p["name"] == name)
    rows = []
    for i in range(pair["windows"]):
        length = 2 + (29 * i) % 511
        rows.append(tuple([1, *[4 + ((7 * i + 3 * j) % 252) for j in range(1, length - 1)], 2]))
    import hashlib

    from common import canonical

    sha = hashlib.sha256(canonical(rows)).hexdigest()
    model = ModelConfig(**f["model_config"])
    train = TokenDataset(tuple(rows), 512, 260, "a" * 64, "train", sha)
    dummy = replace(train, split="validation")
    return model, TrainingConfig(**f["training"]), (train, dummy), pair["model_seed"], sha
