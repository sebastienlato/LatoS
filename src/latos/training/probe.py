"""Bounded CUDA evidence collection. Never selects or launches Phase 17 training."""

import argparse
import copy
import gc
import json
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from latos.data2.acquire import write_json
from latos.data2.resources import peak_rss_bytes
from latos.inference.cache import KVCache
from latos.model import ModelConfig, create_model
from latos.model.storage import file_hash
from latos.training.checkpoint import (
    load_checkpoint,
    runtime_identity,
    save_checkpoint,
    source_identity,
)
from latos.training.config import TrainingConfig
from latos.training.data import TokenDataset, collate
from latos.training.data2 import prepare_probe_dataset
from latos.training.engine import Trainer
from latos.training.precision import autocast_context, check_precision, synchronize

BUNDLE_SHA256 = "5f4fa7a7bc221d06f6b321c4af712456d3188984d759f8a4a898651d5a6d9073"
TOKENIZER_SHA256 = "23b9182d5943e7f8b32c6f415f108b7d06f229c80f5cd8f236258db40a6b94c4"
LAYERS = (8, 14, 20)


def verify_inputs(package: Path) -> dict:
    if file_hash(package / "bundle.json") != BUNDLE_SHA256:
        raise ValueError("Expected the exact accepted Phase 15 input package")
    bundle = json.loads((package / "bundle.json").read_text())
    for name, record in bundle["files"].items():
        path = package / name
        if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
            raise ValueError(f"Input package integrity mismatch: {name}")
    actual = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()}
    if actual != set(bundle["files"]) | {"bundle.json"}:
        raise ValueError("Unexpected or missing input package files")
    return {"bundle_sha256": BUNDLE_SHA256, "verified_files": len(bundle["files"])}


def host_peak_bytes():
    if sys.platform != "win32":
        return peak_rss_bytes()
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in ("peak", "working", "peak_paged", "paged", "peak_np", "np", "page", "pp")
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    get_info = psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not get_info(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return counters.peak


def cuda_environment(output: Path) -> dict:
    check_precision("cuda", "bfloat16")
    properties = torch.cuda.get_device_properties(0)
    if "RTX 4070 SUPER" not in properties.name or properties.total_memory < 10 * 1024**3:
        raise ValueError("This protocol requires the actual RTX 4070 SUPER with >=10 GiB VRAM")
    free, total = torch.cuda.mem_get_info()
    if shutil.disk_usage(output).free < 20 * 1024**3:
        raise ValueError("Probe suite requires at least 20 GiB free disk")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
    except OSError, subprocess.SubprocessError:
        driver = None
    return {
        "runtime": runtime_identity("cuda"),
        "cuda_build": torch.version.cuda,
        "gpu": properties.name,
        "capability": list(torch.cuda.get_device_capability()),
        "total_bytes": total,
        "free_bytes_at_start": free,
        "driver_report": driver,
        "native_bfloat16": True,
        "disk_free_bytes": shutil.disk_usage(output).free,
    }


def memory() -> dict:
    synchronize("cuda")
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "host_peak_working_set_bytes": host_peak_bytes(),
    }


def enforce_headroom(total: int):
    if torch.cuda.max_memory_reserved() > total * 0.85:
        raise ValueError("Measured CUDA reserved memory exceeds 85% headroom policy")


def relative_l2(left, right) -> float:
    numerator, denominator = 0.0, 0.0
    for a, b in zip(left, right, strict=True):
        a, b = a.detach().double().cpu(), b.detach().double().cpu()
        numerator += (a - b).square().sum().item()
        denominator += a.square().sum().item()
    return (numerator / max(denominator, 1e-30)) ** 0.5


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("State keys differ")
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    elif left != right:
        raise ValueError("State values differ")


def numerical_control(output: Path, device="cuda", precision="bfloat16") -> dict:
    """Synthetic mechanics only; CPU float32 mode exists solely for local tests."""
    cfg = ModelConfig(1, 260, 32, 64, 4, 2, 176, 10000.0, 1e-5, "a" * 64)
    rows = ((1, 8, 9, 10, 2), (1, 11, 2), (1, 12, 13, 14, 15, 2), (1, 16, 2))
    train = TokenDataset(rows, 32, 260, "a" * 64, "train", "b" * 64)
    dev = replace(train, windows=((1, 17, 18, 2),), split="validation", source_sha256="c" * 64)
    config = TrainingConfig(sequence_length=32, max_steps=4, warmup_steps=1, learning_rate=0.0003)
    reference = Trainer(create_model(cfg, 160), config, train, dev, device)
    candidate = Trainer(create_model(cfg, 160), config, train, dev, device, precision=precision)
    ids, labels, _ = collate(train, [0, 1, 2], device)
    losses, gradients = [], []
    for trainer in (reference, candidate):
        with autocast_context(device, trainer.precision):
            loss = trainer.model(ids, labels).loss
        loss.backward()
        gradients.append([p.grad.detach().cpu().clone() for p in trainer.model.parameters()])
        losses.append(loss.item())
        trainer.optimizer.zero_grad(set_to_none=True)
    loss_delta = abs(losses[0] - losses[1])
    grad_delta = relative_l2(*gradients)
    if not loss_delta <= 0.03 or not grad_delta <= 0.05:
        raise ValueError(f"Precision reference tolerance failed: {loss_delta=}, {grad_delta=}")
    # Variable-length token weighting must agree with a larger physical batch.
    accumulated = Trainer(
        create_model(cfg, 160),
        replace(config, batch_size=1, accumulation_steps=3),
        train,
        dev,
        device,
        precision=precision,
    )
    combined = Trainer(
        create_model(cfg, 160),
        replace(config, batch_size=3, accumulation_steps=1),
        train,
        dev,
        device,
        precision=precision,
    )
    ma, mb = accumulated.update(), combined.update()
    if ma["targets"] != mb["targets"] or abs(ma["loss"] - mb["loss"]) > 0.03:
        raise ValueError("Precision accumulation mismatch")
    accumulation_delta = relative_l2(combined.model.parameters(), accumulated.model.parameters())
    if accumulation_delta > 0.03:
        raise ValueError("Accumulated update differs from combined batch")
    del accumulated, combined
    for _ in range(2):
        reference.update()
        candidate.update()
    before = copy.deepcopy(candidate.model.state_dict())
    state_before = copy.deepcopy(candidate.optimizer.state_dict())
    rng_before = candidate.stream.generator.get_state().clone()
    candidate.validate()
    assert_state_equal(before, candidate.model.state_dict())
    assert_state_equal(state_before, candidate.optimizer.state_dict())
    if not torch.equal(rng_before, candidate.stream.generator.get_state()):
        raise ValueError("Validation changed sampler RNG")
    save_checkpoint(candidate, output / "recovery")
    restored = load_checkpoint(
        output / "recovery", train, dev, device, expected_precision=precision
    )
    for _ in range(2):
        reference.update()
        a, b = candidate.update(), restored.update()
        if a["targets"] != b["targets"] or abs(a["loss"] - b["loss"]) > 1e-5:
            raise ValueError("Resume trajectory mismatch")
    assert_state_equal(candidate.model.state_dict(), restored.model.state_dict())
    assert_state_equal(candidate.optimizer.state_dict(), restored.optimizer.state_dict())
    if (
        candidate.tokens_seen != restored.tokens_seen
        or candidate.stream.cursor != restored.stream.cursor
        or candidate.stream.epoch != restored.stream.epoch
        or not torch.equal(candidate.stream.order, restored.stream.order)
        or not torch.equal(
            candidate.stream.generator.get_state(), restored.stream.generator.get_state()
        )
    ):
        raise ValueError("Resume counters/order/RNG differ")
    weight_delta = relative_l2(reference.model.parameters(), candidate.model.parameters())
    if not weight_delta <= 0.03:
        raise ValueError("Four-update precision weight tolerance failed")
    # A nonfinite update must fail closed and leave the saved recovery point usable.
    restored.config = replace(config, max_steps=5)
    with torch.no_grad():
        restored.model.embedding.weight.fill_(float("nan"))
    try:
        restored.update()
    except ValueError, RuntimeError:
        if restored.ready:
            raise ValueError("Failed update remained checkpointable") from None
    else:
        raise ValueError("Nonfinite update unexpectedly succeeded")
    load_checkpoint(output / "recovery", train, dev, device, expected_precision=precision)
    return {
        "status": "passed",
        "scope": "original synthetic numerical mechanics only",
        "device": device,
        "precision": precision,
        "loss_absolute_difference": loss_delta,
        "gradient_relative_l2": grad_delta,
        "four_update_weight_relative_l2": weight_delta,
        "accumulation_weight_relative_l2": accumulation_delta,
        "resume_and_failure_recovery": "passed",
        "validation_preserves_state": True,
    }


def inference_measure(model) -> dict:
    model.eval()
    prompt = torch.tensor([[1] + [4 + i % 200 for i in range(127)]], device="cuda")
    results = []
    for _ in range(4):
        cache = KVCache(model)
        synchronize("cuda")
        start = time.perf_counter()
        logits = cache.append(prompt)
        token = logits[:, -1].argmax(-1, keepdim=True)
        synchronize("cuda")
        first = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(63):
            logits = cache.append(token)
            token = logits[:, -1].argmax(-1, keepdim=True)
        synchronize("cuda")
        results.append(
            {
                "first_token_seconds": first,
                "decode_tokens_per_second": 63 / (time.perf_counter() - start),
            }
        )
    return {
        "scope": "float32 native KV cache; synthetic 128-ID prompt; 64 forced tokens; EOS ignored",
        "warmup": results[0],
        "repeats": results[1:],
        "median_first_token_seconds": statistics.median(
            r["first_token_seconds"] for r in results[1:]
        ),
        "median_decode_tokens_per_second": statistics.median(
            r["decode_tokens_per_second"] for r in results[1:]
        ),
    }


def measure_candidate(package: Path, output: Path, layers: int, precision: str, env: dict) -> dict:
    verified = verify_inputs(package)
    start = time.perf_counter()
    train, train_info = prepare_probe_dataset(
        package / "corpus", package / "tokenizer", "train", 512, max_documents=4096
    )
    dev, dev_info = prepare_probe_dataset(
        package / "corpus", package / "tokenizer", "development", 512, max_documents=128
    )
    if set(train_info["groups"]) & set(dev_info["groups"]):
        raise ValueError("Probe group overlap")
    preparation_seconds = time.perf_counter() - start
    write_json(output / "data.json", {"train": train_info, "development": dev_info})
    cfg = ModelConfig(1, 16384, 512, 512, 8, layers, 1408, 10000.0, 1e-5, TOKENIZER_SHA256)
    steps = 8 if precision == "float32" else 64
    config = TrainingConfig(
        sequence_length=512,
        batch_size=2,
        accumulation_steps=8,
        max_steps=steps,
        warmup_steps=4,
        learning_rate=0.0003,
        seed=160,
    )
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer = Trainer(create_model(cfg, 160), config, train, dev, "cuda", precision=precision)
    if trainer.model.parameter_count != cfg.parameter_count:
        raise ValueError("Native parameter count differs")
    write_json(
        output / "run.json",
        {
            "environment": env,
            "source": source_identity(),
            "model": cfg.to_dict(),
            "training": config.to_dict(),
            "precision": precision,
            "datasets": trainer.data_identities,
            "inputs": verified,
        },
    )
    timings = {}

    def timed(name, function):
        synchronize("cuda")
        begin = time.perf_counter()
        result = function()
        synchronize("cuda")
        timings[name] = time.perf_counter() - begin
        enforce_headroom(env["total_bytes"])
        return result

    initial = timed("initial_validation", trainer.validate)
    timed("initial_checkpoint", lambda t=trainer: save_checkpoint(t, output / "initial"))
    metrics = []
    with (output / "updates.jsonl").open("x", encoding="utf-8") as log:
        for _ in range(steps):
            record = trainer.update()
            record.update(memory())
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            metrics.append(record)
            enforce_headroom(env["total_bytes"])
            if trainer.step == steps // 2:
                timed(
                    "midpoint_checkpoint", lambda t=trainer: save_checkpoint(t, output / "midpoint")
                )
                before_resume = timed("midpoint_validation", trainer.validate)
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
                trainer = timed(
                    "midpoint_reload",
                    lambda: load_checkpoint(
                        output / "midpoint",
                        train,
                        dev,
                        "cuda",
                        expected_config=config,
                        expected_precision=precision,
                    ),
                )
                after_resume = timed("reloaded_validation", trainer.validate)
                if abs(before_resume["loss"] - after_resume["loss"]) > 1e-5:
                    raise ValueError("Real-data midpoint recovery changed validation")
    final = timed("final_validation", trainer.validate)
    timed("final_checkpoint", lambda t=trainer: save_checkpoint(t, output / "final"))
    resources = memory()
    # Free optimizer/gradient storage for a separate inference measurement.
    model = trainer.model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    inference = timed("inference", lambda: inference_measure(model))
    resources.update(memory())
    measured = metrics[2:]
    checkpoint_bytes = sum(p.stat().st_size for p in (output / "final").rglob("*") if p.is_file())
    return {
        "status": "measured",
        "phase16_complete": False,
        "selected_configuration": None,
        "layers": layers,
        "precision": precision,
        "parameters": cfg.parameter_count,
        "initial_development": initial,
        "final_development": final,
        "inference": inference,
        "data_preparation_seconds": preparation_seconds,
        "execution_seconds": time.perf_counter() - start,
        "timings": timings,
        "resources": resources,
        "checkpoint_bytes": checkpoint_bytes,
        "tokens_seen": sum(m["targets"] for m in metrics),
        "measured_targets_per_second": sum(m["targets"] for m in measured)
        / sum(m["seconds"] for m in measured),
        "throughput_excluded_warmup_updates": 2,
        "note": "Short feasibility probe, not accepted base training or a learned-quality gate.",
    }


def inventory(output: Path):
    write_json(
        output / "inventory.json",
        {
            p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": file_hash(p)}
            for p in sorted(output.rglob("*"))
            if p.is_file() and p != output / "inventory.json"
        },
    )


def snapshot_source(output: Path):
    """Retain the actually imported original package; no workspace/private scan."""
    package = Path(__file__).resolve().parents[1]
    for path in sorted(package.rglob("*.py")):
        destination = output / "source" / "latos" / path.relative_to(package)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())


def record_failure(output: Path, exc: Exception):
    evidence = {
        "type": type(exc).__name__,
        "message": str(exc),
        "host_peak_working_set_bytes": host_peak_bytes(),
    }
    if torch.cuda.is_available():
        try:
            evidence["cuda"] = memory()
        except RuntimeError:
            evidence["cuda"] = "unavailable after device failure"
    write_json(output / "failure.json", evidence)


def worker(args):
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        snapshot_source(args.output)
        env = cuda_environment(args.output)
        write_json(args.output / "environment.json", env)
        if args.worker == "control":
            result = numerical_control(args.output)
        else:
            result = measure_candidate(args.inputs, args.output, args.layers, args.precision, env)
        write_json(args.output / "result.json", result)
    except Exception as exc:
        record_failure(args.output, exc)
        raise
    finally:
        inventory(args.output)


def suite(args):
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        snapshot_source(args.output)
        write_json(args.output / "inputs.json", verify_inputs(args.inputs))
        write_json(args.output / "environment.json", cuda_environment(args.output))
        jobs = [("control", None, None, 600)] + [
            ("measure", layers, precision, 1800)
            for layers in LAYERS
            for precision in ("float32", "bfloat16")
        ]
        for kind, layers, precision, timeout in jobs:
            name = "control" if kind == "control" else f"layers-{layers}-{precision}"
            command = [
                sys.executable,
                "-m",
                "latos.training.probe",
                "--worker",
                kind,
                "--inputs",
                str(args.inputs.resolve()),
                "--output",
                str((args.output / name).resolve()),
            ]
            if layers is not None:
                command += ["--layers", str(layers), "--precision", precision]
            with (args.output / f"{name}.log").open("x", encoding="utf-8") as log:
                try:
                    completed = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
                    )
                    result = {"job": name, "returncode": completed.returncode, "timeout": False}
                except subprocess.TimeoutExpired:
                    result = {"job": name, "returncode": None, "timeout": True}
            results.append(result)
            write_json(args.output / "progress.json", results)
            if kind == "control" and result["returncode"] != 0:
                raise ValueError("CUDA numerical controls failed; candidate suite not started")
        write_json(
            args.output / "summary.json",
            {
                "status": "awaiting external-evidence review",
                "jobs": results,
                "phase16_complete": False,
                "selected_configuration": None,
                "phase17_started": False,
            },
        )
    except Exception as exc:
        record_failure(args.output, exc)
        raise
    finally:
        inventory(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("control", "measure"))
    parser.add_argument("--layers", type=int, choices=LAYERS)
    parser.add_argument("--precision", choices=("float32", "bfloat16"))
    args = parser.parse_args()
    if args.worker == "measure" and (args.layers is None or args.precision is None):
        parser.error("measure requires --layers and --precision")
    worker(args) if args.worker else suite(args)


if __name__ == "__main__":
    main()
