"""Inspect this execution host and verify one backend with a tiny float32 calculation."""

import os
import platform
import shutil
from pathlib import Path

import torch

from latos import __version__


def available_backends() -> dict[str, bool]:
    """Report what the installed PyTorch runtime advertises, before execution."""
    return {
        "cpu": True,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }


def select_device(requested: str, available: dict[str, bool]) -> str:
    """Choose an advertised backend; never silently replace an explicit request."""
    if requested == "auto":
        return next(name for name in ("cuda", "mps", "cpu") if available[name])
    if requested not in available:
        raise ValueError(f"Unknown device: {requested}")
    if not available[requested]:
        raise ValueError(f"Requested device is unavailable: {requested}")
    return requested


def smoke_check(device: str) -> None:
    """Check matrix multiplication and its derivative against hand-calculated values."""
    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device, requires_grad=True
    )
    weights = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float32, device=device)
    y = x @ weights
    y.sum().backward()
    # Moving results to CPU also waits for accelerator work to complete.
    torch.testing.assert_close(
        y.detach().cpu(), torch.tensor([[2.0, 6.0], [6.0, 12.0]], dtype=torch.float32)
    )
    assert x.grad is not None
    torch.testing.assert_close(
        x.grad.cpu(), torch.tensor([[2.0, 3.0], [2.0, 3.0]], dtype=torch.float32)
    )


def physical_memory_bytes() -> int | None:
    """Return OS-reported physical memory, or null when this API is unavailable."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except AttributeError, OSError, ValueError:
        return None
    return pages * page_size if pages > 0 and page_size > 0 else None


def diagnose(requested: str = "auto") -> dict:
    """Return JSON-compatible evidence; availability alone is not a passing check."""
    available = available_backends()
    report = {
        "schema_version": 1,
        "latos_version": __version__,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "system": platform.system(),
        "os_version": platform.mac_ver()[0] or platform.release(),
        "architecture": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "physical_memory_bytes": physical_memory_bytes(),
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "available_backends": available,
        "mps_built": torch.backends.mps.is_built(),
        "cuda_build_version": torch.version.cuda,
        "mps_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        "requested_device": requested,
        "tested_device": None,
        "precision": "float32",
        "status": "error",
    }
    try:
        device = select_device(requested, available)
        report["tested_device"] = device
        smoke_check(device)
    except (ValueError, RuntimeError, AssertionError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    else:
        report["status"] = "ok"
    return report


def format_report(report: dict) -> str:
    """Render the same fields offered by JSON without paths or hostnames."""
    return "\n".join(f"{key}: {value}" for key, value in report.items())
