"""Explicit CUDA BF16 autocast; float32 parameters and optimizer state are retained."""

from contextlib import nullcontext

import torch


def check_precision(device: str, precision: str) -> None:
    if precision not in ("float32", "bfloat16"):
        raise ValueError("Training precision must be float32 or bfloat16")
    if precision == "bfloat16" and (
        device != "cuda"
        or not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported(including_emulation=False)
    ):
        raise ValueError("BF16 training requires actual CUDA hardware with native BF16 support")


def autocast_context(device: str, precision: str):
    check_precision(device, precision)
    return (
        torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext()
    )


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
