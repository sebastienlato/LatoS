"""Small executable model checks, distinct from training or quality evaluation."""

import math

import torch

from latos.doctor import available_backends, select_device
from latos.model.config import ModelConfig
from latos.model.network import create_model


def check_model(config: ModelConfig, requested_device: str = "cpu") -> dict:
    device = select_device(requested_device, available_backends())
    model = create_model(config, seed=17).to(device)
    length = min(32, config.context_length)
    generator = torch.Generator(device="cpu").manual_seed(31)
    inputs = torch.randint(4, config.vocab_size, (2, length), generator=generator).to(device)
    output = model(inputs, labels=inputs)
    if (
        output.loss is None
        or not torch.isfinite(output.logits).all().item()
        or not torch.isfinite(output.loss).item()
    ):
        raise ValueError("Model produced nonfinite logits or loss")
    output.loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in model.parameters()):
        raise ValueError("Model has absent or nonfinite gradients")
    gradient_norm = math.sqrt(
        sum(float(p.grad.detach().float().square().sum().cpu()) for p in model.parameters())
    )
    if gradient_norm == 0:
        raise ValueError("Model gradient is zero")
    with torch.no_grad():
        baseline = model(inputs).logits
        changed = inputs.clone()
        changed[:, length // 2 :] = (changed[:, length // 2 :] + 1) % config.vocab_size
        difference = (
            (model(changed).logits[:, : length // 2] - baseline[:, : length // 2])
            .abs()
            .max()
            .item()
        )
    if difference > 1e-5:
        raise ValueError("Future input changed earlier model outputs")
    if model.parameter_count != config.parameter_count:
        raise ValueError("Analytical and actual parameter counts disagree")
    return {
        "status": "ok",
        "device": device,
        "dtype": "float32",
        "parameter_count": model.parameter_count,
        "logits_shape": list(output.logits.shape),
        "synthetic_loss": float(output.loss.detach().cpu()),
        "gradient_norm": gradient_norm,
        "causal_prefix_max_abs_difference": difference,
        "optimizer_steps": 0,
        "weights": "random initialization",
        "seed": 17,
    }
