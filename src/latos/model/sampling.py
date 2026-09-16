"""Bounded, single-sequence generation without a KV cache."""

import math

import torch

from latos.model.network import LatoModel


def sample_next(
    logits: torch.Tensor, temperature: float, top_k: int, generator: torch.Generator
) -> int:
    # Transfer first: a combined MPS->CPU float64 cast can corrupt finite values.
    scores = logits.detach().cpu().to(dtype=torch.float64, copy=True)
    if scores.ndim != 1 or scores.numel() < 4 or not torch.isfinite(scores).all().item():
        raise ValueError("Sampling requires a finite vector of logits")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative")
    if type(top_k) is not int or not 0 <= top_k <= scores.numel():
        raise ValueError("top_k must be zero or within the vocabulary")
    # PAD, BOS, and UNK cannot be generated; EOS remains available.
    scores[[0, 1, 3]] = -torch.inf
    if temperature == 0:
        return int(scores.argmax())
    # Subtract before division to avoid overflow from large logits/small temperature.
    scores = (scores - scores.max()) / temperature
    if top_k:
        values, indices = scores.topk(top_k)
        choice = torch.multinomial(values.softmax(dim=-1), 1, generator=generator)
        return int(indices[choice])
    return int(torch.multinomial(scores.softmax(dim=-1), 1, generator=generator))


def generate(
    model: LatoModel,
    prompt_ids: list[int],
    *,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int = 0,
) -> dict:
    if not prompt_ids or any(
        type(i) is not int or not 0 <= i < model.config.vocab_size for i in prompt_ids
    ):
        raise ValueError("Prompt must contain valid integer IDs")
    if len(prompt_ids) > model.config.context_length:
        raise ValueError("Prompt exceeds model context; no silent truncation")
    if type(max_new_tokens) is not int or not 0 <= max_new_tokens <= model.config.context_length:
        raise ValueError("Invalid generation token budget")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Invalid sampling seed")
    if (
        not math.isfinite(temperature)
        or temperature < 0
        or type(top_k) is not int
        or not 0 <= top_k <= model.config.vocab_size
    ):
        raise ValueError("Invalid sampling parameters")
    tokens = list(prompt_ids)
    new_tokens = []
    stop_reason = "max_new_tokens"
    if tokens[-1] == 2:
        return {"ids": tokens, "generated_ids": [], "stop_reason": "eos"}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                if len(tokens) >= model.config.context_length:
                    stop_reason = "context_limit"
                    break
                inputs = torch.tensor(
                    [tokens], dtype=torch.long, device=model.embedding.weight.device
                )
                token = sample_next(model(inputs).logits[0, -1], temperature, top_k, generator)
                tokens.append(token)
                new_tokens.append(token)
                if token == 2:
                    stop_reason = "eos"
                    break
    finally:
        model.train(was_training)
    return {"ids": tokens, "generated_ids": new_tokens, "stop_reason": stop_reason}
