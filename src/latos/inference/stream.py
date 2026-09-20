"""Bounded token events and append-only Unicode text for local inference."""

import math
import time
from collections.abc import Callable, Iterator

import torch
from tokenizers.decoders import DecodeStream

from latos.inference.cache import KVCache
from latos.model.network import LatoModel
from latos.model.sampling import sample_next
from latos.tokenization import LatoTokenizer


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class TextStream:
    """Buffer incomplete UTF-8 and flush unfinished bytes as batch-decoder replacements."""

    def __init__(self, codec: LatoTokenizer):
        self.codec = codec
        self.decoder = DecodeStream(skip_special_tokens=True)
        self.ids = []
        self.text = ""

    def push(self, token: int) -> str:
        self.ids.append(token)
        chunk = self.decoder.step(self.codec._engine, token) or ""
        self.text += chunk
        return chunk

    def finish(self) -> str:
        complete = self.codec.decode(self.ids)
        if not complete.startswith(self.text):
            raise RuntimeError("Streaming decoder disagrees with completed text")
        tail = complete[len(self.text) :]
        self.text = complete
        return tail


def stream_generate(
    model: LatoModel,
    codec: LatoTokenizer,
    prompt_ids: list[int],
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int = 0,
    cached: bool = True,
    cancelled: Callable[[], bool] | None = None,
) -> Iterator[dict]:
    """Yield token events, optional final text delta, then exactly one done event.

    A fresh cache belongs to each iterator. No inference-mode scope crosses a yield.
    Closing the iterator abandons it; callback cancellation emits a final event.
    Timings synchronize accelerators and exclude consumer/terminal delay and loading.
    """
    if any(m.training for m in model.modules()):
        raise ValueError("Streaming requires every model module in eval mode")
    if codec.vocab_size != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match model")
    if not prompt_ids or any(
        type(i) is not int or not 0 <= i < model.config.vocab_size for i in prompt_ids
    ):
        raise ValueError("Prompt must contain valid integer IDs")
    if len(prompt_ids) > model.config.context_length:
        raise ValueError("Prompt exceeds context limit; no silent truncation")
    if type(max_new_tokens) is not int or not 0 <= max_new_tokens <= model.config.context_length:
        raise ValueError("Invalid generation token budget")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Invalid sampling seed")
    if (
        not math.isfinite(temperature)
        or temperature < 0
        or type(top_k) is not int
        or not 0 <= top_k <= model.config.vocab_size
        or type(cached) is not bool
    ):
        raise ValueError("Invalid sampling parameters")
    tokens, generated = list(prompt_ids), []
    decoder = TextStream(codec)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = model.embedding.weight.device
    cache = KVCache(model) if cached else None
    times, forward_times = [], []
    reason = "max_new_tokens"
    if tokens[-1] == 2:
        reason = "eos"
    else:
        for _ in range(max_new_tokens):
            if cancelled is not None and cancelled():
                reason = "cancelled"
                break
            if len(tokens) >= model.config.context_length:
                reason = "context_limit"
                break
            synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                ids = tokens if cache is None or cache.length == 0 else tokens[-1:]
                inputs = torch.tensor([ids], dtype=torch.long, device=device)
                logits = model(inputs).logits if cache is None else cache.append(inputs)
                synchronize(device)
                forward_ms = (time.perf_counter() - start) * 1000
                token = sample_next(logits[0, -1], temperature, top_k, generator)
            chunk = decoder.push(token)
            times.append((time.perf_counter() - start) * 1000)
            forward_times.append(forward_ms)
            tokens.append(token)
            generated.append(token)
            # Classify the sampled boundary before yielding the token event.
            if token == 2:
                reason = "eos"
            elif len(tokens) >= model.config.context_length:
                reason = "context_limit"
            yield {"event": "token", "token_id": token, "text": chunk}
            if reason in ("eos", "context_limit"):
                break
    tail = decoder.finish()
    if tail:
        yield {"event": "text", "text": tail}
    # A signal may arrive during the last device operation or final text write.
    # Cancellation must still prevent that reply from entering conversation history.
    if cancelled is not None and cancelled():
        reason = "cancelled"
    yield {
        "event": "done",
        "generated_ids": generated,
        "text": decoder.text,
        "stop_reason": reason,
        "prompt_tokens": len(prompt_ids),
        "cached": cached,
        "prefill_ms": forward_times[0] if forward_times else None,
        "first_token_ms": times[0] if times else None,
        "subsequent_token_ms": times[1:],
        "cache_bytes": cache.nbytes if cache else 0,
    }
