"""Session-owned, append-only KV state; no cache enters a model state_dict."""

import torch
from torch.nn import functional as F

from latos.model.network import LatoModel, rotary


class KVCache:
    """Single batch, fixed eval model/device/dtype; reset before a new sequence.

    Use one instance per caller. Do not mutate weights while using this object.
    Normal PyTorch parameter updates/device moves are detected; unsupported .data
    edits are outside the contract. Failed appends leave the previous state intact.
    """

    def __init__(self, model: LatoModel):
        self.model = model
        self.reset()

    def _signature(self):
        return tuple((id(p), p._version, p.device, p.dtype) for p in self.model.parameters())

    def reset(self):
        self._layers = []
        self.length = 0
        self._batch = None
        self._weights = self._signature()

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for pair in self._layers for t in pair)

    @torch.inference_mode()
    def append(self, ids: torch.Tensor) -> torch.Tensor:
        """Return all new-token logits; positions and rectangular mask are explicit."""
        model = self.model
        if any(module.training for module in model.modules()):
            raise ValueError("KV inference requires every model module in eval mode")
        if self._weights != self._signature():
            raise ValueError("Model changed; reset the KV cache")
        if (
            ids.ndim != 2
            or ids.shape[0] < 1
            or ids.shape[1] < 1
            or self.length + ids.shape[1] > model.config.context_length
        ):
            raise ValueError("Invalid cache input shape or context limit")
        if ids.dtype != torch.long or ids.device != model.embedding.weight.device:
            raise ValueError("Cache IDs must be int64 on the model device")
        if ((ids < 0) | (ids >= model.config.vocab_size)).any().item():
            raise ValueError("Invalid cache token ID")
        batch, count = ids.shape
        if self._batch is not None and batch != self._batch:
            raise ValueError("Cache batch changed; reset first")
        # SDPA's is_causal=True uses upper-left alignment for rectangular inputs.
        # New queries must instead attend through their absolute prefix positions.
        mask = None
        if self.length:
            queries = torch.arange(self.length, self.length + count, device=ids.device)
            keys = torch.arange(self.length + count, device=ids.device)
            mask = keys[None, :] <= queries[:, None]
        x = model.embedding(ids)
        layers = []
        for index, block in enumerate(model.blocks):
            attention = block.attention
            normalized = block.attention_norm(x)
            projected = attention.qkv(normalized).reshape(
                batch, count, 3, attention.heads, attention.head_dim
            )
            q, k, v = projected.permute(2, 0, 3, 1, 4).unbind(0)
            q, k = rotary(q, k, attention.theta, offset=self.length)
            if self.length:
                old_k, old_v = self._layers[index]
                k, v = torch.cat((old_k, k), dim=2), torch.cat((old_v, v), dim=2)
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=not self.length
            )
            x = x + attention.output(attended.transpose(1, 2).reshape(batch, count, -1))
            x = x + block.ffn(block.ffn_norm(x))
            # Clone the initial V view so unused projected Q/K storage is released.
            layers.append((k.contiguous(), v.clone() if not self.length else v))
        logits = F.linear(model.final_norm(x), model.embedding.weight)
        self._layers = layers
        self._batch = batch
        self.length += count
        return logits
