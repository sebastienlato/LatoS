# Dense decoder model

Phase 3 implements the LatoS transformer directly with PyTorch tensor operations
and modules. It supplies configuration, forward computation, next-token loss,
model-only snapshots, and a small sampler. All Phase 3 weights are random; no
optimizer steps or language training have run. A successful numerical check is
not evidence of useful generated English.

## Architecture

An input `[batch, time]` array of int64 token IDs enters a learned embedding table.
Each block applies an RMS normalization before causal multi-head attention, adds
the residual, then applies another RMS normalization and a SwiGLU feed-forward
layer followed by a second residual. A final RMS normalization feeds a projection
using the same embedding matrix. There is one shared input/output parameter,
no separate output matrix, and no biases or dropout.

- RMS normalization computes `x / sqrt(mean(x²) + eps)` per token, multiplied by
  a learned gain initialized to one. Float32 is the baseline; CPU float64 is used
  for independent numerical checks.
- Q, K, and V are projections of the normalized representation into equally sized
  heads. Rotary positions act on adjacent feature pairs of Q and K. For pair `i`
  at position `p`, the angle is `p / theta^(2i/head_dim)`. V is not rotated.
- Attention computes causal `softmax(QKᵀ / sqrt(head_dim)) V`. The implementation
  calls PyTorch SDPA with `is_causal=True` and zero dropout. This does not promise
  any particular fused kernel. Independent tests explicitly attend only to the
  current and preceding positions and compare both outputs and gradients.
- The feed-forward transformation is `W_down(silu(W_gate x) * W_up x)`.
- Linear and embedding weights begin as normal draws with standard deviation
  0.02. Attention output and feed-forward down projections instead use
  `0.02 / sqrt(2 * layers)` to limit initial residual magnitude.

These are established techniques, independently implemented for this project:
[attention](https://arxiv.org/abs/1706.03762),
[RMSNorm](https://arxiv.org/abs/1910.07467),
[rotary positions](https://arxiv.org/abs/2104.09864), and
[SwiGLU](https://arxiv.org/abs/2002.05202). The attention paper also discusses
sharing input/output embedding weights. No external model implementation or
pretrained weights were imported. See [THIRD_PARTY.md](../THIRD_PARTY.md).

## Configurations and resources

| Configuration | Vocabulary | Width | Heads | Layers | FFN width | Context | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Debug | 8,192 | 64 | 4 | 2 | 192 | 128 | 631,104 |
| Pilot | 8,192 | 384 | 6 | 8 | 1,024 | 512 | 17,308,032 |

The actual unique-parameter count is checked against
`V*D + layers*(4*D² + 3*D*F + 2*D) + D`. The pilot has 69,232,128 bytes of float32
parameters, excluding gradients, activations, optimizer state, tensor-file headers,
and allocator overhead. This is not a full training-memory estimate.

Versioned JSON configurations live in `configs/model/`. Validation requires even
head dimensions, finite numeric settings, bounded dimensions, and a tokenizer
SHA-256. Configurations exceeding 100 million parameters are rejected before
allocation. Bounds are guardrails for this local project, not hardware guarantees.
`create_model(config, seed)` explicitly initializes float32 parameters on CPU,
preserves the caller's CPU RNG state, and does not seed accelerator generators.

Both bundled configurations bind to the Phase 2 tokenizer:
`7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
Initialization and text sampling check that artifact hash and its actual vocabulary
size. Numeric model checks can run without the learned tokenizer using synthetic
IDs. A different tokenizer requires an explicitly matching model configuration.

## Forward and loss contract

`model(input_ids, labels=None)` returns `ModelOutput(logits, loss)` with logits
shaped `[batch, time, vocabulary]`. Inputs must be nonempty int64 tensors on the
model's device, with IDs in range and time within the configured context.
There is no automatic truncation or padding mask. Phase 3 accepts ordinary causal
sequences; padded or packed training semantics belong to the training-engine phase.

If labels are supplied, they have the same shape/device as inputs and are
**unshifted**: logits at position `t` predict labels at `t+1`. The final logit row
has no target in this objective. Label `-100` removes a target from the mean loss;
it does not remove an input from attention. Padding ID zero is not automatically
masked. No valid next-token targets, invalid IDs, wrong dtypes, or sequences
shorter than two positions cause an explicit error rather than a NaN loss.

Tests verify the loss with an independent log-sum-exp calculation and check that
masked/final logits receive no loss gradient. Causality tests change future tokens,
compare prefixes with shorter runs, and verify zero gradient into future input
representations. Batch members are independent.

## Commands

```sh
uv run --locked latos model inspect --config configs/model/pilot.json
uv run --locked latos model check --config configs/model/debug.json --device cpu
uv run --locked latos model check --config configs/model/pilot.json --device mps
uv run --locked latos model initialize --config configs/model/pilot.json --tokenizer-dir artifacts/tokenizers/english-bpe-v1 --output-dir checkpoints/phase-3-pilot-initial --seed 17
uv run --locked latos model sample --model-dir checkpoints/phase-3-pilot-initial --tokenizer-dir artifacts/tokenizers/english-bpe-v1 --prompt 'A quiet morning' --max-new-tokens 8 --temperature 0.8 --top-k 20 --seed 7 --device cpu
```

`check` performs forward/backward and a same-mode causal-prefix check on two
synthetic sequences of up to 32 positions. It performs no optimizer update and
does not open corpus files. Explicit unavailable devices fail. Use a fresh output
directory for another initialization. Sampling these random weights produces
arbitrary text, not a conversational response.

## Model-only snapshots

`save_model` writes float32 finite tensors in Safetensors format, canonical config
JSON, and metadata containing hashes, byte/parameter counts, software versions,
and tokenizer identity. It verifies all tensor values after reload before renaming
a temporary directory into place, and refuses an existing destination. The shared
embedding is serialized once. RoPE values are derived from the config at execution
time and add no learned parameters or persistent buffers.

`load_model` checks configuration and weights hashes, size bounds, tensor names,
shapes, float32 dtype, finite values, parameter count, and tokenizer identity. It
loads on CPU in evaluation mode. Call `.to(device)` and `.train()` explicitly as
needed. The format includes no Python object pickle and no optimizer, scheduler,
RNG-resume state, or training history. These are model-only snapshots, not resumable
training checkpoints; Phase 4 implements that separate requirement. Metadata and
hashes are not signatures against an adversary replacing all files together.

## Bounded sampling

Sampling handles one sequence, uses the whole available prefix each step, and
stops at EOS, the requested token count, or the context limit. It never silently
truncates a prompt or slides the context. Temperature zero chooses the highest
allowed logit. Positive temperature supports optional top-k; zero top-k means the
whole allowed vocabulary. PAD, BOS, and UNK are excluded from generated tokens;
EOS remains possible. A prompt already ending in EOS stops immediately.

Sampling uses a private CPU random generator. Logits transfer to CPU before
float64 probability computation, avoiding the observed MPS combined-transfer/cast
issue and supporting very small positive temperatures. The sampler restores the
model's train/eval mode even after an error and leaves the caller's logits and CPU
RNG unchanged. Reproducible choices are checked on the same backend/environment;
cross-device bitwise sampling equality is not promised.

There is no KV cache, streaming decoder, chat template, batch sampling, beam
search, or top-p in Phase 3. Arbitrary generated byte-token fragments may decode
with replacement characters. Float32 CPU and local MPS paths are tested; CUDA,
mixed precision, distributed execution, and compilation remain untested. SDPA
outputs may differ slightly across backends; tolerances and actual results are in
[the Phase 3 report](../experiments/phase-3/REPORT.md). Weights remain outside Git.
