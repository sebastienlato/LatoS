# Phase 3 transformer evidence

Recorded 2026-09-16 on the macOS 26.6.2 Apple M4 Max execution host (64 GiB RAM).
LatoS 0.4.0, Python 3.14.7, PyTorch 2.14.0, Safetensors 0.8.0.
The source checkpoint is the Phase 3 commit containing this report. Concatenated
model Python files (filename-sorted) have SHA-256 `98147eeb8f14b1a143a9b156e19b8173d0c5ed66e6ee757688c6044330ec75ca`.
MPS was available and the CPU fallback environment flag was unset.

## Implementation and measured sizes

Original dense decoder implementation: pre-RMSNorm residual blocks, rotary
adjacent-pair Q/K positions, causal multi-head SDPA, SwiGLU, final RMSNorm, and
shared input/output embedding weights. No pretrained model code or weights.

| Configuration | Layers / width / heads / FFN | Context | Actual parameters | Float32 parameter bytes |
| --- | --- | ---: | ---: | ---: |
| Debug | 2 / 64 / 4 / 192 | 128 | 631,104 | 2,524,416 |
| Pilot | 8 / 384 / 6 / 1,024 | 512 | 17,308,032 | 69,232,128 |

Both use the 8,192-entry Phase 2 vocabulary and bind its exact artifact hash.
The analytical formula agrees with the actual unique parameter count. Parameter
bytes exclude activations, gradients, optimizer state, allocator usage, and file
headers. No optimizer steps or model training were performed.

## Numerical checks

118 tests passed locally. The model tests add independent scalar RMSNorm and
rotary equations, finite-difference gradient checks, explicit causal attention
and its derivatives, direct SwiGLU comparison, next-token loss alignment and
masking, zero gradient into future input representations, batch independence,
configuration/input rejection, shared embedding verification, seeded initialization,
corrupt or shape-mismatched snapshots, tokenizer binding, and sampler behavior.
The same 118 tests also passed in a fresh locked environment using the built wheel
outside the source directory; that wheel loaded the pilot snapshot and produced
finite logits. Lint, format, dependency compatibility, archive contents, and
workflow YAML checks passed. The MPS sampling test is explicitly skipped on hosts
without that hardware.

CPU float64 attention results match the explicit reference at absolute/relative
output tolerances 1e-11/1e-10; input/weight gradients use 1e-10/1e-9. Float32 model
prefix/batch tests use 1e-6 absolute and relative tolerances. No mixed precision
or CUDA path was validated.

Both debug and pilot ran forward, loss, and backward on CPU and MPS, with finite
nonzero gradients, `[2, 32, 8192]` logits, and zero measured causal-prefix difference
when both comparisons used the same inference mode. Diagnostic seed was 17.
Detailed results are in [debug CPU](debug-cpu.json), [debug MPS](debug-mps.json),
[pilot CPU](pilot-cpu.json), and [pilot MPS](pilot-mps.json). A further check at the pilot's full 512-position
context produced `[1, 512, 8192]` finite logits and finite backward gradients on
both CPU and MPS, with zero optimizer steps; see [full-context.json](full-context.json).

A separate identical-weight debug comparison (seed 29) measured CPU/MPS maximum
absolute differences of **2.3841858e-7** in logits, **9.5367432e-7** in loss, and
**4.6193600e-7** in parameter gradients, below the 2e-5 check tolerance.
See [backend-comparison.json](backend-comparison.json). These results do not imply
cross-device bitwise reproducibility or equivalence for every configuration.

The final pilot diagnostic, including process startup, construction, forward/backward,
and two inference passes, took **0.79 seconds on CPU** and **0.98 seconds on MPS**.
macOS `/usr/bin/time -l` reported maximum resident set sizes of **440,516,608** and
**458,604,544 bytes**, respectively. These are single short process measurements,
not sustained training throughput, full-training memory estimates, or GPU peak
allocation measurements. The small test does not establish which backend is faster
for training at useful batch/context sizes.

## Snapshots and sampling

An untrained pilot initialized with seed 17 was saved to the ignored directory
`checkpoints/phase-3-pilot-initial/`. Every tensor survived save/load exactly; its
CPU logits also reproduced bit for bit. [snapshot-roundtrip.json](snapshot-roundtrip.json)
records that result. Snapshot metadata is in [initial-snapshot.json](initial-snapshot.json).

- Weight file: 69,237,768 bytes.
- Weight SHA-256: `99dce7bf0356bf7d641dff4b29d6616772dfbf9779877e0c085bdea9e368b5f2`.
- Config SHA-256: `c9057cd643303d5a2c5ebfd4c069f06771b72a0f7bb5329bf8fdd22c3c564372`.
- Bound tokenizer SHA-256: `7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.

The sampler produced eight bounded tokens on CPU and MPS from the fixed prompt
“A quiet morning”, seed 7, temperature 0.8, top-k 20. Both sampled the same IDs in
this check; no general cross-device equality is promised. See [sampling.json](sampling.json).
Generated text from random weights is arbitrary and provides no language-quality
result. Unit tests also verify greedy selection, allowed-token filtering, EOS,
context stopping, zero-token requests, deterministic local RNG, and mode restoration.

## Separate review and fixes

Review found a real MPS sampler failure: simultaneous device transfer and float64
conversion produced nonfinite CPU values from finite MPS logits. Copying to CPU
before conversion fixed it; an MPS-specific regression and the full pilot sampling
command passed afterward. Float64 CPU probability arithmetic also avoids NaNs for
extremely small positive temperatures. Sampling uses a copy, preserving input logits.

The causal diagnostic now compares two forwards in the same no-gradient mode;
comparing grad-enabled and no-gradient kernels had introduced rounding differences.
Initialization explicitly uses float32 and a CPU-only generator inside a restored
RNG context, avoiding dependence on the caller's default dtype or accelerator seeds.
These behaviors have regression coverage. No actionable review findings remain.

## Scope and next action

The Phase 2 checkpoint was approved and verified at
`1ed47127c05d8d6de0095f688fd3ad5f24c62b48`; its
[Linux CI](https://github.com/sebastienlato/LatoS/actions/runs/35149689542) passed
85 tests plus the offline tokenizer workflow. Phase 3 Linux CI awaits publication.

Weights and learned tokenizer files remain outside Git. Snapshots contain only
model tensors/configuration, not optimizer or resumable training state. There is
no padding-attention mask, KV cache, streaming, or chat interface. Model quality,
training throughput, context extension, mixed precision, and CUDA remain
unestablished. After this checkpoint is approved and published, Phase 4 implements
the training engine, including tiny-fixture overfit and resume equivalence.
