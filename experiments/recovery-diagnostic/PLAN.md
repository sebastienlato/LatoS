# Recovery-mechanics diagnostic — prospective Phase 17.1 support

The owner explicitly authorized this investigation after Phase 17 closure was
published at `09c1935b085218fbbc9ea88703d8ba12e63c516e`. Phase 17 remains permanently
interrupted/failed. Its `1e-5` criterion and all historical results remain unchanged.
This diagnostic does not reopen that attempt, score/promote its recovered model or
begin Phase 17.1. No publication, paid resource or Phase 18 is authorized.

## Question and bounded hypotheses

Distinguish loss or omission of saved model/optimizer/sampler state from numerical
execution differences after exact restoration. The historical scalar trace first
differs at update 7,124 (loss delta 4.771905e-6 and differing gradient norm), then
exceeds the frozen criterion at 7,125. This timing alone does not identify the cause.
No old tensor checkpoint is loaded by this diagnostic.

Two profiles are fixed before CUDA optimization:

| Profile | Deterministic algorithms | cuDNN deterministic | cuBLAS workspace |
| --- | --- | --- | --- |
| legacy control | false | false | unset |
| prospective correction | true, errors rather than warnings | true | `:4096:8` |

Both retain native CUDA BF16, float32 weights/gradients/AdamW moments, TF32 off,
highest float32 matmul precision, four CPU threads, cuDNN benchmarking off, all
native SDPA backends enabled, no compilation/checkpointed activations/fused optimizer.
The original imported LatoS package remains byte-identical (implementation SHA-256
`76cf4a198ee6acde566fdcc70c9420c0cae8ab6cc0d6b4d7e4c9c0cba5c8f09c`). Only the separately
recorded process policy differs. Neither policy migrates the failed checkpoint.

PyTorch documents deterministic alternatives/error behavior, including CUDA
embedding differentiation, and cautions that the switch alone is insufficient:
[deterministic algorithms](https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html),
[reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
Its [SDPA documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
notes backend-dependent numerics and cuDNN determinism controls.
[NVIDIA cuBLAS reproducibility](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html#results-reproducibility)
documents the workspace setting. These sources motivate a prospective hypothesis;
they do not establish the historical cause or guarantee these comparisons will pass.
The control explicitly pins defaults that were incompletely recorded historically;
it is not claimed to recreate every historical library/allocator cache condition.

## Inputs and comparison design

Use the exact selected 34,087,424-parameter, eight-layer, width-512, context-512,
16,384-vocabulary native model. Each profile starts a fresh random model at seed
17101. No historical model, optimizer or learned token payload is loaded. Token IDs
are original synthetic mixtures of repeated motifs/common/rare IDs, seeded at
17102. The selected tokenizer hash identifies the model's ID space; no tokenizer
artifact is loaded or decoded. This is numerical mechanics, not language learning
or model acceptance.

The only corpus-derived input is a training-window **length** schedule from the
verified training offsets (`4a89e8ec…8350e109`). It corresponds to historical updates
6,489–7,485: 15,940 windows / 5,021,949 targets. Its first 512 updates end at the
historical recovery boundary; the remaining 485 reproduce the relevant padding/
variable-length pattern, including the two-microbatch final update. There is no
training text or actual token content in the transfer. No development/held-out/final
payload is opened. A synthetic dummy validation identity satisfies checkpoint APIs;
it is never evaluated.

For each profile:

1. Run one fresh 997-update reference trajectory, checkpoint/read back at update
   512, and continue uninterrupted to 997. Keep the split and final checkpoints.
2. In a new process, restore **that exact reference checkpoint**, verify the entire
   state, then replay only updates 513–997. The common prefix is executed once.
   This avoids confounding a restart comparison with two independently drifting
   pre-checkpoint trajectories. Keep the replay final checkpoint.

Thus each pair costs 997 + 485 = **1,482 physical optimizer updates**; two pairs cost
at most **2,964**, below the owner's 4,096 ceiling. There are no spare-budget retries,
extra profiles or post-result changes. CPU engineering tests forbid `Trainer.update`
and consume zero optimizer updates. The diagnostic scheduler uses 997 total updates
with 375 warmup and the existing AdamW values, batch two / accumulation eight,
clipping 1.0 and seed-160 shuffle. This bounded synthetic schedule is explicitly
separate from the old 7,485-update run and any future full Phase 17.1 recipe.

## Evidence and decision rule

Record every update's loss, gradient norm, exposure, exact input signature/lengths,
model tensor hashes, optimizer tensor hashes/parameter-name mapping, optimizer
groups, sampler/order/counters, global RNG hashes and resource measurements. Keep
exact fingerprints before and after checkpoint write/readback, after new-process
restore and at both endpoints. Global RNG consumption is unexpected because this
model has no stochastic layers; report it as a failure, not a hidden omission.
Only completed/cleared updates can be fingerprinted/checkpointed.

The original strict checkpoint loader is used unchanged. The diagnostic separately
binds the additional process controls absent from the historical checkpoint format.
Fresh initial state must match across profiles; restored model/optimizer/sampler/RNG
state must exactly match the reference's split. A mismatch stops the study and
identifies a storage/state problem rather than blaming a numerical kernel.

A **prospective correction is demonstrated in this protocol only if** both pairs
complete, the legacy continuation reproduces loss differences above `1e-5`, and
the corrected 485-update continuation has no loss excess AND exactly matching
per-update model/optimizer/sampler/RNG state hashes. Storage/readback/identity and
resource checks must all pass. If legacy drift is not reproduced, if either pair
is incomplete, or if corrected state still differs, report inconclusive/failed and
stop. Do not relax the rule after results.

A positive result supports an execution-policy mechanism after lossless restoration;
these two profiles do not uniquely identify an individual historical kernel or prove
that every historical difference had that cause. They establish only the tested
prospective policy on this host/runtime/synthetic shape schedule. No fixed Phase 14
score, useful language ability, general CUDA reproducibility or Phase 17 success
follows. Returned tensor hashes are evidence from the executed code, not a claim
that Mac numerically replayed absent Windows tensor payloads.

## Time, resource and preservation limits

One append-only supervisor owns the complete study. A persistent exclusive launch
receipt prevents a second invocation. The conservative active wall limit is **3,600
seconds**, including preflight, CPU preparation, imports, hashing, checkpoint I/O,
comparisons and all CUDA work. It is stricter than counting GPU kernels alone.
At most two pairs / 2,964 selected-scale physical updates; **6 GiB new artifacts**,
with a 1 GiB internal reserve for administrative/return copies. Existing reusable
runtime environments and all prior phase artifacts are outside the new artifact
roots and are never modified. The preserved hardware limits remain 8 GiB host
working set / 85% reserved GPU memory. Stop on nonfinite values, identity mismatch,
unsupported deterministic operation, OOM, limits or another execution failure.
No retries, manual backend/precision substitutions or budget reset. The native
SDPA dispatcher may select a supported deterministic kernel under the declared
policy; do not override it after results. Record incomplete output.

Use the existing RTX 4070 SUPER, driver 616.92, Python 3.14.7, Torch 2.14.0+cu130,
NumPy 2.5.3 and Safetensors 0.8.0. Do not upgrade dependencies/driver, reinstall the
historical source, inspect held-out data or rent resources. Raw diagnostic tensors
stay on Windows with inventories; the compact return is not their backup.

After actual CUDA return, Mac independently reconstructs identities, exposure,
comparison decisions and limits, then performs separate review. If a reproducible
correction is established, stop and prepare a concise proposal for a **fresh Phase
17.1 Mac Work chat**: new full pretraining from random initialization, accepted
Phase 15 inputs and Phase 16 model/training settings, only justified prospective
engineering corrections, unchanged fixed acceptance gates. Full Phase 17.1 needs
separate owner authorization and is not launched here. If inconclusive, stop and
report that outcome. Phase 18 remains blocked. No remote publication is authorized.
