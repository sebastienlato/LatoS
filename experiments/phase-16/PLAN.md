# Phase 16 bounded plan

Owner authorized Mac implementation and a separate Windows Work handoff on
2026-09-23. Phase 15 published baseline is
`d3c63d8655415252ac4a9efac023107831e1ad75`. No connected CUDA execution is available
from this Mac. Phase 16 stays open until actual RTX 4070 SUPER evidence is reviewed;
no final model configuration or Phase 17 training is authorized by these probes.

## Inputs and comparison

Use the immutable Phase 15 input package and selected 16,384-entry tokenizer.
Hash-select at most 4,096 training and 128 development documents independently of
model results. Verify file hashes, row hashes, group-derived splits and absence of
cross-split groups. Coalesce retained paragraphs within each document; adjacent
chunks rejoin verbatim, missing chunks/paragraph seams get two newlines. Do not
invent omitted text. Add BOS/EOS once per document. Window with overlap one,
isolating documents and scoring each target once. Bound each document to 262,144
bytes/65,536 IDs and each split to four million content tokens. Record all omissions.
The probe adapter retains a bounded token subset in memory; full Phase 17 streaming
and storage decisions must follow measured host memory and the final budget.

Explore depth while holding width, vocabulary, context, heads and FFN fixed:
width 512, eight 64-dimensional heads, FFN 1,408, context 512, layers 8/14/20.
The native formula gives 34,087,424 / 53,361,152 / 72,634,880 parameters.
These are independently derived comparison points, not selected architectures.
One below-range control helps detect whether extra depth earns its cost. Native
RMSNorm, RoPE, SwiGLU, SDPA and tied embeddings remain unchanged. No imported model.

CUDA preflight must identify RTX 4070 SUPER, actual VRAM, driver, Torch/CUDA,
native BF16 availability, free device memory and free disk. Require at least 10 GiB
total VRAM and 20 GiB free disk. Keep TF32 disabled and float32 matmul highest.
Compare float32 AdamW against BF16 autocast with float32 parameters/moments and
float32-sensitive reductions. BF16 uses no loss scaling; FP16 is outside this
bounded experiment. Verify finite losses/gradients/weights, token-weighted
accumulation, read-only validation and strict precision-preserving recovery.

## Prespecified bounds

Run original tiny synthetic numerical controls first, including an independent
float32 comparison; these establish mechanics only. BF16 loss difference <=0.03,
gradient relative L2 difference <=0.05, four-update weight relative L2 difference
<=0.03; same-backend resume weights atol 1e-6 / rtol 1e-5 with identical counters,
order and optimizer state. Preserve failures; do not relax tolerances after results.

Each candidate uses the identical real-data train/development selection, seed 160,
AdamW learning rate 0.0003, betas 0.9/0.95, decay 0.01, clipping 1, batch 2 and
accumulation 8. Float32 reference: 8 updates. BF16 probe: 64 updates, warmup 4,
cosine schedule to 0.1 peak. Exclude the first two updates only from throughput
statistics, never from exposure. Retain all update metrics and checkpoints at
initial/midpoint/final boundaries. Reload at the midpoint and require development
loss agreement within 1e-5 before continuing. Evaluate the complete selected development subset before
and after. Same data and update budget across scales; no best-checkpoint search.

Maximum six candidate processes (three scales × two precisions), each 30 minutes;
one numerical-control process, ten minutes. A failed correctness control stops
the suite. A candidate OOM/nonfinite/timeout is retained and makes it ineligible;
no automatic retries, parameter changes or smaller replacement run. Stop a process
if measured peak reserved VRAM exceeds 85% of total. Successful engineering probes
do not pass learned-quality gates. Windows Work may fix an actual implementation
defect, but must preserve attempts, review the fix and request a revised bounded
protocol before increasing the run budget or changing scientific thresholds.

## Measurement and later selection

Synchronize CUDA around timing. Record useful (unmasked) targets/s, wall time,
peak allocated/reserved VRAM through training, validation and checkpoint readback,
host peak RSS, checkpoint and total disk cost, actual exposure, data identities,
source/runtime and failure logs. Checkpoint readback allocates another trainer;
its cost belongs in the measured resource budget. Do not add allocator and reserved
memory as if they were independent pools. Host Windows RSS uses the operating
system working-set API; unavailable values are not zero.

Before selection, measure native float32 cached inference at 128 input IDs and
64 generated IDs, three repeats after one warmup. Engineering targets: peak GPU
reserved memory <=85% total, median first-token latency <=1 second and median
decode throughput >=10 tokens/s. Use forced-length original synthetic token prompts
for resource consistency and label them; this is not language quality evidence.

Eligible candidates need passing correctness/recovery, both precision runs complete,
BF16 development loss lower than its own initialization, full overhead memory <=85%,
inference targets met and a projected one-pass Phase 17 budget <=24 hours including
measured evaluation/checkpoint overhead at every 1,000 updates. Among eligible
candidates, choose the smallest within 2% of the lowest final BF16 development loss;
if none qualify, report the blocker and one bounded revision proposal. Early loss
on this subset cannot predict fixed Phase 17 quality gates. Final selection and the
precise Phase 17 token/seed/attempt/storage/recovery plan require Mac review of
actual CUDA evidence, including practicality of full-corpus batching. No automatic
Phase 17 launch. Initial planning ceiling: one full coalesced corpus pass, one seed,
one serious attempt, <=24 hours and <=20 GiB artifacts; this remains a ceiling,
not an approved run or a guarantee of sufficient quality.

Keep Phase 14 evaluations and all historical artifacts unchanged. No reserved
scoring, threshold change, dependency upgrade, paid compute, GitHub write or new
tag/release/assets. The normal reviewed-commit publication approval occurs only
after the required external evidence is incorporated and Phase 16 actually closes.
