# Prospective small-subset quality diagnostic

2026-09-25 — **proposal only; stop for owner/master-planning review**.
The owner rejects the prior 6.6-hour/four-pass study. It must not be implemented.
The maximum for this **entire experiment, all arms combined, is 45 minutes** of
computer execution, including CPU preparation, preflight, GPU work, evaluation,
verification and packaging. No resetting the clock, deferred packaging, paid
compute or separate 45-minute allocations per arm. Target approximately 33 minutes.

## Question and design

**On a fixed small corpus, does repeated exposure with a longer schedule improve
development quality, and does reducing depth help at the same exposure?**
This is an early-learning diagnostic, not an expected accepted Base Model 2.0.
The evidence warrants a modest contrast, not another full-corpus run. The old
monitor was still improving, but both book scores and repetition failed; exposure,
schedule, capacity and domain mismatch remain hypotheses.

Reuse the already selected Phase 16 inputs: 4,096 training documents, 6,105 windows,
**1,943,015 targets per pass**, plus its separate 128-document / 57,036-target
monitor. Reuse the exact identities in [input-repeat.json](../phase-16/input-repeat.json);
no new source, domain mixture, sampling search or tokenizer fitting. These are
encyclopedia data; domain coverage is held fixed and cannot be diagnosed here.

| Fresh arm | Native model | Exposure | Schedule and role |
| --- | --- | --- | --- |
| A | Existing 8-layer / 34,087,424 parameters | 1 pass, 382 updates | Short cosine; control |
| B | Same model | 2 passes, 764 updates | Long cosine; exposure/schedule arm |
| C | Depth reduced to 4 layers / 21,238,272 parameters | Same 2 passes, 764 updates | Same long cosine; depth contrast |

C halves only LatoS's existing depth; width 512, heads 8, FFN 1,408, vocabulary
16,384, context 512 and all other architecture settings remain fixed. Its parameter
count follows the native formula; its runtime and quality are unmeasured. No new
model family, larger model, compilation, packing or optimizer change is proposed.

Each arm starts from random initialization with seed 160 and empty optimizer state;
never load old trained weights. A/B must have identical initial tensors and first-pass
ordering. C uses the same seed and data order, but is not tensor-matched across depth.
Keep accepted BF16/deterministic mechanics, batch 2 × accumulation 8, AdamW peak
0.0003, minimum ratio 0.1, betas 0.9/0.95, decay 0.01 and clipping 1. All arms use
**19 warmup updates**, then cosine decay over their stated horizon. Fix this once;
no learning-rate tuning. Freeze first-pass ordering; B/C share the next permutation
from the persisted CPU sampler RNG. Flush at pass boundaries: 3,053 microbatches,
382 updates, last update five microbatches and last microbatch one window per pass.

Exactly **1,910 serious updates / 9,715,075 target executions** across three arms;
only 1,943,015 distinct training positions, not 9.7M unique data. At most the existing
52 tiny CUDA conformance updates once, included in time: 1,962 physical updates
maximum. No optimizer work outside these bounds, retries, resumptions or extra seeds.

## Observations and interpretation

Monitor only the fixed Phase 16 development subset at initialization and pass
boundaries. Retain checkpoints there (eight across A/B/C), plus per-update counters,
loss, learning rate, clipping, timing and full-state/input identities. No best-checkpoint
search. Score **A1, B1, B2 and C2**, plus the unchanged Phase 5 primary reference,
once each with the fixed full development protocol after training completes.
B1 is a prespecified diagnostic, never an alternate promoted checkpoint. Do not
inspect its fixed-suite result to change later training. No final/reserved scoring.

- A1 vs B1 isolates the short/long schedule at equal first-pass exposure and model.
- B1 vs B2 describes additional exposure along the long schedule; evolving learning
  rate remains coupled to exposure. A1 vs B2 compares the complete budget recipes.
- B2 vs C2 isolates the depth choice at matched data, exposure and optimizer schedule;
  it does not establish a compute-optimal architecture or seed-robust superiority.

Report matched BPB, both book ratios, repetition, ARC raw/byte results and existing
paired uncertainty for every designated endpoint; no composite winner score. Require
consistent direction across book BPB and repetition before treating a difference as
useful planning evidence; report tradeoffs and wide intervals as inconclusive.
Only monitoring improvement suggests in-domain learning, not better general language.
A same-seed result cannot establish general architectural superiority. No additional
quality threshold substitutes for any acceptance gate.

All outcomes return to owner review. If neither repeated exposure nor reduced depth
shows coherent development benefit, recommend rethinking data/training design, not
increasing runtime. A positive result only justifies considering that direction in a
new bounded proposal. No model promotion, automatic follow-up or Phase 18. Even an
unexpected all-gates development pass still needs separately authorized selection,
contamination review and the unchanged final-access process before acceptance.

## Cost and failure limits

Use measured full supervision cost, not GPU update throughput alone:
1,910 × (4,559.0812 / 7,485) = **19.4 minutes** estimated training/evidence time.
This conservatively charges C like the larger model; no speedup is assumed. Setup
and extra short-job overhead are uncertain. Five development observations are
projected from the saved two-model 162.8-second measurement: approximately 6.8 minutes.

| Included work | Planning allowance | Hard stage cap |
| --- | ---: | ---: |
| Input validation, setup and existing conformance | 3 min | 3 min |
| All three training arms, monitors and checkpoints | 20 min | 24 min |
| Five fixed development observations | 7 min | 8 min |
| Verification and compact return packaging | 3 min | 5 min |
| Total | **33 min** | **40 min + 5 min emergency headroom; absolute 45 min** |

Enforce one monotonic deadline from the first experiment computer job. Stop training
at its cap and preserve an incomplete result; never shrink the data or omit scores
post hoc to claim completion. Use emergency headroom only to close/save evidence,
not for extra optimization or expanded evaluation. An interrupted/over-budget study
is inconclusive. If required checks cannot fit, stop; no hidden preparation run.

Use existing RTX 4070 SUPER only, <=85% reserved VRAM, <=8 GiB worker RSS, <=8 GiB
new artifacts and >=12 GiB free disk. Eight checkpoints at the larger model's
~409 MB each project ~3.05 GiB before traces/exports; 8 GiB is a conservative ceiling.
Preserve historical artifacts. Nonfinite state, identity/split/target mismatch,
OOM, time/storage limit or conformance failure ends the study without retry.
Zero paid compute; electricity cost is unpriced. Estimates are not measured feasibility.

## Review boundary and context discipline

No implementation, subset reconstruction, model initialization, training or scoring
has occurred. Before any later execution, freeze exact source/input/runtime identities
and verify multi-pass tails/schedule/accounting with zero-update checks within the
same experiment preparation allowance. If that is insufficient, return a no-go rather
than silently extending budget. Implementation itself still requires owner approval.

Use one authoritative task, one compact manifest and one final table; reuse saved
identities and relevant prior checks. No agent fan-out, broad historical reanalysis,
full optimizer-bearing test suite, duplicate reports or open-ended searches. Store
full machine logs locally and summarize once; avoid sending raw traces into context.
No quantitative GPT-token budget was supplied, so none is invented.

Separate design review checked native parameter arithmetic, per-pass tail counts,
aggregate exposure, timing/storage sums, fixed reference/gates, privacy, and the
single cumulative deadline. Saved Phase 16/17.1 records were used; no new empirical
claim or external recipe was imported. **Phase 17 stays permanently failed/interrupted;
Phase 17.1 stays development-failed; no accepted Base Model 2.0; Phase 18 blocked.**
