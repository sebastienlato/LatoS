# Phase 5: measured English pilot

Completed locally on 2026-09-20; publication approval is pending.

## Outcome

The original 17,308,032-parameter pilot trained from verified seed-17 random
initialization for 3,000 updates on the retained eight-book English training split.
Full held-out validation loss improved from **9.089003 to
4.731898**; perplexity fell from **8857.350 to
113.511**. This is a measured base-model experiment,
not a usable assistant. Samples contain recognizable English but also repetition,
contradictions, implausible events, and incomplete thoughts.

The published Phase 4 closure and unchanged v0.5.0 were verified before starting
this fresh, explicitly authorized session. The [protocol](PLAN.md) fixed the
budget, checkpoint selection, baselines, prompts, and resource policy before the
main run. No test text was opened and no new paid service was used.

## Comparable held-out scores

All three scores use the same 111,523 targets in 1,067 validation windows, the same
8,192-entry BPE, EOS targets, and isolated paragraph/window boundaries. Validation
is two whole books, not a random paragraph sample of training books.

| Model | Validation cross entropy (nats/target) | Perplexity |
| --- | ---: | ---: |
| Random initialization | 9.089003 | 8857.350 |
| Add-one unigram fitted only on training targets | 6.883256 | 975.798 |
| Fixed final checkpoint, update 3,000 | 4.731898 | 113.511 |

The separate first-128-training-window diagnostic (13,153 targets) improved from
9.092997 to 3.767135. It is not a full-corpus
train score and should not be used as an unbiased estimate of the generalization
gap. Perplexity is comparable only under this fixed tokenizer/evaluation contract.

| Update | Target exposures | Full validation loss | Perplexity |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 9.089003 | 8857.350 |
| 500 | 966,033 | 5.389251 | 219.039 |
| 1,000 | 1,921,425 | 5.091386 | 162.615 |
| 1,500 | 2,880,655 | 4.917900 | 136.715 |
| 2,000 | 3,840,858 | 4.783827 | 119.561 |
| 2,500 | 4,799,367 | 4.730916 | 113.399 |
| 3,000 | 5,761,229 | 4.731898 | 113.511 |

The lowest sampled validation loss was at update 2,500. Update 3,000 is slightly
worse by 0.000983 nats/target. The fixed final checkpoint remains selected; no
post-hoc best-checkpoint substitution or additional tuning occurred. This plateau
and the small corpus do not justify a larger model/run for this phase.

## Exposure and resources

- **5,761,229** actual target exposures, across 47,992 windows:
  **4.511389** corpus-equivalent passes. All 1,277,041 distinct
  training target positions and 10,638 windows were visited. Distinct means corpus
  positions, not unique vocabulary or n-grams. Exposure includes repeated epochs.
- Model capacity: 512 tokens; actual training/evaluation windows: 256 tokens.
  Batch 8, accumulation 2; upper target budget 12,240,000. Short paragraphs account
  for the lower actual total. The final epoch-tail microbatch can be short.
- Apple M4 Max, 64 GiB unified memory, MPS float32, one CPU thread, Python 3.14.7,
  PyTorch 2.14.0. CUDA was unavailable. Accepted dependency versions were retained.
- Synchronized optimization: **393.836 s**, **14,628.5 targets/s**.
  Loop including periodic validation/checkpointing: **407.455 s**,
  **14,139.5 targets/s**. Timed run span with preparation,
  initial evaluation, and final samples: **417.224 s** (about 6.95 minutes).
  This excludes interpreter/import startup, output creation, final summary write,
  and artifact inventory hashing. No benchmark or verification time is included.
- Process lifetime peak RSS: **3,458,678,784 bytes**
  (3.221 GiB). Maximum observed MPS driver allocation:
  **4,593,123,328 bytes** (4.278 GiB).
  Maximum observed live MPS allocation: **213,725,696 bytes**.
  Accelerator figures are boundary snapshots, not continuous peaks. They overlap
  unified memory accounting and must not be added to RSS.
- No other training or test workload ran concurrently with the main pilot. Code
  inspection, documentation, and lightweight lint/privacy checks did occur.
- No training interruption, nonfinite update, out-of-memory event, or retry occurred.
  The ten-update resource probe remains a separate discarded model, with metrics
  retained. Its earlier runner revision differs in formatting/artifact bookkeeping;
  the main run preserves its exact runner bytes and hash.

## Fixed samples and limitations

All four prompts, both initial/final continuations, generated token IDs, seeds,
and stopping reasons are retained in [samples.json](samples.json). The settings
were temperature 0.8, top-k 40, seed 71, at most 64 new tokens. No favorable sample
selection or regeneration was performed. Three final cases reached the 64-token
limit; the letter continuation stopped on EOS after 60 generated tokens.

- Morning light: a plausible opening shifts to unrelated events and an unfinished
  word. It does not maintain a coherent scene.
- Letter: fluent sentence fragments include the implausible claim that a house was
  thrown on a table; meaning and spatial consistency fail.
- Ocean: repeats sea/south imagery and ends mid-thought.
- Experiment: drifts into whale anatomy and repeats body/head phrases; it does not
  explain an experiment or follow the implied topic reliably.

The corpus is narrow historical prose. Four qualitative cases and two held-out
books are insufficient for broad language, knowledge, factuality, safety, or
instruction-following claims. This run does not test useful 512-token context,
modern English breadth, chat, mixed precision, or distributed training. The test
split remains reserved for a later explicitly planned evaluation.

## Preserved artifacts and provenance

Full outputs remain locally under `outputs/phase-5-english-pilot/`; the resource
probe is `outputs/phase-5-benchmark/`. Prior corpus, tokenizer, random snapshot, and
Phase 4 artifacts remain untouched. Nothing was uploaded. These local paths are
not a remote backup or a promise that another checkout contains the weights.

| Checkpoint update | Model Safetensors SHA-256 |
| ---: | --- |
| 0 | `99dce7bf0356bf7d641dff4b29d6616772dfbf9779877e0c085bdea9e368b5f2` |
| 1,000 | `097c3d09abd8d9141f2a4f8a7a1ad2a7b8e29bdd759d8f1fff09901384d137d8` |
| 2,000 | `c824c063c3247fbb3f0dc491f36196ca260a875edc1e41483990688bbf7783c6` |
| 3,000 | `f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa` |

The selected model is `outputs/phase-5-english-pilot/step-00003000/model/`.
The enclosing checkpoint also contains optimizer moments, schedule/configuration,
shuffle state, source/runtime/data identities, and exact exposure counters. Initial
and 1,000/2,000/3,000 checkpoints are all retained. [artifacts.json](artifacts.json)
provides every output file's size/hash relative to that run directory, including
optimizer tensors. It is byte-identical to the local inventory; its SHA-256 is
`99621c23dfaf0353da37769ee9f5eecd78f16de42f6ac4b61b6bace558a32637`.

[results.json](results.json) contains full compact metrics, identities, configuration,
resource-probe results, and 500-update online training-loss aggregates. Raw per-update
logs and all learned artifacts remain ignored. Source provenance honestly records
base commit `52b72c9394e022d4cebc85808343bcaee73857c8` with a dirty working tree during
development. Runtime source, runner, and lock hashes bind the executed implementation;
no core model/training source changed in this phase. The final reviewed phase commit
contains the exact main runner. Package/lock metadata advances to 0.6.0 only.

## Validation and review

A separate process rehashed all **25** inventory files and accounted for every
one of the **3,000** updates. It loaded the initial/final recovery checkpoints,
verified initial tensors against a new seed-17 initializer, and recomputed both
validation scores exactly on this host. Loading the update-2,000 checkpoint and
replaying update 2,001 reproduced target exposure and scalar loss (absolute loss
difference **0.0**; accepted atol 1e-6/rtol 1e-5).
[verification.json](verification.json) records the measured scope: this is not a
claim of general MPS/CUDA determinism or all-future-weight equality.

The [review record](REVIEW.md) lists completed checks, corrections, and packaging
validation. Earlier Windows/CUDA PASS is owner-reported Phase 4 evidence only;
hosted Linux CPU CI passed separately at that earlier implementation. Independent
physical Linux remains deferred, not performed. No Phase 5 Windows/Linux execution
or post-publication CI result is claimed.

See [PRETRAINING.md](../../docs/PRETRAINING.md) for reproduction and recovery.
Stop at the reviewed Phase 5 publication checkpoint; no Phase 6 work has begun.
