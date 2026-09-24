# Selected dense model and Phase 17 training contract

Phase 16 selects the native **34,087,424-parameter** dense model for the first
LatoS 2.0 base run. This is a resource/configuration decision. No useful learned
language or instruction capability, final base checkpoint or Phase 17 execution
is established by this selection. Phase 17 requires separate explicit owner start
following approved and verified Phase 16 publication.

## Why this configuration

The prespecified [Phase 16 rule](../experiments/phase-16/PLAN.md) chooses the smallest
eligible model within 2% of the minimum short-probe loss. All three measured scales
pass resource, recovery and inference gates and fall within that band (the 14/20
layer ratios are 1.00780/1.01068). These 322,035-target probes are too short to
rank eventual learned quality. We do not conclude that smaller models learn better.
They do establish that larger configurations are feasible; neither larger size nor
unused VRAM is evidence that their additional cost is warranted for this first run.

| Native BF16 probe | Parameters | Targets/s | Peak reserved GiB | Final subset loss |
| --- | ---: | ---: | ---: | ---: |
| 8 layers — selected | 34,087,424 | 28,983.1 | 1.088 | 7.175839 |
| 14 layers | 53,361,152 | 17,433.5 | 1.629 | 7.231808 |
| 20 layers | 72,634,880 | 12,204.8 | 2.139 | 7.252446 |

The selected model has almost twice the parameters of the historical 17,308,032
base; this is increased capacity, not demonstrated capability. The roadmap's
50–80M region was investigated with actual measurements, not imposed as a minimum.
The selected model processes this short workload 1.66×/2.37× as fast as the two
larger options. Throughput is single-host, single-protocol evidence, not a sustained
full-run guarantee. Float32 probes used fewer updates and a different schedule
length; their final losses must not be used as a BF16 quality comparison.

[Native model config](../configs/training2/selected-model.json): width 512, eight
attention heads, eight layers, FFN 1,408, vocabulary 16,384, context 512, RoPE theta
10,000, RMSNorm epsilon 1e-5, tied embeddings, native causal SDPA/SwiGLU. The chosen
config exactly matches a measured candidate. No third-party model implementation
or dimensions were imported. No GQA, MoE, compilation or new attention mechanism.

CUDA BF16 autocast retains float32 weights, gradients and AdamW moments; no FP16
loss scaler. TF32 is disabled, matmul precision highest, CPU threads four, foreach
and fused optimizer paths off. Activation checkpointing remains off: measured
headroom does not justify its additional complexity or recomputation. The validated
stack is Python 3.14.7 / Torch 2.14.0+cu130, actual RTX 4070 SUPER, driver 616.92.
The existing committed lock remains unchanged.

## Data layout and exact exposure

Use accepted Phase 15 pretraining training text and the selected tokenizer only.
No instructions, development records, reserved records, evaluation prompts or
probe checkpoints become training inputs. Join adjacent chunks verbatim within
retained paragraphs, separate paragraph/gap boundaries with two newlines, add
BOS/EOS once per document, and use isolated overlap-one windows. Documents do not
share attention; dropped content is not reconstructed. All 80,478 accepted training
documents / 332,782 rows fit the existing per-document probe bounds; none is excluded.

Actual [full layout accounting](../experiments/phase-16/full-corpus-layout.json):
37,730,940 content-token positions, **37,811,418 targets including EOS**, and
119,748 windows at context 512. These differ from Phase 15's isolated-record counts
because the boundary policy and inserted newline accounting differ. They are
positions in one corpus pass, not semantically unique knowledge. Development:
1,730 documents, 804,337 targets, 2,552 windows. The two splits share no groups.
No reserved payload was opened for this accounting.

Seed 160 initializes a **new** model and the one-pass shuffle. Batch two, accumulation
eight, token-weighted loss and gradient clipping 1.0 are retained from the measured
protocol. There are 59,874 microbatches: 7,484 ordinary updates and a final update
with **two** microbatches, for **7,485** updates total. The final update must use its
actual target denominator and must not wrap the sampler into another epoch.
The existing generic trainer always takes eight microbatches; do not invoke it
unchanged for this exact one-pass contract. Before any Phase 17 optimization, the
full-data adapter and partial-final-update path must pass conservation, identity,
weighted-gradient and checkpoint-recovery checks. This prerequisite is integration
work for the separately authorized Phase 17, not an untested behavior claimed here.

AdamW: peak learning rate 0.0003, 375-update linear warmup (about 5% of the planned
run), cosine decay to 0.00003, betas 0.9/0.95, epsilon 1e-8, matrix weight decay 0.01
and zero norm-weight decay. The longer schedule is a prospective budgeted recipe;
the 64-update probe did not validate long-run convergence. One serious attempt,
one seed, one full pass. No automatic epoch increase or hyperparameter search.

## Time, memory and storage budget

The actual selected GPU has 12,878,086,144 usable bytes. BF16 peak allocated/reserved
was 0.986/1.088 GiB including validation and duplicate checkpoint readback; Windows
host peak working set was 2.470 GiB for the probe. The float32 reference peaked at
1.246 GiB reserved. These peaks are not estimates for larger batches or contexts.
Keep the measured batch/context/precision; do not fill free VRAM simply because it
is available. Host cap: 8 GiB process working set; GPU cap: 85% reserved VRAM.

Full training fixed-slot utilization is 61.79%, close to the probe's 62.28%.
Actual seed-160 batch-two padding accounting gives 76.40% useful target slots.
Streaming full-layout accounting on Mac peaked at about 0.313 GiB; it did not
measure full-training host RAM. A compact uint32 window cache plus uint64 offsets
would occupy about 0.143 GiB; that size is arithmetic, not a tested full-data cache.
A bounded reader/cache must be verified against the recorded window hash before
Phase 17 starts. Host RAM therefore remains an explicit execution stop condition.

Projection from actual useful-target throughput: **21.74 minutes of updates**.
Scale the slower initial/final measured development timing to the full development
split; nine validations plus initial/eight later checkpoints yield **23.90 minutes**
central training+periodic-overhead estimate. A twofold slowdown margin and five
minutes preprocessing give **52.79 minutes**. These are extrapolations from seconds
of measurement; thermal behavior, full-data I/O and prolonged driver behavior
remain unmeasured. Numerical predictions are not promises.

Allocate **two active hours for preparation/training/recovery**, plus **one active
hour for paired fixed development/final evaluation**, with a three-hour total stop.
This is below the original 24-hour ceiling. The evaluation hour is a reserve, not
a measured benchmark runtime. Record actual times; if any budget expires, preserve
the incomplete result and propose one bounded revision rather than silently extending.
Only existing resources; new paid-service budget zero.

Save immutable checkpoints at update 0, every 1,000 updates and 7,485: nine total.
A measured complete probe checkpoint is 409,128,236 bytes; the full shuffle state
adds about 0.91 MB per checkpoint. Reserve 0.45 GiB per complete checkpoint and
0.15 GiB for initialization, plus dataset/cache, logs, paired evaluation and limited
recovery evidence: **6 GiB planned**, **20 GiB hard cap** for new phase artifacts,
with at least 20 GiB free before starting. Existing Windows probe weights stay
preserved separately; the compact return archive is not their backup. Establish
appropriate local storage and checksummed transfer/backup before discarding any
only copy. No weights or acquired text enter Git history or an unapproved release.

## Recovery, selection and acceptance

A nonfinite loss/gradient/weight/optimizer value, OOM, corruption/identity mismatch,
reserved-data access, sampler/target mismatch, memory/storage overrun or wall-budget
expiry stops the run. Preserve source/environment, configuration, dataset identities,
logs, failed attempts and valid checkpoints. Never save a partly completed update.
Numerical/resource failures do not authorize a fresh attempt or relaxed settings.
At most two exact same-attempt resumptions are allowed for external interruption,
using the same runtime/source/configuration and complete optimizer/sampler state.
Log replayed/lost work separately; repeated execution is not unique data. Exceeding
this bound requires a recorded bounded revision, not an endless retry loop.

Use the last completed planned update only; no best-of-checkpoints/seed selection.
Monitor the complete new development split at checkpoint boundaries. Apply the
unchanged Phase 14 [protocol](../configs/evaluation/protocol-v1.json) and
[access rules](EVALUATION.md) to the completed candidate and paired historical base
on the same runtime/backend/batching. Old token IDs are not used with new weights;
compare matched-text BPB, not different-tokenizer perplexities. Missing historical
weights or pinned evaluation inputs block scoring, never permit substitution.

The Phase 17 gates still require >=10% matched-BPB improvement, bounded per-book
and external regressions, ARC-Easy >=35% and >=5-point gain, the confidence-bound
condition, and fixed generation limits. Lock the candidate only after development
success; then make the single permitted final acceptance comparison with its
reviewed contamination/selection declaration. No final feedback tuning. Failure
blocks Phase 18 and preserves a negative result.

One pass is about 1.11 content-token positions per model parameter. The data is
encyclopedia-dominated and modest; this first full-data budget cannot guarantee
the demanding quality gates or establish adequate training saturation. More epochs
do not create new data. Context 512 also fits only 13,111/14,259 instruction
conversations and about 68% of assistant targets; do not truncate answers. Phase 18
must explicitly revisit whole-conversation inclusion within validated capacity.
Longer-context performance and useful assistant behavior are not claimed.

The machine-readable [planning contract](../configs/training2/phase17-plan.json)
is deliberately not an executable training command. **Phase 17 has not begun.**
