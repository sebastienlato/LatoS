# Phase 17.1 — Base Model 2.0 Corrective Attempt proposal

**Status: reviewed scientific and engineering specification; not authorized to run.**
The completed [diagnostic review](../recovery-diagnostic/RETURN_REVIEW.md) supports
the prospective execution policy. This proposal permits no training by itself.
The [machine-readable contract](plan.json) pins the recipe, original runtime source,
inputs, controls, resource ceilings and gates. A separate Phase 17.1 controller
must still be implemented, reviewed, tested and hash-bound before execution; this
document is not a claim that an executable Windows transfer already exists.

The decision requested is authorization to implement that bounded controller and
preflight, then execute **one fresh attempt** under this specification. Material
changes to the scientific recipe, controls, budgets or gates require renewed owner
approval. No publication or Phase 18 is included. The previous diagnostic budget
is finished and cannot be reused for extra comparisons.

## Preserved history and scientific question

Phase 17 remains the permanently failed/interrupted experiment, with its original
failure records, failed `1e-5` recovery gate and unaccepted recovered model. Neither
its models nor diagnostic/probe weights may initialize, substitute for or become
the new candidate. The new question is whether a correctly executed full-data pass
from random initialization passes the previously fixed Base Model 2.0 quality gates.
The positive synthetic diagnostic establishes no answer to that quality question.

Use attempt identity `phase17-1-base2-attempt1`, fresh seed **160**, one model,
one seed, one pass and no search. Reset AdamW to empty state. Separate output roots,
source manifests, receipts and candidate hashes distinguish this experiment from
every prior phase. The evaluator's existing schema uses numeric phase `17` for the
base gate; retain that enum with a separate Phase 17.1 attempt identity. It must
never relabel the closed Phase 17 or authorize a second final-test candidate.

## Fixed model, data and optimization

Use the Phase 16 native **34,087,424-parameter** model: eight layers, width 512,
eight heads, FFN 1,408, context 512, vocabulary 16,384, tied embeddings, native
causal SDPA/SwiGLU/RoPE/RMSNorm. Selected-model file SHA-256:
`bed0bbcc8b9b145485584ac9475d0c450c196649cb797a09d1c1c47fddc14c66`.
No model-scale, attention, context, batch, precision or dependency change.

Accepted Phase 15 input-bundle manifest:
`5f4fa7a7bc221d06f6b321c4af712456d3188984d759f8a4a898651d5a6d9073`.
Tokenizer: `23b9182d5943e7f8b32c6f415f108b7d06f229c80f5cd8f236258db40a6b94c4`.
The same retained-document, overlap-one policy uses all **80,478 training documents /
332,782 rows / 119,748 windows / 37,811,418 targets**, including EOS. Training-window
sequence SHA-256: `8f13652c2bd0bc387dbd509ca6424a0eb97c097b963c4b64aa5144d8902cfd64`.
No instructions, development records, reserved records or evaluation content enter
optimization. Do not rebuild or recurate accepted data. Rehash and verify the
existing cache or construct a separate byte-equivalent cache with the bound reader.

Fresh seed-160 initialization and one seed-160 CPU permutation; no epoch wrap.
Batch two × accumulation eight, token-weighted loss, clipping 1.0. **7,485 updates**;
the last uses exactly two microbatches and its actual target denominator. Native
CUDA BF16 autocast, float32 parameters/gradients/AdamW moments, no GradScaler.
AdamW peak LR 0.0003; 375-update linear warmup; unchanged cosine decay to 0.00003;
betas 0.9/0.95, epsilon 1e-8, matrix decay 0.01 and zero norm-weight decay.
Monitoring uses only the same new development split: 1,730 documents, 2,552 windows,
804,337 targets. Monitoring never selects an earlier checkpoint.

## Exact prospective engineering policy

Keep the imported native package byte-identical to training commit
`0fa94ad580110fd2dc7aaa2aa3560950abe73a02`, implementation hash
`76cf4a198ee6acde566fdcc70c9420c0cae8ab6cc0d6b4d7e4c9c0cba5c8f09c`.
Implement a separate controller around those original APIs; do not patch their
globals, migrate historical metadata, or invoke the old training/recovery supervisor.

Apply the diagnostic's **combined** prospective correction, rather than attributing
the result to any one switch:

- Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTHONHASHSEED=0` in the child environment
  before Torch is imported or CUDA initialized.
- Enable deterministic algorithms with `warn_only=False` and cuDNN determinism.
  Unsupported deterministic operations fail closed; no backend/precision substitution.
- Preserve TF32 off, highest float32 matmul precision, cuDNN benchmarking off,
  four CPU threads and the measured 20 interop threads, all native SDPA backends
  enabled, and BF16 reduced-precision reduction enabled. Preserve non-fused,
  non-foreach AdamW, no compilation and no activation checkpointing.
- Pin Windows 11 build 26200, Python 3.14.7, Torch 2.14.0+cu130, NumPy 2.5.3,
  Safetensors 0.8.0, CUDA build 13.0, cuDNN 92400 and driver 616.92 on the RTX 4070
  SUPER (capability 8.9). Existing lock and reusable environments stay unchanged.
  Abort on undeclared numerical environment overrides or identity differences.

A single process supervisor uses persistent exclusive launch receipts, OS-held
locks, an append-only fsynced event journal and immutable completion/failure records.
Do not replace a monitored running ledger. Monitor complete journal lines only;
retain torn final lines as incomplete evidence. Reject concurrent or orphaned work,
existing output roots, second initial launches and direct internal-worker bypasses.
All child launch/import, input verification, monitoring, hashing, checkpoint and
readback costs consume the stated active budget. A failed child kills/stops the
stage; no automatic retry or time reset. Resource scans tolerate a vanished path
during atomic checkpoint rename, but no other I/O error is ignored.

Before update zero, persist the 7,485 rates computed by the pinned Windows
`TrainingConfig.learning_rate_at`, source identity and its hash. Validate every
executed update and every restore against that exact same-host table. Mac may
document platform arithmetic differences but may not alter the table or waive a
same-host mismatch. The schedule formula itself does not change.

## Checkpoints, restart validation and evidence

Save immutable checkpoints at **0, 1,000, 2,000, …, 7,000 and 7,485**. Monitor the
complete development split at these boundaries, preserve all metrics and restore
training mode. Checkpoint only completed updates with gradients cleared. Require
exact full-state readback and a successful final readback before declaring completion.

Every checkpoint gets a mandatory, immutable Phase 17.1 policy record binding the
attempt, reviewed controller/transfer, original native package, model/data/config,
runtime/controls, schedule-table hash and full state fingerprints. Bind actual
global RNG bytes or reproducibly initialized unchanged RNG to the fingerprint;
never silently omit it. Treat the native checkpoint plus matching policy record
as a single completed checkpoint. No subsequent update starts until both are
durable and readback succeeds. The original strict loader remains intact; the
controller adds policy and exact-state checks before any continuation. Missing
policy records, historical checkpoints and settings changes are rejected.

Record every update's input signature/lengths, loss, pre-clip gradient norm, exact
model/optimizer/group/parameter-map/sampler/global-RNG fingerprints, schedule,
targets/windows and resources, using the diagnostic's measurement conventions.
Full state hashes must be recomputable from their component records. Retain source
snapshots and input inventories. A new-process zero-update restore check must pass
before an interrupted attempt can resume; it is not extra pretraining.

At most **two same-attempt resumptions** may be considered for genuine external
interruptions after Mac review of the preserved failure, source and resource charge.
No automatic restart. Restore the last completely committed checkpoint under the
same identities. Verify **every available previously logged overlapping update**:
exact input, model, optimizer, sampler, global RNG and same-Windows learning rate;
absolute loss difference <=`1e-5`. Any mismatch is terminal before extending the
logical trajectory. Never use a smaller comparison horizon to bypass a failure.
Missing/torn final logging is unknown work, not verified equality; conservatively
charge its upper bound and disclose it. No checkpoint older than the latest complete
one may be selected. If progress or elapsed time cannot be conservatively bounded,
stop rather than reset the counters. No resumption for numerical, identity, quality,
resource, storage or unsupported-operation failure.

With at most 1,000 completed/possibly completed uncheckpointed updates per interruption,
the absolute serious-work ceiling is **9,485 physical update executions** including
at most 2,000 repeats. The logical trajectory remains 7,485 updates / 37,811,418
targets. Also record exact replayed targets and unknown partial work separately;
neither counts as unique data or an additional logical pass. Time and artifact caps
may stop the attempt before these upper limits.

## Budgets and preflight

One fresh serious attempt; zero new paid services. Retain the original bounds:
**7,200 active seconds** for preparation/preflight/training/monitoring/checkpoints/
recovery; **3,600 cumulative active seconds** for paired development and gated final
evaluation/inference; **10,800 total active seconds**. Waiting for owner/Mac review
does not consume active time. Packaging is separately timed and included in artifact
accounting. No unused diagnostic budget carries over.

Host working set <=8 GiB, reserved GPU memory <=85% of the measured
12,878,086,144 bytes, at least 20 GiB free before launch, approximately 6 GiB planned
and **20 GiB hard cap for all new attempt artifacts**, including source transfer,
logs, tests, checkpoints, return copies and a 1 GiB administrative reserve. Reused
unchanged historical payloads/environments remain separately inventoried; copying
them creates new bytes that must be counted. Never delete evidence to fit the cap.

The deterministic diagnostic reference took 486.824 seconds for 997 updates with
per-update hashing. Linear scaling gives about **3,655 seconds (61 minutes)** for
7,485 updates before full-data preparation/monitoring/checkpoint differences. This
is an estimate, not a measured full-run duration or guarantee. It replaces reliance
on the older ~24-minute estimate that did not include this level of state tracing.
The two-hour cap is retained, with no promise that two resumptions will fit.

Before serious updates, validate the implemented standalone controller in a
separate review and freeze its exact local commit and allowlisted transfer hash.
Use focused zero-update CPU checks for policy/authorization rejection, checkpoint
identity and partial-save refusal, exclusive launch/locks, append-only journal,
deadline/child termination, resource accounting, failure preservation and final-access
gates. Reuse the immutable native source's prior checkout/wheel validation evidence
only within its original scope; do not claim new full-suite execution.

On Windows, add bounded adapter conformance tests: two fresh tiny CPU-defined model
fixtures per runtime (ordinary and partial-tail batching), each eight reference
updates, checkpoint at three, then five new-process replay updates. Execute in the
original checkout and existing isolated non-editable wheel runtime: **52 tiny CUDA
updates maximum total**, charged to the training time/artifact budget and reported
separately. `plan.json:adapter_preflight` fixes both original synthetic fixtures:
one layer, width 32, four heads, FFN 64, vocabulary 260, context 512; 32/30 windows,
batch two, accumulation two, eight updates with two warmup updates, seeds 17110/17111
for initialization and 160 for the shuffle. It also fixes every synthetic token
by formula. Freeze their generated hashes before dispatch; no learned
or selected-scale tuning. These are adapter checks, not another diagnostic or
evidence replacing the accepted 485-update selected-scale comparison. Any CUDA
preflight failure stops for Mac review without serious optimization or automatic
retest. Cache/source/input validation and new-process checkpoint inspection use
zero optimizer updates. Do not run unrelated historical suites or held-out readers.

## Fixed acceptance and final-access boundary

Only the completed new checkpoint at update 7,485 is eligible. Paired reference is
the preserved Phase 5 model
`f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa`
with its original tokenizer. Reverify the original Phase 17 transfer's evaluation
input hashes; availability is mandatory, with no random or failed-model substitution.
Both models use the same pinned CUDA runtime/numerical policy, evaluation batch
eight and four threads. Evaluation runs in fresh processes with no optimizer.

The unchanged [Phase 14 protocol](../../configs/evaluation/protocol-v1.json) requires
all of the following at development and the permitted final comparison:

- Matched-text BPB ratio <=0.90; each book ratio <=1.02.
- ARC-Easy raw accuracy >=0.35 and gain >=0.05; Wilson lower bound above chance.
- No external raw or byte-normalized accuracy regression exceeding 0.03.
- Empty continuation fraction <=0.05, mean repeated-trigram fraction <=0.25,
  and repetition increase <=0.10. Complete fixed cases; no overflow substitution.
- Median first-token latency <=1 second, decode >=10 tokens/second and the fixed
  memory/storage/time bounds. No metric tradeoff can waive a failed gate.

Use matched BPB across tokenizers, never raw perplexity comparisons. Fixed context
256, matched spans 192 bytes, greedy 64-token samples, original prompts/scoring/
bootstrap rules all stay unchanged. Preserve every sample, raw score and negative
result. The encyclopedia-only corpus and one-pass exposure may still fail quality.

After valid training, execute paired **development only**, then return evidence
to this authoritative Mac task. Failed development or resource gates terminate
the attempt. After a pass, Mac independently verifies scores, selection and accepted
Phase 15 contamination evidence and locks the candidate/reference hashes. Only then
create the existing schema-1 acceptance declaration with reviewed comparison/audit
hashes, `accepted_for_final=true`, and the literal no-final-feedback declaration.
Run exactly one paired final comparison under [the unchanged access rules](../../docs/EVALUATION.md).
The old instruction reserved test remains unopened. No final feedback tuning or
second candidate; a negative final result blocks Phase 18 and cannot fund a retry.

Phase 17 never accessed fixed final acceptance, so this proposed fresh attempt does
not reuse an observed final score. The separate attempt identity must accompany
the evaluator's phase-17 schema declaration and the permanent failed-history record.
Passing every gate still grants no Phase 18 or publication authorization here.

## Handoff and stop

The [handoff](HANDOFF.md) is a preparation/authorization brief, not a CUDA command.
After owner authorization, implement only the specified controller, fix routine
engineering findings, freeze the source/fixture/transfer identities and dispatch
the bounded Windows execution. Return evidence for Mac review at each required gate.
Preserve all historical and new originals. Final candidate weights return to Mac;
optimizer/intermediate tensors need retained Windows inventories and a verified
backup before any deletion. No whole-folder archives, credentials or private context
enter an outgoing source package or Git history.

**Stop now. Full Phase 17.1 training is not authorized.** No Phase 18, push, remote
backup, tag, release, model asset upload or other publication is authorized.
