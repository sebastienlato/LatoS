# Phase 17 execution plan

The owner explicitly authorized Phase 17 after verified Phase 16 publication at
`ce9de1b734a8e23fd54e9d137cf269f484631765`. This authorization covers implementation,
validation, one bounded actual RTX 4070 SUPER pretraining attempt, fixed evaluation,
review and evidence integration. It does not authorize publication or Phase 18.
The historical `execution_authorized: false` in the frozen Phase 16 planning JSON
remains unchanged; this dated execution authorization supersedes that start gate.

## Frozen experiment

[Training contract](../../docs/TRAINING_2.md) and
[planning JSON](../../configs/training2/phase17-plan.json) remain byte-identical.
Native 8-layer / 34,087,424-parameter model, vocabulary 16,384, context 512, fresh
seed 160, CUDA BF16 with float32 parameters/gradients/AdamW state. Batch 2 ×
accumulation 8, final update accumulation 2; 7,485 updates consume exactly 119,748
windows and 37,811,418 targets. One pass, one seed, one serious attempt. No probe
checkpoint initialization, batch/context change, hyperparameter search or extra epoch.

Two active hours include input verification/cache/preparation, training, periodic
full new-development monitoring, checkpoints and recovery. A separate cumulative
one-hour allowance covers paired fixed development/final evaluation and inference
resource checks. Three active hours total. Pauses waiting for review do not consume
active time. At most two same-attempt resumptions for externally interrupted work;
no retries for numerical/resource/quality failures. Original failed segments and
replayed/lost exposure remain in the return. Nine immutable checkpoints at 0,
1,000…7,000 and 7,485. Final planned update is the sole candidate.

Runtime stays Python 3.14.7 / Torch 2.14.0+cu130 / measured driver 616.92 on the actual
RTX 4070 SUPER. No dependency or lock changes. Unsupported CUDA access on Mac is
not substituted with MPS or a synthetic learned-quality claim. Mac is authoritative
for code and publication; Windows executes the exact reviewed local transfer.

## Integration prerequisites

The full-corpus cache streams each accepted document, retains isolated overlap-one
windows in read-only little-endian uint32 arrays with uint64 offsets, and rehashes
all window contents against the independent Phase 16 sequence hashes. All train
and development counts/document identities must agree; no omissions. No instruction
or held-out material enters optimization. Training's one-permutation mode refuses
an inconsistent update budget, uses the actual tail target denominator and saves
a distinct strict checkpoint version. Legacy training retains its default mode.

Validate independent full-batch gradient equivalence, odd/even final microbatches,
exact exposure, all checkpoint boundaries, fixed-seed sampler identity, nonfinite
optimizer state rejection, corruption rejection and CUDA BF16 tail recovery.
Run checkout and isolated installed-wheel suites; both CUDA numerical tests must
execute on Windows before serious optimization. These are synthetic engineering
controls, separate from the learned experiment. Cache verification itself creates
no model. Retain actual test failures and fixes without changing scientific gates.

## Evaluation and review

Use the unchanged Phase 14 evaluator and protocol. The package includes the actual
historical Phase 5 model (`f82ed3b…59b2befa`), matching tokenizer, pinned ARC validation
files, original LM corpus report/development and an opaque copy of the original
reserved LM test. Packing/verifying its hash is not scoring. Routine evaluation
reads only development; no old instruction reserved test is included.

Evaluate historical base and final candidate on the same Windows runtime/backend,
batch 8, four CPU threads, identical frozen prompts/decoding/context. Compare matched
BPB across the different tokenizers. Preserve all raw rows/samples and independently
reconstruct aggregates; no test-threshold modification or cherry-picked checkpoint.
Development failure is a valid negative learned result and blocks final access and
Phase 18. A passing development result is not phase acceptance: perform a separate
selection/contamination/resource review before creating the exact declaration in
`docs/EVALUATION.md` and running the one final paired comparison. No final-feedback
training. Return evidence for authoritative Mac integration in either outcome.

Memory, disk, time, identity and finite-value violations stop the attempt. Preserve
only complete valid checkpoints. At least 20 GiB free before execution, 8 GiB peak
host memory, 85% reserved GPU cap, 6 GiB planned experiment artifacts and 20 GiB hard
new-phase artifact cap. Transfer, validation logs, retained build/test outputs and
attempt artifacts must be counted; reusable runtime environments are reported
separately. Packaging time is administrative and recorded separately from training
and evaluation. Final weights return to Mac; optimizer/initial/intermediate tensors
stay on Windows with recorded hashes. The evidence subset is not their backup.

Completion requires actual training/evaluation evidence, separate review and fixes,
negative or positive outcome reporting, and an exact local reviewed commit before
asking for Phase 17 publication approval. No remote backups, PRs, tags or assets.
Phase 18 remains explicitly forbidden, including after Phase 17 publication.
