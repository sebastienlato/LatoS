# Phase 5 fresh-chat handoff (historical)

**Consumed on 2026-09-20:** the owner explicitly started Phase 5 in a fresh Work
session after remote verification of closure commit
`52b72c9394e022d4cebc85808343bcaee73857c8` and unchanged v0.5.0. The instructions
below preserve the original transition context; current status is in
[PROJECT_STATE.md](../PROJECT_STATE.md).

This was preparation only. **Do not start Phase 5 in the closure chat.** Publish
and verify the approved Phase 4 closure first, then use the prompt below in a
fresh Work chat. The owner has not authorized Phase 5 execution yet.

## Starting point and constraints

Read `AGENTS.md`, `PROJECT_STATE.md`, `ROADMAP.md`, this handoff, and the local
private context/blueprint when present. Keep private planning out of tracked files
and GitHub metadata. The closure commit identity is supplied in its approval
request and publication verification; check that actual commit, not just v0.5.0.

Validated implementation: `v0.5.0`, `cb585c321c92f5d774fb59234f76c1d3783a635a`.
The owner reports Windows/RTX 4070 SUPER PASS: 159 passed, two MPS-only skips,
actual CUDA optimization and recovery. Exact CUDA equality was observed only in
that run, not guaranteed generally. Verified GitHub Linux CPU CI passed separately
with 159 passes/two MPS skips and offline training acceptance. Physical Linux is
**deferred, not performed**. See [closure evidence](../experiments/phase-4/CLOSURE.md).

The training engine is float32/single-device with verified dataset/tokenizer
identities, paragraph-isolated windows, AdamW, warmup/cosine schedule, accumulation,
clipping, read-only validation, and optimizer-boundary recovery. Consult
[TRAINING.md](TRAINING.md) before changing a run configuration: checkpoints require
matching implementation/runtime/backend/config/data. A model-only snapshot is not
a resumable optimizer checkpoint. Fixture memorization did not improve held-out
quality and is not a pretrained English starting point.

Preserve these ignored inputs and evidence:

- `data/processed/english-books-v1/`: fixed train/validation/test splits; use only
  training text for optimization, validation for development, and reserve test.
- `artifacts/tokenizers/english-bpe-v1/`: 8,192 entries, SHA-256
  `7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
- `checkpoints/phase-3-pilot-initial/`: randomly initialized 17,308,032-parameter
  pilot, context 512, seed 17; weight SHA-256
  `99dce7bf0356bf7d641dff4b29d6616772dfbf9779877e0c085bdea9e368b5f2`.
  Keep this baseline untouched; create new training outputs.
- Phase 4 acceptance and CLI artifacts under `outputs/phase-4-*`; do not overwrite.

Local uv: `.private/tools/bin/uv`; Python is under `.private/python`, with the
committed lock and `.venv`. Recheck actual host memory, disk, and backends rather
than assuming the external CUDA machine is available. New paid-service budget is
zero. Select a smaller runnable model/run if measured resources require it.

## Phase 5 objective, after explicit start

Execute a measured English pilot from random initialization. Before the run,
record seed, model/tokenizer/data identities, source/environment, token budget,
validation cadence, checkpoint selection rule, and fixed generation prompts/settings.
Measure memory, throughput, and runtime on available hardware before choosing the
run size. A larger run is conditional on feasibility, not a required expense.

Compare held-out validation loss against the random-initialization baseline using
the same tokenization/data/evaluation settings. Record target exposure, loss curves,
fixed samples, runtime, throughput, memory, checkpoint hashes, and failures. Keep
negative results; do not recast fixture overfit as language quality. Own tests,
separate review/fixes, documentation, and the local Phase 5 commit, then stop for
its own concrete push approval. Do not begin instruction tuning or later phases.

## Prompt to use after closure publication is verified

> Start LatoS Phase 5 in this fresh Work chat after verifying the published Phase 4
> closure commit and unchanged v0.5.0 at cb585c321c92f5d774fb59234f76c1d3783a635a.
> Read AGENTS.md, PROJECT_STATE.md, ROADMAP.md, docs/PHASE5_HANDOFF.md, and local
> private context when present. Windows/RTX 4070 SUPER Phase 4 validation passed;
> hosted Linux CPU CI passed separately; physical Linux remains deferred. Observed
> CUDA equality is not a general determinism guarantee. Run a measured English
> pilot from random initialization using available resources and no new paid
> services. Preserve existing artifacts and privacy rules; reserve the test split.
> Record baseline/held-out comparisons, token exposure, fixed samples, runtime,
> throughput, memory, checkpoints, and failures. Complete review and validation,
> then stop at the reviewed Phase 5 push-approval checkpoint.
