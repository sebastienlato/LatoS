# Project state

Updated: 2026-09-19.

## Phase status

**Phase 4 is complete locally, reviewed, and awaiting push approval.** The owner
explicitly authorized it in a fresh Work chat. Before development, read-only remote
inspection verified Phase 3 closure commit
`294d2e0af2712266b398151eb7e335fa7ccdc31e` on `origin/main`; v0.4.2 still points to
`1102b64714b58c1f2289f80863fa032cc192c47b`. No earlier tag moved.

The 0.5.0 implementation adds isolated-record batching, AdamW, warmup/cosine
scheduling, token-weighted accumulation, clipping, validation, metrics, and complete
tensor-only optimizer/sampler recovery. Dependencies remain pinned and unchanged.

## Evidence and limits

- Local macOS CPU/MPS: **161 tests passed** in development and in a fresh locked
  environment using the built wheel. Lint, format, builds, and archive checks passed.
- Full offline fixture: training loss **5.814638 → 0.010030** after 400 updates.
  CPU resume from update 97 reproduces all 303 remaining updates, weights,
  optimizer and sampler state exactly. Separate-process CLI recovery also passes.
- Validation preserves weights, optimizer, gradients, RNG, and module modes.
  Held-out loss worsened **5.806987 → 11.245802**: this is fixture memorization,
  not evidence of improved general English. The English pilot remains untrained.
- Local MPS update/validation/recovery smoke passes at atol=1e-6, rtol=1e-5;
  no full MPS training or bitwise guarantee is claimed.
- Phase 4 Windows/CUDA and GitHub Linux CPU CI: **NOT RUN**. Earlier Windows/CUDA
  PASS and hosted Linux CPU CI PASS belong to Phase 3 only. Independent physical
  Linux validation remains **DEFERRED, NOT PERFORMED**.
- Separate local review completed; actionable findings fixed and checks rerun.
  [Phase 4 report](experiments/phase-4/REPORT.md) records results and limits;
  [training guide](docs/TRAINING.md) defines the resume and batching contracts.

## Pending publication

Remote: `https://github.com/sebastienlato/LatoS.git` (existing private origin).
Destination: `main`. Proposed new annotated tag: **v0.5.0** at the reviewed Phase 4
commit. Existing tags remain fixed; no release, dataset, tokenizer, or weight upload.

- Local commit message: `Complete Phase 4 training engine and recovery`.
  This state belongs to that checkpoint; its exact full ID is supplied in the
  approval request and available from `git log -1 --format=%H`.
- Approval: **PENDING**. No Phase 4 remote writes or tag creation have occurred.
- Next action: stop at the reviewed Phase 4 push-approval checkpoint as requested.
  A later approval authorizes only this described Phase 4 publication. Verify the
  exact remote branch/tag after any approved push and honor the owner's current
  external-validation/transition gate. Phase 5 has not begun.

## Preserved local resources

Existing English corpus audit: 12,595 records, zero exact/near cross-split duplicates.
The accepted English tokenizer and random Phase 3 pilot snapshot still match the
hashes recorded in [the handoff](docs/PHASE4_HANDOFF.md); neither was modified.
Ignored Phase 4 artifacts remain in `outputs/phase-4-acceptance-v2/`, including the
fixture tokenizer, model config, metrics, and interrupted/final checkpoints.
CLI segments remain in `outputs/phase-4-cli/` and `outputs/phase-4-cli-resumed/`.
The earlier acceptance-v1 artifacts are retained as development history.

Keep `.private/` untracked. No new paid services or uploads were used. Local uv is
`.private/tools/bin/uv`; `.venv` uses the runtime under `.private/python`.
