# Phase 4 continuity handoff

## Activation recorded 2026-09-19

The owner explicitly started Phase 4 in a fresh Work chat. Before development,
read-only remote inspection verified the closure commit
`294d2e0af2712266b398151eb7e335fa7ccdc31e` on `origin/main` and the unchanged v0.4.2
tag. This satisfied the original fresh-chat transition gate. The handoff below
records the starting inputs and scope; [PROJECT_STATE.md](../PROJECT_STATE.md)
now controls continuity. Phase 4 stops at its reviewed push-approval checkpoint.

## Read and preserve

Read `AGENTS.md`, `PROJECT_STATE.md`, `ROADMAP.md`, and the local private work
context/blueprint when present. Keep private planning outside tracked files and
GitHub metadata. `PROJECT_STATE.md` records the pending/verified closure publication.

Validated implementation: v0.4.2 at
`1102b64714b58c1f2289f80863fa032cc192c47b`. The owner reports Windows/RTX 4070 SUPER
CUDA PASS (131 passed, one MPS-only skip, actual CUDA tensor and debug/pilot model
execution). GitHub Linux CPU CI passed separately (131 passed, one MPS-only skip).
Independent physical Linux validation is deferred by the owner because the machine
is unavailable, not completed. See [closure evidence](../experiments/phase-3/CLOSURE.md).

Current implementation: original pre-RMSNorm/rotary/SwiGLU causal decoder, tied
input/output weights, internally shifted next-token loss with -100 masking, bounded
sampling, and verified model-only snapshots. Debug: 631,104 parameters, context 128.
Pilot: 17,308,032 parameters, context 512. Neither has been trained.

Local artifacts are ignored and must be preserved:

- `data/processed/english-books-v1/`: fixed train/validation/test splits. Train the
  model only on training text; validation is for development and test remains reserved.
- `artifacts/tokenizers/english-bpe-v1/`: 8,192 entries; tokenizer SHA-256
  `7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
- `checkpoints/phase-3-pilot-initial/`: random seed 17; weight SHA-256
  `99dce7bf0356bf7d641dff4b29d6616772dfbf9779877e0c085bdea9e368b5f2`.
  This contains no optimizer, scheduler, or resumable run state.

Local uv is `.private/tools/bin/uv`; `.venv` uses Python 3.14.7 under `.private/python`.
Use the committed lock. Inspect actual hardware/resources; do not assume access to
the external NVIDIA validation machine. The available Mac has previously supported
CPU/MPS checks. Recheck availability in the fresh execution environment. Record and
test any necessary dependency changes while preserving accepted experiment identities.
The new paid-service budget remains zero; no artifact uploads are authorized by this handoff.

## Phase 4 objective

Build batching, AdamW optimization, learning-rate warmup/decay, gradient accumulation,
clipping, metrics, validation, and complete checkpoint/resume support. Define target
masking/document boundaries explicitly against the existing internally shifted loss.
Prove tiny-fixture overfit, uninterrupted versus resumed equivalence on a named
backend with a stated guarantee, and validation that does not modify weights.

Own implementation, meaningful tests, separate review, fixes, documentation, and
the local commit. Keep experiments small and within actual resources. Record failures
and unexecuted paths honestly. Do not advance to Phase 5 or publish Phase 4 without
its own concrete checkpoint and owner push approval. Preserve the owner's external
validation/transition gate at later checkpoints; the Phase 3 physical Linux deferral
must never be recorded as a completed test.
