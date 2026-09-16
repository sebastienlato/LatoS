# Project state

Updated: 2026-09-16.

## Active phase

Phase 3 — original dense transformer complete locally; publication approval pending.

Implemented bounded model configurations, pre-RMSNorm causal attention with rotary
positions, SwiGLU, tied input/output embeddings, next-token loss, Safetensors model
snapshots, and bounded sampling. Package 0.4.0; Safetensors 0.8.0 is locked.

## Evidence and limits

- Actual unique parameters: debug 631,104; pilot 17,308,032. Both use the pinned
  8,192-entry tokenizer. Counts match an independent analytical formula.
- 118 tests passed in development and fresh built-wheel environments: numerical
  equations/gradients, causal prefixes and gradients, loss alignment/masking,
  shapes, binding, snapshots, invalid inputs, and sampling. Lint/format/build,
  dependency compatibility, archive contents, and workflow parsing also passed.
- Debug and pilot forward/backward passed on CPU and MPS in float32. Pilot also
  passed at its full 512-position context. No optimizer steps were run.
- Debug CPU/MPS max differences: logits 2.3842e-7, loss 9.5367e-7, gradients 4.6194e-7.
  Same-mode causal-prefix differences were zero on both provided configurations.
- Pilot weights and CPU logits round-trip exactly. CPU and MPS bounded sampling
  passed. Review fixed MPS combined-transfer/float64 corruption, tiny-temperature
  probability handling, logit mutation, and causal diagnostic execution-mode mismatch.
- Separate review complete. Detailed limits, timings, hashes, and evidence:
  [Phase 3 report](experiments/phase-3/REPORT.md), [model guide](docs/MODEL.md).
- Weights are random, without language-quality evidence. No optimizer/training
  engine, padding-attention mask, KV cache, or chat. CUDA and mixed precision are
  untested; Phase 3 Linux CI awaits publication.

## Published checkpoints

Private origin: `https://github.com/sebastienlato/LatoS.git`.

- Phase 0: `a33d84d0200a3e46879bc506a3f8c9a82db2ff7c`, v0.1.0, Linux CI 22 tests.
- Phase 1: `fe9818b983aac00d5bba8e2a7b7686637f9e4a7a`, v0.2.0, Linux CI 49 tests.
- Phase 2: `1ed47127c05d8d6de0095f688fd3ad5f24c62b48`, v0.3.0; main/tag verified
  after explicit approval. [Linux CI](https://github.com/sebastienlato/LatoS/actions/runs/35149689542)
  passed 85 tests and offline tokenizer checks.

## Pending publication

- Local branch: `main`; message: `Complete Phase 3 dense transformer`.
  Checkpoint is the commit containing this state; full ID is in the approval
  request and available from `git log -1 --format=%H` at this checkpoint.
- Proposal: push this reviewed commit to existing private `sebastienlato/LatoS`
  main, with annotated tag `v0.4.0`. No releases, access changes, or weight uploads.
- Approval: **not granted for Phase 3**; prior phase approvals do not cover it.
- Preserve ignored `checkpoints/phase-3-pilot-initial/` (random seed 17): weights
  SHA-256 `99dce7bf0356bf7d641dff4b29d6616772dfbf9779877e0c085bdea9e368b5f2`.
- Preserve `artifacts/tokenizers/english-bpe-v1/`: tokenizer SHA-256
  `7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
- Retain ignored `data/raw/english-books-v1/` and `data/processed/english-books-v1/`.

## Next action

Await “Push Phase 3 to GitHub?” approval. On yes, publish the exact checkpoint,
verify remote branch/tag and inspect CI, then begin Phase 4: batching, optimizer,
schedule, validation, checkpoint/resume, tiny-fixture overfit, and uninterrupted
versus resumed comparison on a defined backend. Model-only Phase 3 snapshots do
not contain resumable training state. Phase 4 publication needs its own approval.

Local uv: `.private/tools/bin/uv`; `.venv` uses the runtime under `.private/python`.
