# Project state

Updated: 2026-09-19.

## Phase status

Phase 3 is formally closed locally with Windows/CUDA PASS and an explicit owner
deferral of independent physical Linux testing. Closure documentation is ready
for publication approval. Phase 4 has not begun and must not begin in this chat.

Validated implementation: **v0.4.2**, `1102b64714b58c1f2289f80863fa032cc192c47b`.
This closure changes documentation only; runtime code and package version remain
0.4.2. Existing tags must remain fixed.

## Evidence and limits

- Local Mac CPU/MPS evidence: 132 tests passed in development and clean installation,
  with numerical, snapshot, sampling, dependency, and checkout checks already recorded.
- Independent Windows/RTX 4070 SUPER CUDA: **PASS**, reported by the owner for the
  exact v0.4.2 commit. 131 tests passed, one MPS-only skip. Real CUDA matrix multiply/
  backward and debug/pilot forward, loss, gradients, and causality passed, along with
  CLI, diagnostics, offline data/tokenizer workflows, builds, and archive inspection.
  All 74 tracked files remained unchanged; Windows made no commits or pushes.
  This is external owner evidence; raw logs were not supplied here.
- GitHub-hosted Linux CPU CI: **PASS**, independently inspected at the same commit:
  [run 35463376332](https://github.com/sebastienlato/LatoS/actions/runs/35463376332).
  Python 3.14.7 / PyTorch 2.14.0+cpu; 131 passed and one MPS-only skip, plus locked
  setup, lint/format, CPU/debug-model checks, offline workflows, and package builds.
- Independent physical Linux validation: **DEFERRED, NOT PERFORMED**, explicitly
  directed by the owner because the machine is temporarily unavailable. Hosted CI
  is separate evidence. The earlier both-physical-platforms gate is superseded for
  Phase 3 closure; this is not a physical Linux pass or Phase 4 authorization.
- Models remain randomly initialized. No training engine, optimizer steps, model
  quality evidence, mixed precision, distributed execution, KV cache, or chat yet.
- Consolidated evidence: [Phase 3 closure](experiments/phase-3/CLOSURE.md).

## Pending closure publication

Private origin: `https://github.com/sebastienlato/LatoS.git`; destination: `main`.
The published implementation remains at v0.4.2. Earlier Phase 3 checkpoints v0.4.0
and v0.4.1 remain historical and unchanged.

- Local closure message: `Close Phase 3 with external validation evidence`.
  This state belongs to the closure commit; its full ID is supplied in the approval
  request and available via `git log -1 --format=%H` at this checkpoint.
- Approval: **PENDING**. No closure documentation has been pushed.
- Proposed write: push only this reviewed documentation commit to private main.
  No new tag, tag movement, release, dataset, tokenizer, or weight upload.
- After approval, verify main at the exact closure commit and v0.4.2 still at the
  validated implementation, record the publication locally, and stop in this chat.

## Continuity and next action

Stop for closure publication approval. Phase 4 may start only in a **fresh Work
chat**, with explicit owner authorization after closure publication is verified.
A closure push approval does not authorize development in this chat.
[PHASE4_HANDOFF.md](docs/PHASE4_HANDOFF.md) records the implementation, resources,
next deliverables, acceptance evidence, and publication gate. Physical Linux
validation remains deferred unless the owner later supplies a result.

Retain ignored raw/prepared corpora, `artifacts/tokenizers/english-bpe-v1/`, and
`checkpoints/phase-3-pilot-initial/`. Their recorded identities were checked locally
when preparing this closure. External validation machines are not assumed available
for development or compute. New paid-service budget remains zero.

Local uv: `.private/tools/bin/uv`; `.venv` uses the runtime under `.private/python`.
