# Project state

Updated: 2026-09-19.

## Phase status

**Phase 4 is formally closed locally; documentation publication awaits approval.**
The implementation was published and verified on main and annotated tag v0.5.0 at
`cb585c321c92f5d774fb59234f76c1d3783a635a`. Package/runtime version remains 0.5.0;
this closure changes only documentation and evidence. Keep the validated tag fixed.

**Phase 5 has not begun and must not begin in this chat.** Closure push approval
alone does not override the owner's fresh-chat transition gate.

## Evidence and limits

- Local Mac CPU/MPS: previously recorded **161 tests passed** in development and
  clean wheel installation; fixture overfit, exact CPU recovery, validation state
  preservation, and MPS smoke checks passed.
- Independent Windows / RTX 4070 SUPER: **PASS**, reported by the owner for exact
  v0.5.0. **159 passed, two MPS-only skips, zero failures**; actual CUDA training,
  AdamW, scheduling, accumulation/clipping, validation preservation, checkpointing,
  and separate-process resume passed. All 91 tracked files stayed unchanged and
  Windows made no commits/pushes. Raw Windows logs were not supplied here.
- Windows CUDA fixture loss **5.814638 → 0.010030** over 400 updates. Observed
  exact CUDA equality is a result of that run, **not a general determinism guarantee**.
- GitHub-hosted Linux CPU CI: **PASS**, completed run metadata/workflow/logs directly
  verified at the same implementation commit: [run 35471367201](https://github.com/sebastienlato/LatoS/actions/runs/35471367201).
  **159 passed, two MPS-only skips**, plus locked setup, lint/format, diagnostics,
  offline data/tokenizer workflows, builds, and CPU overfit/exact-resume acceptance.
- Independent physical Linux: **DEFERRED, NOT PERFORMED**. Hosted CI is distinct evidence.
- Held-out loss worsening remains fixture memorization, not general language
  improvement. The English pilot remains untrained. No Phase 5 execution occurred.
- Consolidated evidence and limits: [Phase 4 closure](experiments/phase-4/CLOSURE.md).

## Pending closure publication

Remote: `https://github.com/sebastienlato/LatoS.git` (existing private origin).
Destination: `main`. **No new tag, tag movement, release, dataset, or weight upload.**

- Local closure message: `Close Phase 4 with external validation evidence`.
  This state belongs to the closure commit; its full ID is supplied in the approval
  request and available via `git log -1 --format=%H` at this checkpoint.
- Approval: **PENDING**. The earlier implementation push approval does not authorize
  this new closure commit. No closure documentation has been pushed.
- After explicit approval, publish the exact reviewed closure, verify main at that
  commit and v0.5.0 still at the validated implementation, record locally, and stop.
- Start Phase 5 only in a **fresh Work chat**, with explicit owner instruction after
  closure publication is verified. [PHASE5_HANDOFF.md](docs/PHASE5_HANDOFF.md)
  contains the preserved inputs, objective, constraints, and reusable start prompt.

## Preserved resources and closure checks

Preserve the ignored English corpus, accepted 8,192-entry BPE tokenizer, random
Phase 3 pilot snapshot, and `outputs/phase-4-*` acceptance/CLI artifacts. Tokenizer
and pilot identities were checked again when preparing closure; their hashes are
in the handoff. External validation does not grant access to Windows GPU compute.
No new paid services or artifact uploads are authorized.

Closure review checks evidence attribution, links, commit/test-count consistency,
unchanged runtime/dependencies/configuration, privacy, and the transition gate.
No new runtime tests or training are claimed for this documentation-only closure.
Local uv: `.private/tools/bin/uv`; `.venv` uses the runtime under `.private/python`.
