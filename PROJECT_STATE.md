# Project state

Updated: 2026-09-20.

## Phase status and owner gate

**Phase 5 is formally closed locally; closure publication awaits explicit approval.**
Implementation is published and verified on main and annotated v0.6.0 at
`fa0c3ac3b7f3890ffdcad411968a656da2f74b3b`. This closure changes documentation/evidence
only; package remains 0.6.0 and the validated tag stays fixed.

**Phase 6 has not begun and must not begin in this chat.** The Windows/CUDA result
satisfies the previous validation wait, but the owner now requires closure publication
and verification followed by explicit Phase 6 start in a fresh Work chat. Closure
push approval alone does not override that gate.

## Evidence and limits

- Mac CPU/MPS, previously recorded: **164 tests passed** in development and a fresh
  non-editable wheel installation. Full 17.3M English pilot: 3,000 updates,
  **5,761,229 target exposures**, validation loss **9.089003 → 4.731898**.
  All samples/measurements/checkpoints are retained; useful assistant capability
  is not established. Selected final update 3,000 stays fixed.
- Independent Windows / RTX 4070 SUPER: **PASS**, owner-reported at exact v0.6.0.
  **162 passed, two MPS-only skips, zero failures**; complete suite also passed in
  a fresh wheel installation. Unchanged runner: **four CUDA updates, 416 target
  exposures**, training/generation/recovery, **20 inventory hashes**, timing/memory
  mechanisms and validation isolation passed. All 108 tracked files stayed unchanged;
  no Windows commits/pushes. Raw Windows logs were not supplied here.
- The bounded Windows exercise **did not reproduce the full Mac English pilot**
  or its exact scores. It does not establish general CUDA determinism, cross-device
  numerical equivalence, sustained-pilot performance, language quality, instruction
  following or Phase 6 capability.
- GitHub-hosted Linux CPU: **PASS**, directly inspected metadata/workflow/logs for
  [run 35515094307](https://github.com/sebastienlato/LatoS/actions/runs/35515094307)
  at the same implementation commit: **162 passed, two MPS-only skips**. Tiny CPU
  Phase 5 tests and Phase 4 CPU fixture acceptance, locked setup, lint/format,
  diagnostics, offline data/tokenizer workflows and builds. No full-pilot run,
  fresh wheel-installed suite, standalone Phase 5 verifier or CUDA test claimed for CI.
- Independent physical Linux: **DEFERRED, NOT PERFORMED**. Hosted CI is separate.
- Consolidated attribution, checks and limits: [Phase 5 closure](experiments/phase-5/CLOSURE.md).
  No new runtime tests, builds or training were performed for this documentation-only closure.

## Preserved inputs

Selected base: `outputs/phase-5-english-pilot/step-00003000/model/`; weight SHA-256
`f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa`.
Preserve all existing corpus/tokenizer/random-baseline files and `outputs/phase-4-*`
and `outputs/phase-5-*` artifacts. They remain ignored local files, not remote backups.
Test text remains reserved. Artifact identities were rechecked for closure.

## Pending closure publication

- Commit message: `Close Phase 5 with external validation evidence`.
  The full closure commit ID is supplied in the approval request and available via
  `git log -1 --format=%H` at this checkpoint; this file belongs to that commit.
- Approval: **PENDING**. The earlier implementation push approval does not authorize
  this closure. No closure documentation has been pushed.
- Remote: existing private `https://github.com/sebastienlato/LatoS.git`; branch `main`.
  **No new tag, tag movement, release or artifact upload.** Keep v0.6.0 fixed and
  retain v0.5.0 at `cb585c321c92f5d774fb59234f76c1d3783a635a`.
- After explicit approval, publish the exact reviewed closure, verify remote main
  and unchanged tags, record verification locally, and **stop**.
- Use [PHASE6_HANDOFF.md](docs/PHASE6_HANDOFF.md) only after verified closure
  publication and an explicit owner start in a fresh Work chat.

No new paid services are authorized; external validation does not grant GPU access.
Local uv: `.private/tools/bin/uv`; accepted runtime remains in `.venv`.
