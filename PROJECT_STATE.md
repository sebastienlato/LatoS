# Project state

Updated: 2026-09-20.

## Phase status and owner gate

**Phase 6 is formally closed locally; closure publication awaits explicit approval.**
Implementation is published and verified on main and annotated v0.7.0 at
`f6636af34933b9678824cc8dd50f6ab6559a2de5`. This closure changes documentation/evidence
only; package stays 0.7.0 and the validated tag remains fixed.

**Phase 7 has not begun and must not begin in this chat.** The Windows/CUDA PASS
satisfies the previous validation wait. The owner now requires reviewed closure
publication and verification, followed by explicit Phase 7 start in a fresh Work
chat. Closure push approval alone does not override that gate.

## Evidence and limits

- Mac CPU/MPS, previously recorded: **187 tests passed** in development and a fresh
  non-editable wheel installation. Full SFT: **200 updates / 6,188 assistant-target
  exposures**. Assistant loss **8.354879 → 5.815233**, exact replies **0/32 → 0/32**,
  English loss **4.731898 → 5.653422**. Preserve the negative result and preselected
  final update 200; the SFT artifact is not an improved base or useful assistant.
- Independent Windows / RTX 4070 SUPER: **PASS**, owner-reported at exact v0.7.0.
  **184 passed, three MPS-only skips, zero failures**, plus fresh wheel suite.
  Bounded CUDA SFT: **four updates / 175 assistant-target exposures**; masking,
  alignment, active clipping, finite state, recovery, 49 artifact checks and 64
  response replays passed. Both tiny models scored 0/32. Both test payloads and
  326 prior Phase 5 evidence files preserved; 128 tracked files unchanged, clean
  tree, no Windows commits/pushes. Raw Windows logs were not supplied here.
- The Windows exercise **did not reproduce the full Mac experiment or its learned
  artifact**. It does not establish useful instruction following, general CUDA
  determinism, cross-device equality, mixed-precision/distributed behavior or
  capabilities beyond the documented mechanisms.
- GitHub-hosted Linux CPU: **PASS**, directly inspected metadata, workflow and logs
  for [run 35522033012](https://github.com/sebastienlato/LatoS/actions/runs/35522033012)
  at the implementation commit: **184 passed, three MPS-only skips**. Tiny CPU SFT
  runner/verifier tests, Phase 4 CPU fixture acceptance, locked setup, lint/format,
  diagnostics, offline workflows and builds. No full Mac SFT reproduction, fresh
  wheel-installed suite, standalone Phase 6 verifier CLI or CUDA result claimed.
- Independent physical Linux: **DEFERRED, NOT PERFORMED**. Hosted CI is separate.
- [Closure evidence and attribution](experiments/phase-6/CLOSURE.md). No new runtime
  tests, builds or training were performed for this documentation-only closure.

## Preserved inputs

Base: `outputs/phase-5-english-pilot/step-00003000/model/`, weight SHA-256
`f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa`.
SFT: `outputs/phase-6-instruction-reviewed/step-00000200/model/`, weight SHA-256
`62b890214e5544292cc54b725a4e505730fca0d501b3337cdd3f051a622d09ae`.
Tokenizer: `artifacts/tokenizers/english-bpe-v1/`, SHA-256
`7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
All 49 SFT inventory files, the base, tokenizer and random baseline were rehashed
unchanged for closure. Keep all Phase 4/5/6 outputs, sources, failures, datasets and
logs; both test payloads remain reserved. These ignored local files are not remote
backups. [Phase 7 handoff](docs/PHASE7_HANDOFF.md) records identities and constraints.

## Pending closure publication

- Commit message: `Close Phase 6 with external validation evidence`.
  The exact closure commit ID is supplied in the approval request and available via
  `git log -1 --format=%H`; this state file belongs to that commit.
- Approval: **PENDING**. The earlier implementation approval does not cover closure.
  No closure changes have been pushed.
- Existing private remote: `https://github.com/sebastienlato/LatoS.git`; branch `main`.
  **No new tag, tag movement, release or artifact upload.** Preserve all existing
  tags, including v0.7.0, v0.6.0 and v0.5.0.
- After explicit approval, publish the exact reviewed closure commit, verify remote
  main and unchanged tags, record verification locally, and **stop**.
- Phase 7 starts only through an explicit fresh-chat request after verified closure
  publication, using [PHASE7_HANDOFF.md](docs/PHASE7_HANDOFF.md).

No new paid services. External validation does not grant GPU access. Local uv:
`.private/tools/bin/uv`; locked runtime: `.venv`. Original optimizer checkpoints
require their original runtime/implementation for recovery.
