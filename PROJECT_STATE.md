# Project state

Updated: 2026-09-20.

## Phase status and owner gate

**Phase 7 is formally closed locally; closure publication awaits explicit approval.**
Implementation is published and verified on main and annotated v0.8.0 at
`04031e5ea98da8db495242165a78c216ab1d4cf4`. Closure changes documentation/evidence
only; package stays 0.8.0 and the validated tag remains fixed.

**Phase 8 has not begun and must not begin in this chat.** The owner-reported
Windows/CUDA PASS satisfies the prior external-validation wait. The owner now
requires reviewed closure publication and verification, followed by explicit
Phase 8 start in a fresh Work chat. Closure push approval alone never overrides
that gate. Prepared [Phase 8 handoff](docs/PHASE8_HANDOFF.md) is planning only.

## Evidence and limits

- Mac CPU/MPS, previously recorded: **210 passed** in development and a fresh
  non-editable wheel. Full preserved Mac base/SFT cache agreement through 512
  positions: CPU atol/rtol=1e-5, MPS atol/rtol=1e-4; fixed greedy IDs matched.
  [Original report](experiments/phase-7/REPORT.md) retains latency/memory and scope.
- Windows / RTX 4070 SUPER: **owner-reported PASS** at exact v0.8.0. **202 passed,
  eight expected skips, zero failures**, plus fresh wheel suite and actual CUDA
  inference/CLI checks. Tiny base/SFT artifacts through **capacity 256**, plus a
  **separate synthetic 512-position/batch-two model**. Same-CUDA full-vocabulary
  logits met atol=1e-5, rtol=1e-5; fixed cached/uncached/non-streaming greedy IDs
  agreed. Cache, streaming, stopping, callback cancellation, history, Unicode and
  context checks passed. All 143 tracked and 694 prior Phase 5/6 evidence files
  unchanged on Windows; no commits/pushes there. Raw logs/hashes/latency numbers
  were not supplied here; scoped latency collection is reported without comparison.
- **The full Mac learned artifacts were unavailable and were not validated on
  Windows. Windows OS-level Ctrl-C/console-event delivery remains unvalidated**;
  the POSIX SIGINT test was intentionally skipped. Callback PASS is distinct.
- GitHub-hosted Linux CPU: **separately verified PASS**, run/job metadata,
  exact-commit workflow/tests and logs inspected for
  [run 35529617332](https://github.com/sebastienlato/LatoS/actions/runs/35529617332):
  **203 passed, seven MPS-only skips**. Tiny CPU inference/CLI/POSIX SIGINT and
  existing SFT/recovery fixtures, locked setup, lint/format, offline workflows and
  builds. No full Mac learned-artifact validation, CUDA/MPS, retained-artifact
  latency matrix or fresh-wheel-installed test suite is claimed for CI.
- Physical Linux remains **DEFERRED, NOT PERFORMED**. Useful instruction following,
  general CUDA determinism, cross-device equality, production latency, mixed
  precision and distributed serving remain unestablished.
- Full Mac SFT quality remains negative: **200 updates / 6,188 assistant-target
  exposures**;
  assistant loss **8.354879 → 5.815233**, exact replies **0/32 → 0/32**, English
  loss **4.731898 → 5.653422**. Keep final update 200; SFT is not an improved base.
  512-position mechanics do not establish language quality beyond prior 256 windows.
- [Closure evidence and attribution](experiments/phase-7/CLOSURE.md). **No new runtime
  tests, training, inference benchmarks or builds were run for this closure.**

## Preserved inputs

Base: `outputs/phase-5-english-pilot/step-00003000/model/`, weights SHA-256
`f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa`.
Experimental SFT: `outputs/phase-6-instruction-reviewed/step-00000200/model/`, weights
`62b890214e5544292cc54b725a4e505730fca0d501b3337cdd3f051a622d09ae`.
Tokenizer: `artifacts/tokenizers/english-bpe-v1/`, SHA-256
`7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
Closure rehashed **357 prior Mac artifact/data files**, **83 Phase 7 evidence
entries** and all **49 SFT inventory entries** unchanged; these inventories can
overlap and are not a unique combined count. Keep all sources, random baseline,
attempts, failures, logs and both reserved test payloads. Tests were integrity-
hashed only, never parsed/evaluated. Shared chat format and runtime/lock unchanged.
Ignored local artifacts are not remote backups. Original optimizer recovery still
requires its recorded runtime/implementation.

## Pending closure publication

- Commit message: `Close Phase 7 with scoped external validation evidence`.
  This file belongs to that reviewed commit; exact ID is supplied in the approval
  request and available via `git log -1 --format=%H`.
- Approval: **PENDING**. Prior implementation approval does not cover closure.
  No closure changes have been pushed.
- Existing private remote: `https://github.com/sebastienlato/LatoS.git`; branch `main`.
  **No new tag, tag movement, release or artifact upload.** Preserve all existing
  tags, especially v0.8.0 and v0.7.0; package stays 0.8.0.
- After explicit approval, publish the exact reviewed closure commit, verify remote
  main and unchanged tags, record verification locally, and **stop**.
- Phase 8 requires explicit fresh-chat start after verified closure publication,
  using [PHASE8_HANDOFF.md](docs/PHASE8_HANDOFF.md).

No new paid services. External validation does not grant GPU access. Local uv:
`.private/tools/bin/uv`; locked runtime: `.venv`.
