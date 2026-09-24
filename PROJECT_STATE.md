# Project state

Updated: 2026-09-23.

## Active checkpoint

**Phase 16 — Model & CUDA Training 2.0: Mac implementation complete for external
validation; the phase remains open.** The owner explicitly authorized Phase 16,
then directed a self-contained separate Windows Work handoff because this Mac has
no authenticated connection to the RTX 4070 SUPER. Stop at that external checkpoint.
No final training configuration is selected. **Do not begin Phase 17.**

Published Phase 15 starting state was verified at
`d3c63d8655415252ac4a9efac023107831e1ad75`: HEAD/main/origin/main and live remote main
matched, clean and zero ahead/behind; eleven tags and the existing release/four
asset records were unchanged. The verified local publication record supersedes
Phase 15's tracked pending-publication snapshot. Existing package remains **1.3.0**;
**v1.0.0** remains at `10d9ef7bf0618b364f887ff9d279408e3cbc33c9`.

## Completed locally

- [Bounded plan and limits](experiments/phase-16/PLAN.md): independent native dense
  depth comparison, fixed inputs/budgets/tolerances, CUDA resource/inference targets
  and selection rule. Candidate sizes are comparison points, not accepted models.
- Explicit CUDA BF16 autocast with float32 parameters/optimizer state; strict
  precision-aware checkpoint recovery, finite-update checks, synchronized CUDA
  timing and refusal to substitute CPU/MPS for CUDA.
- Verified Data 2.0 document coalescing and isolated overlap-one probe windows.
  Actual selected training subset: 4,096 documents, 1,943,015 targets, 6,105 windows.
  Fixed-slot utilization 22.14% isolated versus 62.28% coalesced; no omissions.
  [Input accounting](experiments/phase-16/input-check.json) and
  [identical repeat](experiments/phase-16/input-repeat.json). No GPU speed claim.
- Bounded worker suite with original synthetic numerical controls, real-data
  candidate runs, midpoint recovery, validation, memory/timing/checkpoint/inference
  records, source snapshots and retained failures. Full Phase 17 data scaling is
  not established by the bounded probe adapter.
- [Separate local review](experiments/phase-16/REVIEW.md) and fixes; reviewed local
  source checkpoint and explicit-input transfer archive prepared for Windows.
  [Windows Work handoff](docs/PHASE16_WINDOWS_HANDOFF.md) defines the next action.

## Actual validation and limits

[Validation](experiments/phase-16/validation.json): **402 passed / 1 CUDA skip** in
both development and a fresh non-editable wheel installation. Lint/format, unchanged
lock, dependency compatibility, build/package inspection and CPU doctor pass.
All **73,679** historical inventory files rehash unchanged; all 12 recorded Phase 15
artifacts and all 17 packaged input files match. Two actual bounded input preparations
produce identical selection, windows, token accounting and identities.

A real CLI attempt on this Mac refuses CUDA before running controls or candidates;
its failure is retained. No Windows/CUDA measurement, real-corpus model optimization,
new learned capability, new CI result or Phase 17 run is claimed. Mac tests include
original disposable synthetic optimization. Physical Linux remains deferred.
New paid-service budget remains zero. No dependency upgrade or access change.

Phase 15 corpus/tokenizer and fixed Phase 14 gates remain unchanged. Historical
base: **1.872944 BPB**, ARC-Easy **157/570**, Challenge **66/299**; all five artifacts
**0/96 instructions**. Preserve original SFT/adaptation/DPO/tool negatives.

## Publication status and next action

This is an **external-validation checkpoint, not the Phase 16 push checkpoint**.
The local source commit is not published; its exact identity and the transfer ZIP's
checksum belong in the handoff message and ignored local record. Existing remote is
`https://github.com/sebastienlato/LatoS.git`, eventual destination **main**.
No remote branch, PR, tag, release, asset upload or other GitHub write is authorized.

Run the exact reviewed transfer package in a separate Windows Work session on the
existing RTX 4070 SUPER. Return actual evidence and all failures for Mac review.
Do not close Phase 16 or choose the final training configuration until its required
CUDA criteria are supported. Finish the measured Phase 17 budget/recovery plan,
review and fixes, then prepare the final local phase commit and ask
**“Push Phase 16 to GitHub?”** with its concrete publication details. No such push
approval is requested at this incomplete checkpoint. **Phase 17 remains prohibited.**

Local uv: `.private/tools/bin/uv`; runtime `.venv`. Mac remains authoritative for
development and publication. Private context and acquired/learned artifacts stay
out of normal Git history; the local Windows transfer is not a public release.
