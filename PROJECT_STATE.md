# Project state

Updated: 2026-09-21.

## Active checkpoint

**Phase 8 is complete and reviewed locally; publication approval is PENDING.**
The owner explicitly started this phase in a fresh Work chat. Remote Phase 7 closure
was verified at `e4fd79ba7ab1fe5bcc19e733c2f6639a2214ba94`; annotated v0.8.0 remains
at `04031e5ea98da8db495242165a78c216ab1d4cf4`, object
`485aaf36b7ed81a95c948434e1c14634b220be07`. No Phase 8 remote write has occurred.

Delivered: [reproduction/release guide](docs/RELEASE.md), [data card](docs/DATA_CARD.md),
[model card](docs/MODEL_CARD.md), [report](experiments/phase-8/REPORT.md),
[validation](experiments/phase-8/validation.json), [separate review](experiments/phase-8/REVIEW.md)
and a hash-pinned tiny fixture packager. Version is 1.0.0; runtime implementation,
shared chat contract and dependency versions remain unchanged.

## Validation and preserved limits

- Mac M4 Max, 64 GiB, CPU/MPS, Python 3.14.7 / PyTorch 2.14.0: **213 tests passed**
  in development, an isolated tracked checkout and its non-editable wheel.
- Documented tiny 400-update CPU acceptance and exact 303-update recovery passed;
  five tiny artifact files reproduced byte-for-byte. Actual vocabulary 328,
  127,808 parameters, capacity 64. Fixture train loss 5.814638 → 0.010030 but
  validation worsened 5.806987 → 11.245802: memorization only.
- Both retained full Mac models loaded/chatted from the wheel on CPU/MPS. Lint,
  build, archive inspection, documentation and privacy checks passed. Final commit
  clone/asset verification is recorded locally before the approval request.
- **441 distinct prior files** rehashed unchanged, including overlapping 25/49/83
  Phase 5/6/7 inventories and both reserved test payloads. Tests were integrity-
  hashed only, never parsed/evaluated. No full retraining or old optimizer replay.
- Full Mac base/SFT/tokenizer identities remain exactly as in the model card and
  Phase 8 handoff. Preserve all prior attempts/failures, random baseline and logs.
  Ignored inputs are not remote backups; original optimizer recovery requires its
  recorded source/runtime. Full learned artifacts and acquired corpora stay local.
- Negative SFT remains: 200 updates / 6,188 assistant-target exposures; assistant
  loss 8.354879 → 5.815233; exact replies 0/32 → 0/32; English loss 4.731898 →
  5.653422. Keep final update 200; do not promote it as an improved base.
- Historical Windows/CUDA Phase 7 owner-reported PASS used tiny 256-position
  artifacts plus a synthetic 512-position/batch-two model, **not the full Mac
  learned artifacts**. Windows console Ctrl-C remains unvalidated. Hosted Linux
  CPU CI passed separately within documented tiny CPU/workflow scope. Physical
  Linux is deferred. No new Phase 8 Windows/Linux execution is claimed.
- Useful instruction following, general CUDA determinism, cross-device equality,
  production latency, mixed precision, distributed serving and language quality
  beyond prior 256-token windows remain unestablished.

## Concrete pending publication

- Local commit message: `Prepare Phase 8 reproducible educational release`.
  This state is included in that commit; its exact ID is in the approval request
  and `git log -1 --format=%H` once prepared.
- Existing remote: `https://github.com/sebastienlato/LatoS.git`; branch **main**.
  Current read-only GitHub lookup reports **public**, superseding earlier private
  descriptions. No visibility change is proposed.
- Proposed annotated tag **v1.0.0** and GitHub release
  **LatoS v1.0.0 — reproducible educational release** using the reviewed
  [notes](experiments/phase-8/RELEASE_NOTES.md).
- Four assets in ignored `dist/phase-8-release/`: `latos-1.0.0.tar.gz`,
  `latos-1.0.0-py3-none-any.whl`, `latos-1.0.0-tiny-fixture.zip`, `SHA256SUMS`.
  Source/wheel plus MIT tiny fixture model/tokenizer only; no acquired data,
  full Mac learned artifacts, optimizers, logs, environments or private context.
- Preserve all ten existing tags. No PyPI upload, new paid service or remote backup.
- **Next action: stop for explicit Phase 8 publication approval.** Publish only
  the exact reviewed commit/assets if approved and verify remote identities.
  Phase 9 has not begun. This request ends at the Phase 8 approval checkpoint;
  no later phase publication is authorized.

Local uv: `.private/tools/bin/uv`; locked runtime: `.venv`. Final exact commit,
asset hashes and publication status are retained in `.private/phase8-publication.json`.
