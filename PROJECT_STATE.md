# Project state

Updated: 2026-09-20.

## Phase status and gate

**Phase 6 is complete locally; reviewed publication approval is pending.** The
owner explicitly started this phase in a fresh Work chat after verified Phase 5
closure `3878a0167a78f6ad149e8523b4feb97cda77ef5e`. Remote main matched that commit;
v0.6.0 remains fixed at `fa0c3ac3b7f3890ffdcad411968a656da2f74b3b`.

Original synthetic conversations, a shared chat formatter, assistant-only target
masking, measured SFT, held-out base comparisons and English regression checks are
implemented. Package is 0.7.0; dependency versions are unchanged. Separate review
and fixes are complete. **Stop at the Phase 6 push-approval checkpoint.** No Phase 7
work is included and no Phase 6 remote write has occurred.

## Outcome and validation

- 192 training / 32 validation / 32 reserved test conversations. Fixed 200-update
  MPS experiment: **6,188 assistant-target exposures**, 744 distinct positions.
- Assistant validation loss **8.354879 → 5.815233**, but **0/32 exact replies for
  both base and SFT**. EOS stops improve 0/32 → 32/32. English all-target validation
  loss **4.731898 → 5.653422**. This is a negative quality/regression result, not a
  useful assistant or replacement base. Final update 200 was selected beforehand.
- **187 tests passed** in development and a fresh non-editable wheel installation.
  Lint, formatting, build and packaging/privacy inspections passed. Tiny masked CPU
  recovery is exact on the tested host/runtime; MPS recovery uses stated tolerance.
- Independent process verifier: 49 inventory files, all 200 updates/6,188 targets,
  both objectives recomputed with zero recorded loss difference, all 64 responses
  replayed exactly. MPS update 100→200 maximum weight difference **1.1921e-7**,
  within atol=1e-6, rtol=1e-5. No broader determinism claim.
- 19,806 prior artifact files verified unchanged. Both test payloads remain reserved
  from tuning/evaluation. Initial data-writing failure, calibration and pre-review
  pilot are retained; the reviewed repetition did not change the fixed protocol.
- Phase 6 Windows/CUDA and hosted Linux execution **not performed**. Prior owner
  Windows/RTX 4070 SUPER PASS was only four CUDA updates / 416 targets, not the full
  Mac English pilot. Prior hosted Linux CPU CI passed separately in its documented
  Phase 5 scope. Physical Linux remains deferred. No general CUDA determinism or
  cross-device equivalence is established.

Details: [report](experiments/phase-6/REPORT.md), [review](experiments/phase-6/REVIEW.md),
[validation](experiments/phase-6/validation.json), [guide](docs/INSTRUCTION_TUNING.md).

## Preserved artifacts

Base: `outputs/phase-5-english-pilot/step-00003000/model/`, weight SHA-256
`f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa`.
Tokenizer: `artifacts/tokenizers/english-bpe-v1/`, SHA-256
`7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
All Phase 4/5 outputs, corpus splits and random baseline remain preserved.

Reviewed SFT: `outputs/phase-6-instruction-reviewed/step-00000200/model/`, weight
SHA-256 `62b890214e5544292cc54b725a4e505730fca0d501b3337cdd3f051a622d09ae`.
Run inventory: [artifacts.json](experiments/phase-6/artifacts.json), SHA-256
`f0d88d3de0ab3da03d7c9f8734aea3de24f7191afc9ea76499d5c7bfccda01c7`.
The ignored run retains exact package source, lock, all samples/metrics, and
initial/update 100/update 200 recovery checkpoints. Generated conversations are
in `data/processed/english-instructions-v1/`. These local artifacts are not remote
backups. Preserve them, including negative results and prior attempts.

## Pending publication

- Reviewed commit message: `Complete Phase 6 measured instruction-tuning experiment`.
  The exact commit ID is supplied in the approval request and `git log -1 --format=%H`;
  this state file belongs to that commit.
- Approval: **PENDING**, specific to Phase 6. Earlier phase approvals do not apply.
- Existing private remote: `https://github.com/sebastienlato/LatoS.git`.
  Destination branch: `main`. Proposed annotated tag: **v0.7.0** on that exact commit.
- No release, artifact upload, repository setup or old-tag movement is proposed.
  Retain v0.6.0 and v0.5.0 unchanged.
- Next action: present the exact reviewed local commit and ask **“Push Phase 6 to
  GitHub?”** Wait for approval before any remote write. If approved, publish that
  exact state and verify branch/tag, honoring any new owner transition gate.

No new paid services. Local uv: `.private/tools/bin/uv`; locked development runtime:
`.venv`. Old optimizer checkpoints require their original package/runtime;
Phase 6 is a fresh objective initialized from model weights only.
