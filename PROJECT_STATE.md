# Project state

Updated: 2026-09-23.

## Active checkpoint

**Phase 12 Windows path-portability correction is validated and separately reviewed locally; publication
approval is required. Do not close Phase 12 or start another extension until the
owner supplies a fresh independent Windows/CUDA result on the RTX 4070 SUPER.**

Original Phase 12 is published at
`5b72872a2032f8999aae9c28ff0ab5c704e3d893`; exact remote main and all eleven unchanged
tag objects/targets were verified before correction. Verified publication supersedes
historical pending snapshots. v1.0.0 stays at
`10d9ef7bf0618b364f887ff9d279408e3cbc33c9`, object
`1acb6b94f0adfbca85014ad6601dbfc716e3c651`.

Owner-reported Windows validation **failed at the first project failure**: a mixed
native/POSIX inventory-path representation. Fresh exact checkout and locked install
passed; prescribed suite **299 passed / 13 expected skips / zero failures/errors**.
CUDA detection succeeded, but actual Phase 12 CUDA execution was not reached.
Owner reports 229 tracked and 3,045 prior evidence/report files unchanged; reserved
payloads untouched; no Windows fixes, commits or pushes. This is not a dependency,
CUDA, model-quality or missing-artifact failure. Raw handoff logs were not supplied
locally; the detailed owner message is the evidence source.

Correction uses portable POSIX-relative evidence keys and a shared final inventory
writer. A nested two-condition regression reproduces the original failure with
simulated Windows separators and passes after the fix. No research rerun is needed;
original hypothesis, results, prompts, models, scoring, context limits, tolerances
and evidence remain unchanged. See [correction](experiments/phase-12/PORTABILITY_CORRECTION.md).
Mac development and fresh isolated non-editable wheel suites each pass **314 tests**;
nine focused checks pass, including both nested inventory variants. All **78,404**
prior artifact/evidence files match; only the two authorized working experiment
source files differ in the 78,406-file snapshot. Original Phase 12 evidence unchanged.
See [validation](experiments/phase-12/portability-correction.json) and
[separate review](experiments/phase-12/PORTABILITY_REVIEW.md).
Physical Linux remains deferred; no new hosted Linux result is claimed.

One research extension: a fixed **256-versus-512 inference session budget** within
the preserved models' existing capacity. See [plan](experiments/phase-12/PLAN.md),
[report](experiments/phase-12/REPORT.md), [results](experiments/phase-12/results.json),
[verification](experiments/phase-12/verification.json), and
[separate review](experiments/phase-12/REVIEW.md).

- Baseline sessions/scores/metrics reproduce Phase 11 exactly; all 44 previously
  context-limited sessions emit additional replies at 512. H1 passes.
- H2 fails: base/SFT/DPO still have zero valid JSON, zero calls, 0/12 tasks and 0/4
  learned failure handling each. At 512, parsing is 0/41, 0/64, 0/64 emitted replies.
  More room for retries is not learned tool use or general long-context quality.
- Two bounded runs agree on raw responses/attempts/scores/metrics: 2,915 tokens each,
  5,830 total; zero training/new services. Original prompt and chat contract stay fixed.
- Experiment code/tests/evidence only, plus source-package inclusion. Public runtime,
  defaults, version 1.3.0, dependencies and lock remain unchanged. Mac CPU/MPS tests;
  full learned evaluation MPS only. Original full evaluation is Mac-only; the external Windows stop is recorded above.
- Validation: 312 tests in development and fresh non-editable wheel; evidence replay
  covers 96 learned sessions and independent recount covers 128 including controls.
  Full details and retained initial test failure: [validation](experiments/phase-12/validation.json).
- All 78,036 pre-existing inventory files preserved, including older models/evidence
  and both opaque-hashed reserved tests. No reserved payload parsed or evaluated.

## Prior scoped evidence and preserved limits

- **Owner-reported Windows RTX 4070 SUPER PASS:** 293 passed / 12 expected skips /
  zero failures, fresh exact checkout/locked install and fresh non-editable wheel.
  Phase 11: 50 passes / one MPS-only skip in each environment. Parsing, schemas,
  bounds, failures, separate metrics, CLI/packaging/privacy and scoped side-effect
  audit pass. Scripted controls: 12/12 normal tasks and 4/4 expected failures handled.
- Windows retained **tiny** base/SFT/DPO, 256 capacity; base is a random fixture.
  Original system prompt plus cases correctly rejected with context_limit before
  generation. **One fixed compact prompt** enabled 1,024 CUDA tokens/model across
  16 cases. Other controls unchanged per owner. Each model: **0/16 JSON**, no calls,
  **0/12 tasks and 0/4 learned failure handling**. Not the full Mac prompting experiment;
  full Mac artifacts/tokenizer unavailable. Exact compact text/hash, raw Windows logs,
  artifact hashes, full skip breakdown and numerical maxima not supplied.
- **Separately inspected hosted Linux CPU** [run 35862021715](https://github.com/sebastienlato/LatoS/actions/runs/35862021715):
  295 passed / ten MPS-only skips, exact implementation; Phase 11 50 CPU passes / one
  skip. Protocol/scripted/tiny CPU inference and existing workflows/builds. Verifier
  fixtures labeled base/SFT/DPO are scripts, not real model evaluation. No full Mac
  experiment, Windows compact-prompt exercise, CUDA/MPS or fresh installed-wheel suite.
  Dense fixture recovery is not tool learning or adapter/DPO resume. Physical Linux deferred.
- **Original Mac unchanged:** base/SFT/DPO each 0/16 JSON, no calls, 0/12 tasks and
  0/4 learned failure handling; 595/48/51 generated tokens. Twelve base retry-context
  failures plus four incomplete replies; SFT/DPO sixteen retry-context failures each.
  No second learned turn. Scripted 12/12 and 4/4 are mechanics only. Development and
  isolated wheel each passed 305 tests. Learned tool use remains **unestablished**.
- **59,999 local files rehashed unchanged**, including the prior 59,873 snapshot and
  126 additional distinct Phase 11 run/inventory/log/original evidence files. Both
  reserved tests opaque-hashed only, never parsed/evaluated. Windows preservation is
  separately owner-reported: 213 tracked and 2,543 prior regular report/evidence files,
  clean tree, no Windows commit/push. Preserve all models, tokenizer and chat contract 1.
- [Phase 10 closure](experiments/phase-10/CLOSURE.md) retains full Mac negative DPO:
  100 updates / 400 pairs / 2,498 targets, ranking 16/32 → 15/32, exact 0/32 → 0/32;
  English 5.653422 → 5.603780 remains worse than original base 4.731898. Distinct tiny
  Windows four-update/16-pair/169-target ranking 16/32 → 16/32 and separately scoped
  hosted Linux evidence remain unchanged. Negative SFT/LoRA/control results retained.
- Merge atol=rtol=1e-4 remains amended; original CPU 1e-5 failure retained. Cache
  bounds CPU/CUDA 1e-5, MPS 1e-4. Adapter/DPO optimizer resume unsupported; dense
  recovery version-bound. Windows OS-level Ctrl-C remains unvalidated. No useful
  instruction/tool following, factual reliability, safety alignment, general device
  determinism, cross-device equality, production performance, mixed precision,
  distributed serving or learned language quality beyond 256 is established.

## Pending publication and next action

- Commit message: `Fix Phase 12 evidence path portability`.
- Exact reviewed correction commit will be recorded in the approval request and
  `.private/phase12-portability-publication.json` after preparation.
- Existing remote: `https://github.com/sebastienlato/LatoS.git`, destination **main**,
  public visibility unchanged. No tag, release, asset upload or remote backup.
- Correction approval pending; no correction remote write. Stop and ask
  **Push Phase 12 to GitHub?** Approval applies only to the exact correction.
- After approved publication, verify main/tags and wait for the owner's fresh
  Windows/CUDA retest. No Phase 12 closure or subsequent experiment before that result.

Local uv: `.private/tools/bin/uv`; runtime `.venv`. Original experiment attempts and
raw logs remain ignored and unchanged. Correction logs are separate under
`outputs/phase-12-portability-validation`; neither is a remote backup.
