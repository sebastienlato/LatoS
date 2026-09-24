# Changelog

## Phase 17 — Base Model 2.0 execution handoff (local, incomplete)

- Add verified compact full-corpus windows, exact one-pass tail accumulation and
  strict single-pass recovery while retaining legacy training defaults.
- Prepare the bounded RTX 4070 SUPER training/evaluation supervisor, actual-CUDA
  preflight, pinned historical baseline transfer and independent return verification.
- Preserve frozen model/training/evaluation contracts. Mac checks establish
  integration mechanics only; serious training and learned-quality results await
  Windows execution. No publication or Phase 18 work.
- See [execution plan](experiments/phase-17/PLAN.md) and
  [handoff](docs/PHASE17_WINDOWS_HANDOFF.md).

## Phase 16 — Model & CUDA Training 2.0 (local, publication pending)

- Validate native CUDA BF16, numerical/recovery controls and six bounded real-data
  probes on the RTX 4070 SUPER; independently verify returned source and evidence.
- Select the 34,087,424-parameter native dense model under the prespecified smallest-
  eligible rule; larger measured candidates remain feasible, with no learned-quality ranking.
- Account for the entire accepted corpus and freeze a one-pass, one-seed Phase 17
  plan with explicit timing/storage/recovery limits and unchanged quality gates.
- Preserve all historical evidence and unsuccessful validation invocations.
  No new learned-quality claim, Phase 17 run, tag, release or asset publication.
  See [closure](experiments/phase-16/CLOSURE.md) and [training contract](docs/TRAINING_2.md).


## Phase 14 — Evaluation Foundation (local, approval pending)

- Add versioned native evaluation, fixed generation/instruction cases, pinned ARC
  likelihood scoring, matched-text BPB, durable raw records and regression gates.
- Retrospectively measure five unchanged historical models; all score 0/96 on new
  instructions. Preserve negative historical evidence and both reserved tests.
- Add optional PyArrow 25.0.1 and evaluation-only dataset manifests; no training,
  architecture/tokenizer changes or new release. Package remains 1.3.0.
- Phase 13 was published and verified at
  `565bf08fbd8aaa360056fc670b08b8879be9312c`; its pending snapshot below is historical.

## Phase 13 — Repository Repositioning & Roadmap 2.0 (local, approval pending)

- Rebuild the README as a landing page, remove obsolete identity wording from active
  branding and package/CLI descriptions, and add a documentation/evidence index.
- Define finite Phases 13–21 with evaluation-first quality gates, measured CUDA
  scaling, explicit non-goals and a bounded LatoS 2.0 release destination.
- Correct stale release/publication guidance; retain prior roadmap/checkpoint
  snapshots, original experiment evidence and historical release wording.
- Preserve model/training behavior, weights, data, dependencies, lock and package
  version 1.3.0. No new learned experiment or Phase 14 work. No tag/release proposed.

## Publication reconciliation — 2026-09-23

Phase 12 closure is published at `c74290ee1208dc6b0e4077ce3c7b2aad262f3c42`;
correction `b222c1fa844289247531b922d5cb18134e99c7b1` and original implementation
`5b72872a2032f8999aae9c28ff0ab5c704e3d893` are also published. Phase 11 closure
`e2d61100a7c0c74d759bdc4ded5ddcda89cb112b` and earlier phases are published.
All eleven tags remain fixed, including v1.0.0 at
`10d9ef7bf0618b364f887ff9d279408e3cbc33c9`; the stable release has four assets.
Historical entries below retain the pending state when written. This reconciliation
supersedes their publication status only, not any result or platform limitation.

## Phase 12 closure — documentation only (local, approval pending)

- Record owner-reported corrected Windows/CUDA PASS: 301 passed / 13 skips in fresh
  checkout and wheel; native inventory correction, bounded 512-position CUDA checks.
- Separate Windows 16 synthetic initial blocks removed from Mac 44 learned retry
  transitions and scripted controls; no successful learned tool capability.
- Record inspected correction-commit Linux CPU CI: 303 passed / 11 MPS skips within
  synthetic tests/workflows. Physical Linux deferred; original failed run preserved.
- Preserve all research/correction evidence and prepare a fresh-chat Phase 13 handoff.
  Closure publication approval required; no next experiment selected or started.

## Phase 12 portability correction — local, approval pending

- Standardize experiment inventory, source and input paths as POSIX-relative keys;
  include nested condition inventories through the same final inventory writer.
- Add a two-condition regression using real file I/O and simulated Windows path
  serialization; retain completeness, hash and self-exclusion checks.
- Record the owner-reported Windows stop before Phase 12 CUDA execution. Preserve
  original results and evidence without rerunning the research. Fresh retest required.


## Phase 12 — inference context-budget experiment (local, approval pending)

- Compare fixed 256/512 session budgets with identical models, prompt and cases.
  All 44 formerly context-limited sessions gain replies; no learned tool success.
- Record attempt-level token budgets, replayable evidence, independent metric
  recount, reviewed verifier fixes and preservation of all earlier artifacts.
- Add experiment-local code/tests and source-package inclusion; runtime/version
  1.3.0, defaults, dependencies and prior platform/capability limits unchanged.
- Phase 11 closure publication verified; no Phase 12 remote write or new tag.

## Phase 11 closure — documentation only (2026-09-23)

- Record owner-reported Windows RTX 4070 SUPER PASS: 293 passed / 12 skips, fresh
  wheel; Phase 11 50 passes / one MPS skip. Tiny models use a fixed compact prompt
  after original-prompt context rejection; not the full Mac prompting experiment.
- Record separate exact-commit hosted Linux CPU CI: 295 passed / ten MPS skips,
  scoped protocol/scripted/tiny CPU mechanics and workflows/builds. Physical Linux deferred.
- Preserve original Mac results and tiny Windows 0/16 JSON per model, zero calls
  and no learned tool use. Scripted successes remain mechanics only.
- Prepare reviewed closure and fresh-chat Phase 12 handoff; publication approval
  required. No Phase 12 selection/development or runtime/version change.

## 1.3.0 — Phase 11 implementation published, no Phase 11 tag

- Add strict local JSON tool calls, bounded addition/lookup, capped multi-turn runner
  and CLI; keep tokenizer, chat contract and all models unchanged.
- Measure parsing, arguments, task success and failure handling separately. Scripted
  controls pass; unchanged base/SFT/DPO show no learned tool use on fixed cases.
- Retain all responses, context limits, prior evidence, failures and reserved tests.
  No new dependency, paid service, tag, release or asset upload.

## Phase 10 closure — documentation only (2026-09-22)

- Record owner-reported Windows RTX 4070 SUPER scoped PASS: 243 passed / 11 skips,
  fresh wheel suite, four CUDA updates / 16 pairs / 169 response targets.
- Record separately inspected exact-implementation hosted Linux CPU CI: 245 passed /
  9 MPS-only skips, tiny CPU DPO and offline workflows/builds. Physical Linux deferred.
- Preserve Mac ranking regression 16/32 → 15/32 and exact 0/32, distinct from tiny
  Windows ranking 16/32 → 16/32. English remains worse than the original base.
- Prepare reviewed closure and fresh-chat Phase 11 handoff; approval required.
  No Phase 11 work or runtime/version change.

## 1.2.0 — Phase 10 implementation published, no Phase 10 tag

- Add bounded DPO with an immutable SFT reference, response-only sequence scoring,
  documented synthetic preference pairs, and a non-overwriting CLI runner.
- Record held-out SFT and English comparisons, negative preference results, all
  samples, objective verification, preservation and separate review/fixes.
- Preserve all previous artifacts and limitations; no new dependencies or tag.

## Phase 9 closure — documentation only (2026-09-21)

- Record owner-reported Windows/RTX 4070 SUPER PASS at the exact Phase 9 commit:
  226 passed / 10 skips, fresh wheel suite, tiny CUDA LoRA/full tuning at 4 updates
  and 175 targets. Keep tiny 2.44399% and full-architecture 0.851951279% distinct.
- Separately verify exact-commit Linux CPU CI: 228 passed / 8 MPS-only skips;
  tiny CPU mechanisms and offline workflows, not full learned artifacts or a fresh
  installed-wheel suite. Physical Linux remains deferred.
- Preserve amended 1e-4 merge contract and original failure, negative quality,
  unsupported adapter optimizer resume and all scoped platform limits.
- Prepare closure evidence and a fresh-chat Phase 10 handoff. Closure publication
  requires explicit approval; Phase 10 has not begun. No runtime/version change.

## 1.1.0 — Phase 9 implementation published, no Phase 9 tag

- Add bounded float32 attention LoRA training, identity-bound adapter-only snapshots
  and merge into independent dense model snapshots for existing chat/cache paths.
- Compare fixed-budget LoRA and full tuning with unchanged data, tokenizer and chat
  formatting; preserve all samples and prior negative SFT evidence.
- Validate freezing, gradients, reloads, merge/cache agreement, corruption rejection
  and reserved-test exclusion. See [Phase 9 report](experiments/phase-9/REPORT.md).
- Dependency versions remain unchanged. No new platform, useful-assistant,
  general determinism or production-performance claim is introduced.

## 1.0.0 — Phase 8, published 2026-09-21

- Add fresh-checkout reproduction and release guides, data/model cards, and a
  consolidated report preserving the negative SFT result and platform boundaries.
- Prepare source/wheel and a separately labeled MIT tiny fixture model/tokenizer
  package, with explicit file allowlists, hashes, no-overwrite behavior and tests.
- Preserve full base/SFT, English tokenizer, prior evidence, shared chat contract
  and both reserved test payloads. No runtime model/training/inference change.
- Full Mac learned artifacts and acquired corpora remain local. Phase 7 tags stay
  fixed; Phase 8 publication, annotated v1.0.0 and assets require explicit approval.

## Phase 7 closure — documentation only (2026-09-20)

- Record owner-reported Windows/RTX 4070 SUPER PASS at exact v0.8.0: 202 passed,
  eight expected skips, fresh wheel suite and actual cached/uncached CUDA inference.
- Keep the tiny 256-position Windows artifacts and separate synthetic 512-position
  model distinct from the full Mac learned artifacts, which were unavailable there.
  Windows OS-level Ctrl-C/console-event delivery remains unvalidated.
- Record separately inspected hosted Linux CPU CI: 203 passed, seven MPS-only skips,
  tiny inference/CLI/POSIX SIGINT tests, existing offline fixtures and builds.
- Preserve negative quality results and all evidence limits. Physical Linux remains
  deferred. Prepare the Phase 8 handoff without beginning Phase 8.

Closure publication awaits explicit approval. Package remains 0.8.0; no new tag,
tag movement, release or artifact upload. After approved closure publication is
verified, stop; Phase 8 requires an explicit fresh-chat start.

## 0.8.0 — Phase 7, published (2026-09-20)

- Add explicit-artifact terminal chat with streaming Unicode text, JSON events,
  EOS stopping, reply budgets, cancellation and transactional multi-turn history.
- Add an inference-only KV cache with absolute positions, causal offset masks,
  reset and validation; preserve training forward and model-only snapshot format.
- Verify same-backend cached/uncached agreement through 512 positions and measure
  fixed CPU/MPS latency without retraining or consuming either reserved test set.
- Keep the Phase 6 negative instruction result and all platform limitations visible.

Published after owner approval at `04031e5ea98da8db495242165a78c216ab1d4cf4`;
remote main and annotated v0.8.0 were verified. See the
[Phase 7 report](experiments/phase-7/REPORT.md). Closure above has its own pending
publication approval. No Phase 8 implementation is included.

## Phase 6 closure — documentation only (2026-09-20)

- Record owner-reported Windows/RTX 4070 SUPER PASS at exact v0.7.0: 184 passed,
  three MPS-only skips, fresh wheel suite, four CUDA updates / 175 assistant-target
  exposures, 49 verified artifact files and 64 independently replayed responses.
- Record directly verified hosted Linux CPU CI separately: 184 passed, three skips,
  tiny SFT runner/verifier tests, CPU fixture acceptance, offline workflows/builds.
- Preserve the full Mac experiment, its negative quality/regression result and all
  scope limits. The bounded Windows exercise did not reproduce that run or artifact;
  physical Linux remains deferred. No broader CUDA or assistant claim follows.
- Prepare the fresh-chat Phase 7 handoff without beginning Phase 7.

Closure was published and verified at `c4ed8c9564c41f26be2c10ee178f5a10ac80d3a8`.
It kept package 0.7.0 and all tags unchanged. The owner subsequently started
Phase 7 explicitly in a fresh Work chat.

## 0.7.0 — Phase 6, published (2026-09-20)

- Add original synthetic English conversation generation and verified split loading.
- Share chat serialization across tuning and evaluation; hash assistant target masks
  into dataset identities, weight accumulation/exposure by eligible next-token targets.
- Run the fixed 200-update MPS experiment from the selected Phase 5 base, preserve
  all samples and checkpoints, replay recovery, and measure English regressions.
- Record the negative quality result: held-out exact completion remains 0/32 and
  English validation loss worsens. No useful assistant claim.
- Keep dependencies and prior artifacts fixed; version/lock root becomes 0.7.0.

Published after owner approval at `f6636af34933b9678824cc8dd50f6ab6559a2de5`;
remote main and annotated v0.7.0 were verified. See the
[report](experiments/phase-6/REPORT.md). The subsequent closure and Phase 7
inference work are recorded above.

## Phase 5 closure — documentation only (2026-09-20)

- Record owner-reported Windows/RTX 4070 SUPER PASS at exact v0.6.0: 162 tests
  passed, two MPS-only skips, fresh wheel suite, and a bounded four-update CUDA
  exercise with 416 target exposures and 20 verified artifact hashes.
- Record directly verified hosted Linux CPU CI separately: 162 passed, two MPS-only
  skips, tiny CPU Phase 5 tests, fixture acceptance, offline workflows and builds.
- Preserve the full Mac pilot results and limits; neither external check reproduced
  that run. Physical Linux remains deferred; no broader CUDA/quality claim follows.
- Prepare the fresh-chat Phase 6 handoff without beginning Phase 6.

Closure was published at `3878a0167a78f6ad149e8523b4feb97cda77ef5e` and verified
before the explicit fresh-chat Phase 6 start. That closure kept package 0.6.0;
v0.6.0 and v0.5.0 remain fixed.

## 0.6.0 — Phase 5, published (2026-09-20)

- Run the 17.3M-parameter English pilot from verified random initialization on MPS.
- Preserve held-out random/unigram comparisons, fixed samples, token exposure,
  synchronized timing, memory observations, and full recovery checkpoints.
- Add a reproducible pilot runner, separate artifact/recovery verifier, focused
  evidence/failure tests, and reproduction/results documentation.
- Keep test text reserved and all learned artifacts outside Git. Dependencies and
  core training implementation remain unchanged; package/lock version is 0.6.0.

See [Phase 5 results](experiments/phase-5/REPORT.md) for actual metrics and limits.
Published after explicit owner approval at
`fa0c3ac3b7f3890ffdcad411968a656da2f74b3b`; main and annotated v0.6.0 were verified.
The subsequent closure and Phase 6 experiment are recorded above.

## Phase 4 closure — documentation only (2026-09-19)

- Record owner-reported Windows/RTX 4070 SUPER CUDA PASS at exact v0.5.0: 159
  passed, two MPS-only skips, actual training, optimizer updates and recovery.
- Record completed, directly verified GitHub Linux CPU CI separately: 159 passed,
  two MPS-only skips, and offline overfit/exact-resume acceptance.
- Preserve fixture-memorization limits, the absence of a general CUDA determinism
  guarantee, and physical Linux validation as deferred, not performed.
- Prepare the fresh-chat Phase 5 handoff without starting Phase 5.

Closure was published at `52b72c9394e022d4cebc85808343bcaee73857c8` and verified
before the owner-authorized fresh Phase 5 session. That closure retained package
version 0.5.0 and the validated v0.5.0 tag, which remains fixed.

## 0.5.0 — Phase 4, published (2026-09-19)

- Isolated-record batching, token-weighted accumulation, AdamW, warmup/cosine scheduling, and clipping.
- Validation without state changes; atomic tensor-only checkpoints with complete optimizer and shuffle state.
- Training/resume CLI, full offline fixture overfit, exact CPU recovery, and local MPS smoke checks.
- Dependencies unchanged; package version and lock entry advance to 0.5.0.

Published after owner approval at cb585c321c92f5d774fb59234f76c1d3783a635a; main
and annotated v0.5.0 were verified. Phase 5 has not started. The Phase 3 closure below
was subsequently verified published at 294d2e0af2712266b398151eb7e335fa7ccdc31e.

## Phase 3 closure — documentation only (2026-09-19)

- Record owner-reported Windows/CUDA PASS on exact v0.4.2: 131 tests passed and
  one Apple-MPS-only test skipped, with actual RTX 4070 SUPER tensor/model execution.
- Record GitHub Linux CPU CI separately: 131 passed, one MPS-only skip.
- Record independent physical Linux validation as deferred by the owner due to
  machine unavailability; do not substitute CI for that physical-machine test.
- Close Phase 3 locally and prepare the fresh-chat Phase 4 handoff. Publication
  approval is pending, and no Phase 4 development is authorized in this chat.

Package/code version remains 0.4.2; the externally validated tag is unchanged.

## 0.4.2 — Phase 3 fixture checkout correction (published, 2026-09-19)

- Force LF checkout for fixture text and JSON manifests through `.gitattributes`.
- Preserve all existing fixture payload bytes, manifest sizes/hashes, and strict
  runtime verification; include the attributes in the source distribution.
- Exercise actual Git checkout conversion across four settings, then verify
  acquisition/preparation and rejection of deliberately converted CRLF input.

Published after approval and verified at 1102b64714b58c1f2289f80863fa032cc192c47b.
The independent Windows/CUDA retest passed. See the closure entry for limits and
the Phase 4 hold. Existing tags remain fixed.

## 0.4.1 — Phase 3 Windows environment correction (published, 2026-09-18)

- Add Windows x86-64 to the lock and select official PyTorch 2.14.0+cu130 there.
- Preserve macOS/Linux dependency versions, sources, and existing wheel hashes.
- Check platform selection and binary dependency coverage; remove two Windows-only
  assumptions in the tests (Unix memory API availability and default file encoding).
- Record the reported installation blocker and the external retest procedure.

Published after approval at 14c76284d1ec670e4022584638a5030a9da87140. Installation
passed externally, but fixture checkout conversion blocked testing until 0.4.2.
No model code or Phase 4 work was included. The original tag remains fixed.

## 0.4.0 — Phase 3, implemented 2026-09-16; published 2026-09-18

- Original dense decoder, RMSNorm, rotary attention, SwiGLU, and tied output projection.
- Next-token loss, independent numerical references, finite gradients, and causality checks.
- Bounded configurations, verified Safetensors snapshots, and deterministic basic sampling.
- CPU/MPS validation, including a fix for nonfinite values from a combined MPS transfer/cast.

Published after approval at deed6e9b0303967dd82f7a3205e03b9c4f05ffe7. The initial
Windows lock exclusion was corrected in 0.4.1. Weights remain untrained.

## 0.3.0 — Phase 2, published (2026-09-16)

- Original train-only byte-level BPE integration using stable Tokenizers 0.23.2.
- Explicit Unicode, literal-special-token, BOS/EOS, and verified-loading contract.
- Learned 8,192-entry local vocabulary; reproducible artifacts and compression evidence.
- Tokenizer CLI, bounded configurations, tests, and offline CI demonstration.

Published after approval; branch/tag verified and Linux CI passed all 85 tests.
This phase included no language model.

## 0.2.0 — Phase 1, published (2026-09-16)

- Pinned English source manifest and bounded, verified acquisition CLI.
- Deterministic cleaning with whole-document/author splits and exact/near deduplication.
- Reproducible JSONL artifacts, integrity audit, compact corpus evidence, and offline fixture.
- Data tests and an offline CI demonstration; no new Python dependencies.

Published after approval; branch/tag verified and Linux CI passed all 49 tests.
This phase included no tokenizer or model.

## 0.1.0 — Phase 0, published (2026-09-16)

- Installable Python package with CLI help, version, and environment doctor.
- Backend availability reporting and actual float32 arithmetic/gradient checks.
- Locked environment, focused tests, lint/format checks, and CPU CI definition.
- Roadmap, setup instructions, environment evidence, and dependency provenance.

Published after owner approval; remote branch/tag verified and Linux CPU CI passed.
This phase included no model, tokenizer, data pipeline, trained weights, or chat.

## Phase 15 — Data 2.0 (local checkpoint, 2026-09-23)

- Add separately versioned pretraining/instruction construction with pinned source
  terms, deterministic selection, quality filtering, exact/near joins, group splits
  and controlled contamination protection. Original corpora/evaluations remain intact.
- Prepare 37,226,334 pretraining content tokens and 14,259 training conversations;
  select an original 16,384-entry BPE through training-only fitting and new-development
  analysis. No learned model improvement is claimed.
- Add local Phase 16 inputs, token/length/resource accounting, reproducibility checks
  and explicit limitations. Package remains 1.3.0; no tag/release proposed.
