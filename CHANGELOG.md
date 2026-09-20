# Changelog

## Phase 5 closure — documentation only (2026-09-20)

- Record owner-reported Windows/RTX 4070 SUPER PASS at exact v0.6.0: 162 tests
  passed, two MPS-only skips, fresh wheel suite, and a bounded four-update CUDA
  exercise with 416 target exposures and 20 verified artifact hashes.
- Record directly verified hosted Linux CPU CI separately: 162 passed, two MPS-only
  skips, tiny CPU Phase 5 tests, fixture acceptance, offline workflows and builds.
- Preserve the full Mac pilot results and limits; neither external check reproduced
  that run. Physical Linux remains deferred; no broader CUDA/quality claim follows.
- Prepare the fresh-chat Phase 6 handoff without beginning Phase 6.

Closure publication approval is pending. Package remains 0.6.0; v0.6.0 and v0.5.0
stay fixed. After approved closure publication is verified, stop; Phase 6 requires
explicit owner start in a fresh Work chat.

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
No Phase 6 work has begun. The closure above has its own pending approval.

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
