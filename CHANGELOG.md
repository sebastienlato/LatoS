# Changelog

## 0.5.0 — Phase 4, local checkpoint (2026-09-19)

- Isolated-record batching, token-weighted accumulation, AdamW, warmup/cosine scheduling, and clipping.
- Validation without state changes; atomic tensor-only checkpoints with complete optimizer and shuffle state.
- Training/resume CLI, full offline fixture overfit, exact CPU recovery, and local MPS smoke checks.
- Dependencies unchanged; package version and lock entry advance to 0.5.0.

Publication approval is pending. Phase 5 has not started. The Phase 3 closure below
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
