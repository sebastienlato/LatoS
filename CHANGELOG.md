# Changelog

## 0.4.2 — Phase 3 fixture checkout correction (local, 2026-09-19)

- Force LF checkout for fixture text and JSON manifests through `.gitattributes`.
- Preserve all existing fixture payload bytes, manifest sizes/hashes, and strict
  runtime verification; include the attributes in the source distribution.
- Exercise actual Git checkout conversion across four settings, then verify
  acquisition/preparation and rejection of deliberately converted CRLF input.

Publication awaits approval. Native Windows/CUDA retesting and Phase 4 remain paused.
The original v0.4.0 and v0.4.1 tags must remain fixed.

## 0.4.1 — Phase 3 Windows environment correction (local, 2026-09-18)

- Add Windows x86-64 to the lock and select official PyTorch 2.14.0+cu130 there.
- Preserve macOS/Linux dependency versions, sources, and existing wheel hashes.
- Check platform selection and binary dependency coverage; remove two Windows-only
  assumptions in the tests (Unix memory API availability and default file encoding).
- Record the reported installation blocker and the external retest procedure.

Publication awaits approval. Windows/CUDA execution remains unvalidated; no model
code or Phase 4 work is included. The original v0.4.0 tag must not be moved.

## 0.4.0 — Phase 3, local checkpoint (2026-09-16)

- Original dense decoder, RMSNorm, rotary attention, SwiGLU, and tied output projection.
- Next-token loss, independent numerical references, finite gradients, and causality checks.
- Bounded configurations, verified Safetensors snapshots, and deterministic basic sampling.
- CPU/MPS validation, including a fix for nonfinite values from a combined MPS transfer/cast.

Phase 3 publication is pending approval. Weights are untrained; no training engine exists yet.

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
