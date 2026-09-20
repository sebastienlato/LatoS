# Project state

Updated: 2026-09-20.

## Phase status

**Phase 5 is complete locally and reviewed; push approval is PENDING.**
The owner started it in a fresh Work session after remote main was verified at
closure `52b72c9394e022d4cebc85808343bcaee73857c8`. Annotated v0.5.0 remains at
`cb585c321c92f5d774fb59234f76c1d3783a635a`. No Phase 5 remote writes have occurred.

## Outcome and validation

- Original 17,308,032-parameter pilot, verified seed-17 random initialization,
  MPS float32: **3,000 updates**, **5,761,229 target exposures**, **4.511389**
  corpus-equivalent passes. Timed run span **417.224 s**; synchronized optimization
  throughput **14,628.5 targets/s**. No training failure or new paid service.
- Full validation: loss **9.089003 → 4.731898**, perplexity **8857.350 → 113.511**;
  train-only unigram baseline loss **6.883256**. Test text remains reserved.
- Final checkpoint is the predeclared update 3,000, despite slightly better
  validation at 2,500. Samples show English surface structure but remain repetitive
  and incoherent; useful chat/instruction-following is not established.
- **164 tests passed** in the locked development environment and separately in a
  fresh non-editable wheel installation. Lint/format, MPS doctor, source/wheel builds,
  archive/privacy inspection, and separate review passed. Setup-invocation failures
  were corrected and recorded; no dependency or core engine changes were needed.
- Independent artifact check verified **25 files**, all update/exposure totals,
  random initial tensors, initial/final validation, and a replayed trained update.
  Same-host scalar equality is not a general determinism guarantee.
- [Report](experiments/phase-5/REPORT.md), [review](experiments/phase-5/REVIEW.md),
  [reproduction guide](docs/PRETRAINING.md), and compact JSON evidence give scope.

Phase 5 execution is MPS-only. Earlier Windows/CUDA PASS is owner-reported Phase 4
validation; hosted Linux CPU CI PASS was verified separately for Phase 4. Physical
Linux remains deferred, not performed. No Phase 5 external-platform/CI pass is claimed.

## Preserved local artifacts

`outputs/phase-5-english-pilot/` retains initial and 1,000/2,000/3,000 optimizer
checkpoints, exact runner, plan, raw metrics, samples, and hashed inventory. Selected
model: `step-00003000/model/`; weight SHA-256
`f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa`.
`outputs/phase-5-benchmark/` and validation logs are retained separately.
Prior corpus, tokenizer, random snapshot, and Phase 4 outputs remain untouched.
Artifacts are ignored local files, not uploaded backups or part of a fresh clone.

## Publication checkpoint and next action

- Exact reviewed commit: the commit containing this state, supplied in the owner
  approval request and available with `git log -1 --format=%H` at this checkpoint.
- Commit message: `Complete Phase 5 measured English pilot`.
- Remote: existing private `https://github.com/sebastienlato/LatoS.git`.
- Destination: `main`; proposed new annotated tag **v0.6.0** at that same commit.
  Keep v0.5.0 fixed. No release, dataset, learned tokenizer, weight, or log upload.
- Approval: **PENDING**. Stop here for the owner's Phase 5 push decision. No Phase 6
  work has begun. After approval, publish only the described reviewed state and
  verify remote branch/tag before any subsequent roadmap work.

Local uv: `.private/tools/bin/uv`; accepted runtime remains in `.venv`.
