# Project state

Updated: 2026-09-16.

## Active phase

Phase 1 — English data pipeline complete locally; publication approval pending.

Implemented checksum-pinned acquisition, English text cleaning, whole-book and
whole-author split validation, exact/near duplicate removal, JSONL export, an
integrity audit, a tiny original fixture, provenance, and reproducible reports.
Package version: 0.2.0. No new Python dependency was added.

## Evidence and limits

- 12 source books acquired through the implemented CLI: 7,718,091 raw bytes.
- Final corpus: 9,171 training, 1,030 validation, 2,394 test paragraphs; 8/2/2 books.
- One exact and one near duplicate removed; audit found none remaining under the
  declared lexical metric. The fixture verifies cross-split held-out priority.
- Two independent downloads/preparations with different hash seeds produced
  identical files. Preparation plus audit: 3.46 seconds, 514,048,000 bytes maximum
  resident set size on the Mac host; these are pipeline, not training measurements.
- 49 tests passed in development and clean built-wheel environments. Lint, format,
  dependency compatibility, package contents, and CLI fixture checks passed.
- Separate review completed; checksum-before-parse, precise credit filtering,
  and alphabetic-word filtering fixes validated by regression tests.
- Historical corpus has narrow coverage and dated/bias-prone content. The script
  filter is not language identification; duplicate checks are lexical, not semantic.
  No tokenizer, model, training, or language-quality result exists yet.
- Phase 1 Linux CI awaits publication. Details and hashes:
  [Phase 1 report](experiments/phase-1/REPORT.md), [data guide](docs/DATA.md).

## Publication history

Phase 0 was explicitly approved and published on 2026-09-16. Remote `main` and
annotated tag `v0.1.0` were verified at
`a33d84d0200a3e46879bc506a3f8c9a82db2ff7c` in private `sebastienlato/LatoS`.
[Linux CPU CI](https://github.com/sebastienlato/LatoS/actions/runs/35145730007)
passed all Phase 0 checks, including 22 tests.

## Pending publication

- Local branch: `main`; checkpoint message: `Complete Phase 1 English data pipeline`.
  The checkpoint is the commit containing this Phase 1 state; its full ID is in
  the approval request and available from `git log -1 --format=%H` at this checkpoint.
- Existing origin: `https://github.com/sebastienlato/LatoS.git` (private).
- Proposal: push this reviewed commit to `main`, create/push annotated tag
  `v0.2.0` at the same commit. No repository/access changes, release, or data upload.
- Approval: **not yet granted for Phase 1**. Phase 0 approval does not cover it.
- Raw and prepared corpora remain ignored at `data/raw/english-books-v1/` and
  `data/processed/english-books-v1/`; retain them. A verified second copy/run is
  available in sibling directories. The tracked report records artifact hashes.

## Next action

Await “Push Phase 1 to GitHub?” approval. After yes, publish the exact checkpoint,
verify remote branch/tag identities and inspect CI, then begin Phase 2: train an
original byte-level BPE tokenizer on `train.jsonl` only. Keep validation and test
reserved. Phase 2 publication requires its own approval.

Local uv: `.private/tools/bin/uv`; existing `.venv` uses the project-local runtime
under `.private/python`. See setup documentation for fresh environments.
