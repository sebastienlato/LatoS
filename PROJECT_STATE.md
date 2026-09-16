# Project state

Updated: 2026-09-16.

## Active phase

Phase 2 — tokenizer complete locally; publication approval pending.

Implemented train-only byte-level BPE fitting, an identity-normalization codec,
explicit reserved IDs and BOS/EOS flags, verified save/load, compression metrics,
CLI commands, bounded configurations, tests, and offline CI steps. Package 0.3.0;
Tokenizers 0.23.2 added and locked with compatible transitive dependencies.

## Evidence and limits

- New vocabulary: 8,192 entries = 256 byte symbols + 4 reserved IDs + 7,932 merges.
- Fitted only on the eight training books (9,171 paragraphs). Every training
  paragraph round-tripped exactly and encoded identically after save/load.
- Training: 1,267,870 content tokens, 4.2089 UTF-8 bytes/token. Frozen validation:
  110,493 tokens, 3.9652 bytes/token; zero round-trip failures or unknown IDs.
- Real test text was not opened for fitting or evaluated in this phase.
- Repeated runs produced identical tokenizer/metadata files. Rebuilding the corpus
  under 0.3.0 also reproduced tokenizer JSON; versioned input-report metadata differs.
- 85 tests passed in development and clean wheel environments. Lint, format,
  dependency checks, package-content inspection, and offline CLI checks passed.
- Separate review verified byte coverage, split isolation, and serialization;
  loader restores the nonserialized literal-special-token runtime flag and checks
  the merge/vocabulary count relationship. Regression checks passed.
- Training plus verification: 4.16 seconds, 107,446,272 bytes maximum resident set
  size on local CPU. These are tokenizer, not language-model training measurements.
- Phase 2 Linux CI awaits publication. Identity normalization applies to valid
  Unicode scalar strings; partial byte-token sequences can decode with replacement
  characters. Narrow historical corpus; no model, chat, or model-training engine.
- Details: [tokenizer guide](docs/TOKENIZER.md), [Phase 2 report](experiments/phase-2/REPORT.md).

## Published checkpoints

Private origin: `https://github.com/sebastienlato/LatoS.git`.

- Phase 0: `a33d84d0200a3e46879bc506a3f8c9a82db2ff7c`, main/v0.1.0 verified;
  Linux CPU CI passed 22 tests.
- Phase 1: `fe9818b983aac00d5bba8e2a7b7686637f9e4a7a`, main/v0.2.0 verified after
  explicit approval; [Linux CI](https://github.com/sebastienlato/LatoS/actions/runs/35147998448)
  passed 49 tests and the offline data fixture.

## Pending publication

- Local branch: `main`; message: `Complete Phase 2 tokenizer`.
  Checkpoint is the commit containing this state; full ID is in the approval
  request and available through `git log -1 --format=%H` at this checkpoint.
- Proposal: push that reviewed commit to existing private `sebastienlato/LatoS`
  main, and create/push annotated tag `v0.3.0` at it. No releases or learned-file uploads.
- Approval: **not granted for Phase 2**; previous approvals do not authorize it.
- Accepted learned artifact: ignored `artifacts/tokenizers/english-bpe-v1/`.
  Tokenizer SHA-256: `7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
  Preserve it, its metadata, and the repeat copy; hashes are tracked in the report.
- Retain ignored `data/raw/english-books-v1/` and `data/processed/english-books-v1/`.

## Next action

Await “Push Phase 2 to GitHub?” approval. On yes, publish the exact checkpoint,
verify remote branch/tag and inspect CI, then begin Phase 3: original dense
transformer implementation and numerical tests. Bind future model configuration
and checkpoints to the recorded tokenizer identity. Phase 3 publication needs
its own approval.

Local uv: `.private/tools/bin/uv`; `.venv` uses the runtime under `.private/python`.
