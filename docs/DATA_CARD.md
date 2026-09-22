# Data card — LatoS v1.0.0 candidate

Recorded 2026-09-21. These datasets serve different purposes and must not be mixed.
The release distributes original fixture inputs and generator source only. Acquired
books, prepared corpora and both real reserved test payloads remain local.

## English books v1

Purpose: bounded historical English language-modeling research. Twelve editions
were acquired on 2026-09-16; the [manifest](../data/manifests/english-books-v1.json)
pins source/catalog URLs, authors, credit lines, edition dates, byte counts,
SHA-256 hashes, body markers, terms and whole-book/author-group split assignments.
Manifest SHA-256: `c83a221eba52532de100a30eb0badec5eddf009d17369b5fac981ff89b2ca87c`.

| Split | Books | Retained paragraphs | Characters |
| --- | ---: | ---: | ---: |
| Training | 8 | 9,171 | 5,286,727 |
| Validation | 2 | 1,030 | 432,293 |
| Reserved test | 2 | 2,394 | 833,886 |

Training books: Pride and Prejudice; Frankenstein; The Adventures of Sherlock
Holmes; Moby Dick; The Adventures of Tom Sawyer; Dracula; Treasure Island; The
Origin of Species. Validation: Alice's Adventures in Wonderland and The War of
the Worlds. Reserved test: A Tale of Two Cities and The Wonderful Wizard of Oz.

The pipeline chooses body text, normalizes Unicode NFC and whitespace, filters
paragraphs by size/script/word count, and removes exact casefolded and five-word
shingle duplicates at Jaccard >= 0.80. Test has priority over validation over train.
Whole-document and author assignments precede chunking. One exact and one near
copy were removed; this lexical criterion does not establish semantic isolation.
See [full policy](DATA.md) and [original evidence](../experiments/phase-1/REPORT.md).
The frozen tokenizer fits training only. Training/evaluation use isolated paragraphs
and 256-token windows with one-token overlap, including EOS transitions.

The source catalogs identify the editions as unrestricted under US copyright;
[Project Gutenberg's license](https://www.gutenberg.org/policy/license.html) and
[permission guidance](https://www.gutenberg.org/policy/permission.html), rechecked
2026-09-21, distinguish text rights, trademarks and jurisdiction. The original
selection rationale for Canada is recorded in DATA.md. This release does not
redistribute acquired text or declare worldwide clearance for learned artifacts.
Original notices remain intact in the local raw cache. MIT on repository code does
not relicense third-party texts. No external dataset or dependency is vendored.

Historical prose is narrow, dated and potentially offensive. It includes bias,
obsolete claims and uneven representation. The script filter is not language
identification, and short dialogue is often filtered out. It is not a factual,
modern-English, safety or instruction-following benchmark.

## English instructions v1

Purpose: a deliberately small assistant-only experiment. The original
[generator](../experiments/phase-6/data.py) and word groups were authored with Codex
assistance on 2026-09-20, under MIT; no external generation API or human demonstration
collection was used. There are 192 train, 32 validation and 32 reserved test
conversations, derived from 48/8/8 distinct word groups. Each group supplies copy,
label extraction, first-letter and two-turn recall tasks. Shared templates are
intentional. This measures new words in familiar tasks, not unseen task types.

The generator checks full-conversation duplicates; readers check group and ID
separation. The [chat contract](INSTRUCTION_TUNING.md) masks every target except
assistant content and its EOS. The full run saw 744 distinct training target
positions, repeated to 6,188 exposures. Recall scoring uses gold previous replies.
The test reservation is never a routine tuning or release acceptance input.
The generated dataset is reproducible from source but is not a release asset.

## Tiny fixture

The [three original fixture documents](../data/fixtures/tiny/README.md) were
AI-assisted project writing on 2026-09-16, under MIT. Intentional train/test exact
and near copies test deduplication; after filtering there are 3/3/4 paragraphs.
This offline fixture supplies the downloadable tiny model and tokenizer. Its test
file is disposable engineering input, distinct from both reserved real test sets.
The acceptance exercise removes its prepared test file before training.

## Reservations and integrity

| Reserved payload | SHA-256 |
| --- | --- |
| English test JSONL | `a01a55c31d0401ab5dba64cf0f14bd96e8ed9900012a7099f972b3377d58a954` |
| Instruction test JSONL | `5e0fe6050dcd1df7e85ce46ebcd830ee6cec8b4841b9c1f8f270b110889395a7` |

Phase 8 hashes these existing payloads as opaque bytes only. It does not parse,
evaluate, tune on or select artifacts using them. They remain reserved. Hashes
provide integrity against recorded identities, not authenticity or a backup.

## Phase 10 — Synthetic preference data

See the [bounded preference experiment](../experiments/phase-10/REPORT.md) and
[preference provenance](../experiments/phase-10/DATA.md). The original base, SFT,
tokenizer and their identities above remain unchanged. DPO is a separate local
experimental model, not a promoted replacement. Both reserved tests remain unused.
