# Phase 1 validation report

Recorded 2026-09-16. Corpus: `english-books-v1`; pipeline: `english-paragraphs-v1`.
Package 0.2.0, Python 3.14.7, Unicode 16.0.0; macOS arm64 on the Phase 0 host.
The source checkpoint is the Phase 1 commit containing this report. Pipeline
implementation SHA-256: `e0bd54e2bd6db9aeeb4bc370ddb177c15ae35e2c00fc5a6601dc050ed35dbb2c`.
Manifest SHA-256: `c83a221eba52532de100a30eb0badec5eddf009d17369b5fac981ff89b2ca87c`.

## Corpus result

Twelve pinned source files, 7,718,091 raw bytes. Complete originals are retained
under ignored `data/raw/english-books-v1/`; a second independently downloaded
copy is under `data/raw/english-books-v1-verified/`. Final prepared artifacts are
under ignored `data/processed/english-books-v1/`. None of these text files are
part of the publication. The full compact metadata report is
[corpus-report.json](corpus-report.json).

| Split | Books | Paragraphs | Characters | Whitespace words |
| --- | ---: | ---: | ---: | ---: |
| train | 8 | 9,171 | 5,286,727 | 948,942 |
| validation | 2 | 1,030 | 432,293 | 78,341 |
| test | 2 | 2,394 | 833,886 | 153,270 |

There were 12,597 candidates after cleaning. One exact and one near duplicate
were removed, both within a split. A fresh audit of all 12,595 retained paragraphs
found zero duplicate pairs under the declared metric and verified all source/split
assignments. The independent tiny fixture deliberately creates cross-split
copies: one exact and one near training copy are removed in favor of test copies.

Preparation plus its automatic audit took **3.46 seconds wall time**, measured
with macOS `/usr/bin/time -l`; reported maximum resident set size was
**514,048,000 bytes** (about 490 MiB). This is one local process measurement,
not training throughput or a cross-platform benchmark. It excludes downloads.

## Reproduction and checks

- The implemented acquisition CLI downloaded all twelve sources into a fresh
  directory and verified the pinned sizes/hashes. Verified cache reuse also passed.
- Two separate preparation processes used Python hash seeds 1 and 987 and the
  independently acquired raw directories. All five output files matched byte for
  byte, including the report and duplicate ledger.
- 49 tests passed, including original environment/CLI tests, offline acquisition,
  corrupted cache and output, document/author leakage, marker/Unicode handling,
  empty splits, duplicate priority, input-order independence, and a brute-force
  oracle for indexed Jaccard results and its boundary.
- The built wheel passed the same 49 tests in a fresh locked environment outside
  the source directory; its installed pipeline hash matched and its corpus audit
  passed. Source archive and wheel contents excluded private and acquired data.
- Lint and formatting checks passed. No new third-party Python dependency was
  needed. Phase 1's Linux CI run awaits publication; its fixture workflow is
  configured to avoid downloading the books.

The separate review corrected checksum validation order before JSON parsing,
restricted an overbroad production-credit filter, and made the minimum-word check
count alphabetic words so numeric tables cannot bypass it. Regression checks pass.
The test fixture's intentional near copy changes a final word, giving a known
above-threshold pair rather than assuming any one-word edit always meets 0.80.

## Artifact hashes

| File | SHA-256 |
| --- | --- |
| duplicates.jsonl | `7e20dfef5a65da962a064fd2f6ea4720699a1619410b1bdda47ac42c65071606` |
| report.json | `f6f698a9f1bacf924cbf122a2a5a9657504d5805deaa37899053b9f7c75994ff` |
| test.jsonl | `a01a55c31d0401ab5dba64cf0f14bd96e8ed9900012a7099f972b3377d58a954` |
| train.jsonl | `0e0cec6846d133971af2f4919a1a057cec114ed7df42f0ee03f46019df808261` |
| validation.jsonl | `09647b26e4cee7ced1808ed29c72c8d98753e032b215e520a2b3dbd5a718af40` |

## Limits and next action

These are historical literary/scientific texts with period bias and language,
limited topical breadth, and no modern factual-quality guarantee. The script
filter is not language identification. Lexical near-duplicate checks do not cover
semantic paraphrases or every resegmentation. Paragraph length filtering removes
short dialogue; it is a documented baseline, not an optimized quality choice.
Raw and processed files remain local, with provenance and terms in
[the data guide](../../docs/DATA.md). No tokenizer or training was run.

After the Phase 1 publication is approved and verified, Phase 2 will train a new
byte-level BPE tokenizer on the training split only, preserving validation and
test separation. No Phase 2 publication approval is implied.

## Publication follow-up — 2026-09-16

Phase 1 was approved and published at `fe9818b983aac00d5bba8e2a7b7686637f9e4a7a`,
with remote main and annotated tag v0.2.0 verified. Linux CI passed all 49 tests
and the offline data fixture. The publication status above is the original
pre-publication record. Current development state is in `PROJECT_STATE.md`.
