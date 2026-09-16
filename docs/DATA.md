# English data pipeline

Phase 1 provides a bounded, reproducible corpus preparation path. It does not
train a tokenizer or model. The small historical collection is intended for
pipeline development and a later pilot, not general assistant training.

## Sources and rights

`data/manifests/english-books-v1.json` pins twelve English text files, totaling
7,718,091 raw bytes. Each source records its title, author, document and author
group identities, fixed split, catalog URL, edition update date, retrieval date,
original credit line where present, terms, byte count, SHA-256, and a reviewed
start-of-body marker. The hash identifies the actual edition bytes; an upstream
date alone does not. The raw files retain their original notices and are ignored
by Git. They were obtained from an HTTPS mirror listed by
[Project Gutenberg](https://www.gutenberg.org/MIRRORS.ALL), with serial requests
and a two-second delay, rather than crawling the catalog.

| Split | Whole books |
| --- | --- |
| Train | Pride and Prejudice; Frankenstein; The Adventures of Sherlock Holmes; Moby Dick; The Adventures of Tom Sawyer; Dracula; Treasure Island; The Origin of Species |
| Validation | Alice's Adventures in Wonderland; The War of the Worlds |
| Test | A Tale of Two Cities; The Wonderful Wizard of Oz |

The catalogs declare these editions English and public domain in the USA. Their
[license](https://www.gutenberg.org/policy/license.html) and
[permission guidance](https://www.gutenberg.org/policy/permission.html) distinguish
underlying texts from trademarks and impose conditions on branded redistribution.
The selected original authors died before 1947. For the Canadian execution host,
the selection also takes account of the
[Canadian copyright term guidance](https://ised-isde.canada.ca/site/canadian-intellectual-property-office/en/guide-copyright),
which says the 2022 extension did not revive already expired copyright. This is
the selection rationale, not a claim that every edition or future use is cleared
worldwide. New edition material requires its own review.

Only narrative/body text is selected; front matter and prefaces before the
reviewed marker, catalog summaries, images, and the external license wrapper are
excluded from model input. Some in-book footnotes and original captions may
remain. Original downloads, credit notices, and terms stay intact in the local
raw cache. No acquired text or processed corpus is proposed for Git publication;
the repository contains manifests, hashes, code, and aggregate reports. Model and
dataset redistribution terms must be assessed separately in their release phase.
Source metadata and terms were checked on 2026-09-16.

The independent tiny fixture under `data/fixtures/tiny/` contains original
AI-assisted English examples under MIT, with intentional exact and near copies
across splits. It is for offline tests only and is not mixed into the book corpus.

## Reproduce

From the repository root, after the locked setup:

```sh
uv run --locked latos data acquire --manifest data/manifests/english-books-v1.json --raw-dir data/raw/english-books-v1
uv run --locked latos data prepare --manifest data/manifests/english-books-v1.json --raw-dir data/raw/english-books-v1 --output-dir data/processed/english-books-v1
uv run --locked latos data audit --manifest data/manifests/english-books-v1.json --output-dir data/processed/english-books-v1
```

This acquisition command covers every source in the manifest. The first command
downloads only missing files and verifies all cache hits. A changed upstream file
or corrupt cache fails without replacing existing bytes. It never adopts a new
hash automatically. A future source revision needs an explicit manifest update
and fresh evidence. HTTPS certificate checks remain enabled and redirects cannot
downgrade to HTTP. Each source is limited to 4 MB and the manifest to 20 MB.

Preparation and audit have no network dependency. Preparation refuses an existing
output directory; use a new destination for comparison runs. Files are written to
a temporary sibling directory, checked, and renamed into place only after success.
Raw inputs are preserved. A failed run may leave successfully acquired raw files,
which the next acquisition verifies and reuses.

For an offline demonstration, replace the manifest with
`data/fixtures/tiny/manifest.json` and use separate raw/output directories.
CI uses this fixture without downloading books. Network acquisition itself was
also exercised locally against the real pinned sources.

## Transformations and separation

1. Validate provenance, safe identifiers, size/hash declarations, and fixed source
   assignments. A document ID or author group may not occur in more than one split.
2. Verify raw bytes before decoding strict UTF-8. For Gutenberg files, require one
   ordered marker pair and an English header. Apply the source's unique body-start
   marker to exclude front matter. Plain fixture files use their entire contents.
3. Extract paragraphs within the already assigned document. Normalize CRLF/CR
   line endings and Unicode NFC, then collapse intra-paragraph whitespace.
4. Keep paragraphs of 120–8,000 characters with at least 20 alphabetic words and
   at least 90% ASCII among their letters. Reject control/replacement characters
   and specific production/illustration notices. Rejection counts are recorded.
5. Remove exact casefolded text duplicates and lexical near duplicates throughout
   the corpus, within and across splits. Keep test before validation before train;
   source ID and paragraph position break ties deterministically.
6. Write source-ordered JSONL files for the three splits and a duplicate ledger.
   Audit all resulting files before marking the output complete.

Whole-book and author separation happens before paragraphs or future token chunks
exist. The split assignments are explicit and fixed, not a paragraph-level random
shuffle. Validation is for later development; test books remain reserved from
tokenizer fitting, training, and routine model selection. Inspecting aggregate
integrity statistics is allowed. Tokenizer fitting must read only `train.jsonl`.

Near-duplicate comparison uses sets of consecutive five-word sequences, with
casefolded alphabetic words and internal apostrophes. It removes a paragraph if
its Jaccard intersection/union with an already retained paragraph is at least
0.80. Every sequence enters an inverted index; size bounds only discard pairs
that cannot meet the threshold. There is no sketch sampling or approximate
candidate search. A separate brute-force test checks index results, including
the threshold boundary. The standard technique is described in
[Manning, Raghavan, and Schütze, *Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/html/htmledition/near-duplicates-and-shingling-1.html).
LatoS implements its own bounded index and retention policy.

The guarantee is pairwise absence under this lexical metric among retained
paragraphs. It does not cover semantic paraphrases, translations, a short passage
embedded in a much longer one, or all possible paragraph-boundary changes. It is
not a transitive clustering rule. No universal contamination-free claim is made.

## Artifacts and audit

Each JSONL record has `id`, `source_id`, `document_id`, `group_id`, `split`,
`paragraph`, `text_sha256`, and `text`. Paragraph positions refer to the extracted
body, including rejected paragraphs. IDs include source, position, and text hash.

`report.json` contains the manifest hash, pipeline implementation hash, package,
Python and Unicode versions, policy, per-source input/filter counts, duplicate
counts, and split document/record/character/whitespace-word counts and file hashes.
Words are not tokenizer tokens. `duplicates.jsonl` records removed/retained IDs,
splits, and match type without copying their text. Reports omit changing timestamps
and local absolute paths, so repeated runs with the same environment and source
bytes can be compared byte for byte. The report hash is recorded separately in
the experiment evidence.

`audit` checks file hashes before JSON parsing, record identities, split membership,
cleaning rules, reported output statistics, and exact/near duplicates. It verifies
the duplicate ledger hash. Hashes establish integrity relative to the recorded
report, not an authenticated signature; they do not defend against someone
rewriting all artifacts and their reports together. The audit does not re-fetch
raw files. A changed pipeline implementation requires a rebuild under a new output
directory or checking out the recorded implementation.

## Limits and next phase

This corpus is mostly older British/American literary prose, with one historical
science book. It contains period spelling, dated claims, bias, and potentially
offensive language. It is neither modern factual reference material nor an
instruction/safety dataset. Short dialogue is disproportionately removed by the
minimum length rule. Source language declarations plus a script filter do not
constitute language identification; Latin-script non-English quotations can remain.

The algorithm keeps an in-memory index and is capped at 50,000 candidate records.
It is a small-corpus implementation, not a web-scale deduplication system. Actual
measurements and hashes are in [the Phase 1 report](../experiments/phase-1/REPORT.md).
No tokenizer, tokens, training run, or language-quality outcome exists in Phase 1.
