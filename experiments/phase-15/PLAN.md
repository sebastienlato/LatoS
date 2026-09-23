# Phase 15 — Data 2.0 plan

Recorded before payload acquisition, 2026-09-23. Start: published Phase 14
8eaf32d06df43449529ec7d2a1827637261a0f47; HEAD/main/origin/main/remote main,
eleven tag objects/targets and existing release/four assets verified unchanged.
Explicit owner authorization starts this phase. No language-model training.

## Selection and resource budget

Host: Apple M4 Max, 16 CPU cores, 64 GiB unified memory, approximately 627 GiB
available storage. Budget: <=2 GB compressed acquired payload, <=12 GB new disk,
<=24 GiB process RSS, no paid services; stop and report if those bounds prevent
acceptance. Keep all old corpora/artifacts/evaluations intact. Save failures.

Candidates: the author-hosted Wikimedia 20231101 English Wikipedia snapshot for
modern expository breadth; Databricks Dolly 15k for human instructions in seven
reported categories; OpenAssistant OASST1 for human conversation trees and quality
labels. Review each source's primary documentation/terms and pin revisions/hashes
before admission. Avoid benchmark mixtures, untraceable scraped instruction sets,
API-generated corpora with unresolved generator terms, and unrestricted web crawls.
No pretrained language model or tokenizer. Language identification models bundled
with a reviewed detector are preprocessing only, never LatoS model initialization.

Select two English Wikipedia shards by lowest SHA-256 of
`latos-data2-v1:<shard filename>`, then up to 60,000 whole articles by lowest
SHA-256 of `wiki:<page id>` across those shards. This bounds full-document work
while spreading selection across pages. It is not representative random sampling
of all Wikipedia; report shard and topic limitations. Other candidates are small
enough to acquire completely. Upper bound 400 MB selected Wikipedia text, skipping
an individually oversized document (>100 KB) before deterministic selection.

Acceptance minima, fixed before acquisition: >=100 MB retained unique pretraining
text (about 19 times the old 5.29 MB) and >=10,000 retained training instructions
(>52 times the old 192), multiple independently collected instruction sources,
all seven Dolly categories represented in the retained collection, actual multi-turn
conversations, and broad expository subjects documented by reproducible samples.
These are independently chosen construction/resource floors, not learned-quality
gates. No repeat/epoch count counts toward unique data. If filtering prevents
these floors, preserve the run and report a bounded amendment/blocker; never
relax contamination, rights or evaluation quality gates to make counts pass.

## Preparation and audits

Strict decoding/schema validation, stable source/document/conversation identifiers,
recorded rejections, normalized line endings/NFC with content structure retained.
English identification plus source language metadata; reject empty/corrupt text,
excessive repetition, obvious contact/credential patterns, overt abuse/unsafe
requests and benchmark/source markers. Retain source notices. These filters are
fallible and neither prove factual accuracy nor comprehensive safety/privacy.

Deduplicate exact normalized documents and audit lexical near copies using word
five-shingles at Jaccard >=0.80. Record full versus approximate coverage and test
candidate retrieval against exhaustive small fixtures. Related article-title groups,
conversation trees, prompt/template families and discovered duplicate clusters
must stay together. Assign groups deterministically 96/2/2 train/development/reserved
before chunking; new reservations are not model selection inputs. Audit paragraphs
and prompts as well as full documents, including across sources and splits.

Protect all historical held-out sources, original instruction reservations, frozen
Phase 14 prompts/answers and generated outputs, and all ARC splits/source families.
Controlled audit may read protected content only into comparison fingerprints:
no displaying examples, scoring, template construction or final-access override.
Retain only hashes/counts/dispositions in public audit evidence. Check exact,
13-word containment and lexical near overlap with explicit short-answer limits.

Quality inspection: smallest stable hash of record ID, stratified by source and
Dolly task category, before and after filtering; at least 10 examples per source
and 3 per instruction category. Inspect all selected examples, retain local sample
identities and findings, report defects and limitations, not just favorable cases.

## Tokenizer and handoff

Compare historical 8,192 BPE against new 8,192 and 16,384 candidates only. Fit the
same deterministic <=32 MB training-only subset, balanced with up to 8 MB instruction
text. No held-out fitting. Keep codec/chat contract 1 and old artifacts unchanged.
Measure new development split compression/domain distributions, round trips,
Unicode/whitespace/control-token behavior, used vocabulary, instruction lengths,
256/512/1024/2048 context overflow, packing/waste and tied embedding costs. Select
16,384 only if at least 8% fewer development tokens than new 8,192, with no source
compression regression; otherwise select new 8,192 if it improves on historical.
Selection is codec efficiency, not a perplexity/learned-quality claim.

Report actual retained content tokens and serialized assistant targets separately;
construct Phase 16 data-budget/exposure options and hypothetical throughput-based
time ranges. CUDA throughput/memory is unmeasured here. Architecture, mixed precision
and all training belong to later phases.

Run full quality suite/evaluation extra, fresh installed wheel tests, corpus-scale
integrity/audit/reproducibility checks, preservation verification, separate local
review and fixes. Prepare reviewed local commit; no remote write before explicit
phase push approval. No new tag/release proposed.

## Bounded selection amendment after the first complete pass

The 60,000-page selection used 200,144,967 raw text bytes but retained only
57,246,824 pretraining bytes across all splits: below the declared 100 MB floor.
The full pass retained 14,263 instruction-training conversations (736 multi-turn).
Its 493.45-second runtime and 3.90 GB peak RSS support expanding document selection
without more downloads or relaxed filters. Preserve it at `data/processed/data2-v1`
as a non-accepted construction attempt, not the final package.

Increase the page ceiling to 120,000 under the SAME 400 MB selected-text, 2 GB
acquisition, 12 GB disk and 24 GiB process budgets. The two existing shards suffice;
keep identical hash order, sources, filtering, dedup thresholds and acceptance
minima. New destination: `data/processed/data2-v1-expanded`. This is a bounded
resource refinement based on actual retention, not a lowered acceptance gate.

### Final resource refinement and quality-rule correction

The complete first-output audit passed on 134,894 records with zero cross-split
or protected matches. Its prescribed post-filter samples nevertheless exposed
additional empty numeric-template forms (for example absent lengths/weights),
which lexical overlap checks do not detect. Extend the defect rule and quarantine
one inspected abruptly incomplete source extraction. Preserve these findings.
The 120,000-page attempt was stopped incomplete during filtering for this fix.

The measured 28.6% pretraining retention means a 400 MB selection has too little
margin above the 100 MB floor when additional defective documents are removed.
Use one final selection bound of 180,000 pages / 600 MB raw text from the SAME
already acquired shards. Projected preparation RSS is under 12 GB based on the
3.90 GB measured initial peak, below the unchanged 24 GiB ceiling. Acquired bytes,
12 GB overall artifact-disk budget, all sources and acceptance minima are unchanged.
New destination: `data/processed/data2-v1-reviewed`. This is the final bounded
construction run; if it cannot meet the data floor, report the shortfall. No new
source, lower filter threshold or language-model experiment is authorized by this
refinement. Preserve both earlier partial attempts and the complete undersized run.
