# Phase 15 — Data 2.0 results

Data construction is complete locally. The final corpus, provenance/rights review,
audits, tokenizers and local Phase 16 input package are prepared. This establishes
an audited data foundation, not successful training or useful assistant capability.
Publication remains pending explicit approval; no remote write, tag or release.

## Actual retained inputs

| Split | Pretraining documents / rows | Pretraining UTF-8 bytes | Instruction conversations |
| --- | ---: | ---: | ---: |
| train | 80,478 / 332,782 | 158,795,397 | 14,259 |
| development | 1,730 / 7,144 | 3,362,351 | 292 |
| reserved | 1,724 / 6,795 | 3,229,568 | 243 |

Training pretraining text is **29.76 times** the original 5,336,384 UTF-8 bytes.
There are **74.27 times**
as many training conversations as the original 192. No repeated epoch counts
toward these sizes. The independent construction floors (100 MB total pretraining
text and 10,000 training conversations) pass without lowered filters/gates.

Pretraining is entirely English encyclopedia prose from two pinned Wikipedia
shards. It broadens the old literary domain but is not a balanced web/textbook/code
mixture. Task categories and exact source counts are in [corpus.json](corpus.json).
Training instructions: **12,156 Dolly + 2,103 OASST**, including **734 multi-turn**
conversations. All eight observed Dolly categories remain: brainstorming 1,568;
classification 1,626; closed QA 1,299; creative writing 624; general QA 2,005;
extraction 1,070; open QA 3,178; summarization 786. OASST contributes actual
ranked conversation paths, one per tree, not combinatorial shared-prefix inflation.

## Provenance, selection and rights

Author-hosted inputs: Wikimedia Wikipedia (20231101 English), Databricks Dolly
15k, and OpenAssistant OASST1 ready trees. The [manifest](../../configs/data2/sources-v1.json)
pins revisions, bytes, hashes, notices, source/lineage and intended use. See
[rights and transformations](../../docs/DATA_2.md) for primary sources and terms.
Wikipedia uses controlling CC BY-SA 4.0 terms; its stale 3.0 dataset-card wording
is preserved and explained. Dolly and its identified Wikipedia contexts use
CC BY-SA 3.0. OASST contributions use Apache-2.0, with reviewed nonsynthetic
metadata and quality labels. No generator API or pretrained LatoS model/tokenizer
was used. Lingua statistical language detection is an optional preprocessing dependency.

No acquired text, learned tokenizer or weights enter Git. Redistribution of
adapted CC text needs attribution/change notices/share-alike; Apache notices also
remain required. Dolly lacks individual context-article URLs upstream: resolve
them before derived-data release. Known suspect source material is excluded;
automated filtering and source licenses cannot certify every undisclosed quotation
or contribution. Future tokenizer/weight distribution requires its own review.

Raw inputs comprise 312,578 Wiki pages in the acquired shards, 15,011 Dolly rows
and 10,364 OASST trees. Hash selection retains 178,854 Wiki candidates totaling
599,999,907 raw text bytes; other pages are outside the sample, not unique retained
data. 7,781 OASST trees fail metadata/path admission. 196,448 selected candidates
enter text screening. [Rejection counts](corpus.json) and
[per-source dispositions](source-dispositions.json) record actual outcomes.
Oversize/budget exclusions are subsets of Wiki not-selected counts, so do not add
these overlapping acquisition/selection denominators together.

## Filtering, separation and contamination

NFC/line-ending cleanup, footer removal, language identification, structural/role
checks, repetition/prose bounds, contact/secret and unsuitable-content patterns,
quotation/rights screening and missing-template checks are versioned separately
from history. Eight explicit sample-review quarantines remove complete records,
without rewriting answers. [Quality review](quality-review.json) records 44 unique
before and 44 unique final sampled records, fixed excerpts and replacement IDs.
Residual typos, confident simplifications, dated facts, public names/biographies,
unverified factual claims and omitted short leads/lists remain limitations.

Full-set lexical joins remove **260 documents**, **3,128 paragraphs**, then
**three emitted pretraining chunks** overlapping Dolly conversations. Exact and
Jaccard-0.8 five-shingle checks cover all retained candidates, not a sampled
estimate. Shared title/prompt/context families and duplicate components are grouped
before chunking. Prompt three-shingle Jaccard >=0.6 and normalized stems address
template similarity; semantic task families are not guaranteed disjoint.

The final [audit](contamination.json) checks **361,515 records**,
**98,726 documents** and **91,148 groups**: zero observed
remaining full-record exact/near pairs within or across splits, zero split-group
overlap and zero protected matches. Pre-admission exclusions include 429 source-
family matches, 63 protected exact matches and one 13-word match; entire affected
documents/conversations were rejected. IDs/reasons remain in local ledgers.

The guard fingerprints all ARC splits (including train), old LM development/test,
old instructions, both frozen instruction suites and retained Phase 14 outputs.
Controlled reserved access is contamination-only: no example export, scoring,
final-access override or template design. Short common answers are not banned
substrings everywhere. Unknown-origin/semantic overlap, translations and hash
collisions remain possible. This is not universal contamination certification.

## Tokenizer evidence and decision

Both original candidates fit the same **31,999,933-byte / 59,701-record**,
training-only sample. The historical tokenizer remains untouched. New development
analysis chooses **16,384 entries**: **9.67% fewer content tokens** than new 8,192,
and **25.39% fewer** than historical 8,192, with no source compression
regression versus new 8,192. The declared 8% selection margin passes.

| Codec | Wiki development bytes/token | Dolly bytes/token | OASST bytes/token |
| --- | ---: | ---: | ---: |
| historical8192 | 3.1570 | 3.2297 | 3.3761 |
| new8192 | 3.8357 | 3.7439 | 3.7961 |
| new16384 | 4.2466 | 4.1456 | 4.1935 |

**Training accounting with the selected codec:**

- Pretraining: **37,226,334 content tokens**, or **37,559,116** next-token
  targets per isolated-record pass including EOS.
- Instructions: **2,831,301 content tokens**, **2,995,610**
  chat-stream tokens and **1,787,217 assistant targets** including EOS.
- Combined exact-segment accounting: **40,055,334 content tokens** after collapsing
  identical whole text segments/messages, versus 40,057,635 raw content positions.
  This is not a count of unique n-grams or proof of semantic uniqueness.
- Training uses 16,297 of 16,380 ordinary IDs; development uses 15,820. Unseen byte
  IDs remain available. Vocabulary use does not establish learned competence.

All analyzed train/development text round-trips without control IDs from content;
Unicode, whitespace, literal specials, save/load and chat contract 1 checks pass.
The frozen 192-byte BPB spans remain compatible. No reserved split is tokenized
for fitting/selection. Do not compare different-tokenizer perplexities or bind old
weights to the new IDs. [Full comparison](tokenizer-comparison.json),
[training counts](token-accounting.json) and [artifact hashes](artifacts.json).

Instruction bytes in corpus ledgers include one newline between messages for
record identity; tokenizer compression counts message bodies without that joiner
and accounts for actual chat headers/boundaries separately.

## Execution, review and limitations

Full preparation: **1273.29 s**, **8.73 GB** peak RSS.
Final curation: **35.65 s**. Tokenizer fitting/comparison/accounting:
**102.64 s**, **1.26 GB** peak RSS. These are Mac CPU
process measurements; timings exclude imports and some final serialization.
Retained phase roots total **4.30 GB** logical file bytes,
including attempts, repeats and isolated environment/builds, below the 12 GB
artifact budget. [Resources](resources.json) defines scope; RSS is process lifetime,
not GPU memory or additive unified-memory usage.

A 4,000-real-candidate two-pass replay is byte-identical. The entire final curation
was also repeated with byte-identical data, ledgers and report. No second complete
raw-to-final full-corpus run is claimed. Synthetic tests separately cover joins
against exhaustive comparisons, corruption, malformed roles, split isolation,
protected matching, train-only fitting, codec compatibility and cross-source
final duplicate removal. [Validation](validation.json) and [review](REVIEW.md)
record actual final checks and every unsuccessful attempt.

The initial 60k-page pass retained only 57.25 MB and failed the construction floor.
It and two partial scans remain local. Resource-based selection amendments and
quality fixes are documented in the original plan; no acceptance minimum was
lowered. Intermediate curation attempts and a tokenizer entrypoint import failure
are retained. The latter occurred before any candidate fitting and was fixed by
renaming a script that shadowed the Tokenizers dependency.

The local [Phase 16 input guide](../../docs/PHASE16_INPUTS.md) binds actual data and
tokenizer hashes, length/padding constraints, exposure options and hypothetical
time scenarios. CUDA throughput, mixed precision, architecture and memory feasibility
are unmeasured. No new corpus-based pretraining/SFT/DPO/LoRA/tool/RL experiment ran;
the normal test suite includes disposable synthetic optimization fixtures.

The fixed historical results remain **1.872944 BPB**, ARC-Easy **157/570 (27.54%)**,
ARC-Challenge **66/299 (22.07%)**, and all five artifacts **0/96 instructions**,
every category **0/16**. Original SFT/adaptation/DPO/tool negatives remain separate.
Dataset construction is not learned-quality improvement or permission to lower gates.

## Publication checkpoint

One reviewed local source/evidence commit is proposed to existing
`https://github.com/sebastienlato/LatoS.git`, branch `main`, without tag/release/assets.
Acquired data, local package and learned tokenizers remain ignored. Exact commit
and explicit approval status belong in the checkpoint message and local record.
Stop for **“Push Phase 15 to GitHub?”**; no Phase 16 implementation before that
publication checkpoint is satisfied.
