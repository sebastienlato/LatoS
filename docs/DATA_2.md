# Data 2.0 construction and use

Phase 15 builds separately versioned local inputs for future pretraining and
instruction tuning. It trains tokenizer candidates only. No new learned language
model, CUDA throughput, assistant improvement or passed evaluation gate follows
from accepting a dataset. Historical corpora, tokenizer, learned artifacts and
Phase 14 evaluations remain unchanged. See the [plan](../experiments/phase-15/PLAN.md)
and [results](../experiments/phase-15/REPORT.md).

## Sources, provenance and terms

The [source manifest](../configs/data2/sources-v1.json) pins upstream repository
revisions, individual payload/notice byte counts and SHA-256, review date, intended
use and rights. Local acquisition records add actual timestamp and duration.
Each prepared row retains its original page ID/URL or conversation/row identity,
raw-file identity, source, category and transformations through versioned code.
Notices and unchanged raw downloads remain in `data/raw/data2-v1`.

- **Wikipedia:** Wikimedia's November 2023 English snapshot, shards 4 and 3 out of
  41, selected by a fixed hash order. Retain page URLs for attribution/history.
  The pinned [dataset card](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md)
  still says CC BY-SA 3.0. The controlling [Wikimedia terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)
  and [dump guidance](https://dumps.wikimedia.org/legal.html) specify CC BY-SA 4.0;
  this post-June-2023 text is handled under 4.0, with the conflicting card retained.
  Images are excluded. Long quotations/rights markers are conservatively rejected;
  undocumented exceptions can still remain.
- **Dolly 15k:** Databricks employee instructions, with supplied Wikipedia contexts
  in some categories. [Author documentation](https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/README.md)
  identifies CC BY-SA 3.0 for both contributions and those contexts. Copyright 2023
  Databricks, Inc.; Wikipedia editors and contributors. Context article URLs are
  absent upstream: retain the available attribution and resolve per-article
  attribution before any derived-data release. Do not claim MIT rights over text.
- **OpenAssistant OASST1:** [Author-hosted ready trees](https://huggingface.co/datasets/OpenAssistant/oasst1)
  and [Köpf et al., 2023](https://arxiv.org/abs/2304.07327), under Apache-2.0.
  Keep tree/message IDs and observed quality scores; omit volunteer user IDs from
  prepared data. Accept only reviewed English nonsynthetic messages with no model
  attribution, quality >=0.5 and no positive spam/PII/inappropriate/hate/sexual labels.
  Choose one ranked path per tree, at most six user/assistant pairs, ending at an
  actual assistant reply. [Contributor guidance](https://projects.laion.ai/Open-Assistant/docs/guides/guidelines)
  discourages copied answers; metadata is contributor evidence, not proof of
  authorship or correctness. Suspected copied/defective inspected records are
  quarantined; upstream license does not erase third-party rights.

Author-published licenses permit local use and transformation subject to their
conditions. Redistribution is a separate act: CC adapted text needs attribution,
change notices and applicable share-alike terms; Apache contributions require
license/notices and modification identification. Neither source licenses nor
repository MIT settle every future tokenizer/weight release question. No raw or
derived text, learned tokenizer, source archive of acquired data, or weights are
published in Phase 15. No external generation API or synthetic instruction mixture
is used. The original 192 synthetic conversations remain separate and protected.

## Deterministic preparation

Select up to 180,000 whole Wikipedia pages (see the measured plan amendment) by SHA-256 of stable IDs from the two
pinned shards, excluding >100 KB pages and limiting selected text to 600 MB. This
is bounded shard sampling, not representative coverage of all Wikipedia. Keep
Dolly and admissible OASST trees in their independent source identities. Fail on
changed hashes, malformed upstream schemas or unsafe paths; never adopt new bytes
or execute dataset scripts. Preserve failed output directories.

Normalize Unicode NFC and line endings; preserve whitespace inside instruction
content. Remove encyclopedia reference/footer sections. Reject corrupt/control
characters, insufficient prose, excessive repetition, uncertain English, contact
or secret patterns, selected unsafe lexical patterns, long quotations, missing
obvious template values and unframed personal-life claims. Lingua 2.2.0 in low
accuracy mode compares all supported languages using at most 1,500 characters
from the start/middle/end; require English confidence >=0.6. That score is not a
calibrated correctness probability. Source language and detector decisions are
both evidence, with short/mixed-language false positives/negatives possible.

Every selected rejection receives an ID and reason. The raw shard rows outside
selection are counted separately; acquisition volume never counts as retained
unique data. The deterministic [quality exclusions](../configs/data2/quality-exclusions-v1.json)
record defects found in prescribed samples, not evaluations. Samples use smallest
SHA-256 of `data2-quality-v1:<id>`: ten per source and three per Dolly category.
Before samples describe parsed source candidates; OASST already has metadata
screening. Those construction samples precede assignment, so the new reservations
are not claimed to be wholly unseen by the data curator; they are not the frozen
Phase 14 acceptance tests. After samples describe retained train/development text only. Findings
and inspection scope are in the report; automated filters do not prove accuracy,
privacy, suitability or absence of third-party content.

## Deduplication and separation

Normalize comparison text with NFKC, casefold and Unicode word tokens. Full
normalized text equality uses SHA-256. Word five-shingles use 64-bit BLAKE2b
fingerprints. Full-document and >=40-word paragraph/context pairs use Jaccard >=0.8.
An original exhaustive prefix index compares full sets: for n shingles the prefix
contains n-ceil(0.8n)+1 entries under common sorted-hash order. Any qualifying pair
must share a prefix entry. All candidates undergo full-set verification. There is
no MinHash/LSH sampling; fingerprint collisions remain a small, unmeasured source of lexical decision error. Tests
compare joins to independent all-pairs calculations, including boundary cases.

Union all detected document/paragraph/context links transitively. Also group
normalized article titles with numeric/disambiguating qualifiers removed, exact
instruction prompts, their first five normalized words with quoted/numeric slots
removed, common context, and prompt three-shingle Jaccard >=0.6. This goes beyond
word substitution in identical templates, but is lexical family isolation, not a
guarantee that semantic task types are disjoint. Broad skills occur in all splits;
these splits are data development inputs, not replacements for the fixed tests.

Assign each complete component by hash, nominally 96/2/2 train/development/reserved;
large groups make realized proportions uneven. Remove duplicate documents and
pretraining paragraphs, preserving instruction context intact. Only then emit
pretraining paragraphs with >=40 words, splitting >8,000-character paragraphs
without losing scalars. No conversation is chunked or silently truncated. Keep
source/group/document IDs on every row. Small headings/lists are omitted, which
biases the corpus toward prose and reduces reference/list coverage.

## Evaluation protection

[Controlled audit](EVALUATION.md) permits fingerprint comparison of protected
material, with no model scoring or example export. The guard verifies frozen
identities, reads historical LM validation/test, all old instructions, both fixed
Phase 14 suites, all historical Phase 14 generated outputs and
[all pinned ARC splits](../configs/data2/protected-arc-v1.json). ARC train/test files
are acquired into a separate audit-only cache, never added to training. Final-access
overrides and acceptance scoring are not used.

Reject source-family markers and the four held-out books/authors; compare whole
segments for exact equality, any 13-word match and five-shingle Jaccard >=0.8.
Screen full documents and prose fragments before admission. Generic short answers
are checked as whole messages, not forbidden substrings everywhere. Short wiki
headings are omitted from fragment checks; they are not emitted as training rows.
A final removal-only curation applies the recorded post-filter sample quarantine
and removes near-duplicate emitted chunks, keeping group/split assignments and
all surviving text unchanged. Audit this final output again, including
within/across-split duplicate counts and group membership. Reports expose only affected IDs, reasons, counts and input hashes.
Unknown benchmark origin, semantic paraphrases, translations and near containment
without any 13-word run remain residual risks. Zero observed overlap is a bounded
lexical result, not a universal contamination certificate.

## Tokenizer and Phase 16 interface

Prepared schema 1 uses six JSONL files: `pretrain-{train,development,reserved}` and
`instruction-{train,development,reserved}`, plus document, rejection, duplicate and
integrity ledgers. Pretraining rows contain text; instructions contain exact ordered
role/content pairs. Consumers must verify report/file/row hashes and respect group
and split IDs. Historical corpus readers keep their old bounds; Phase 16 must
integrate these new inputs explicitly, not bypass old identity checks.

New codecs share the existing byte-level and chat contract 1. Fit 8,192 and 16,384
candidates on the same deterministic training-only subset, <=24 MB pretraining
and <=8 MB instruction serialization. New development data selects compression;
reserved text is not tokenized for selection. Literal controls remain text,
Unicode/whitespace round trips are checked, and old weight/token IDs are never
rebound. Cross-tokenizer perplexities remain incomparable; the fixed matched-text
BPB method remains compatible.

The report provides token counts, used vocabulary, sequence distributions,
overflow, isolated-window padding and embedding costs. Instruction budgets retain
whole conversations; overflow means defer/reject that conversation for the chosen
context, not train on a cut answer. Packing would require an explicitly validated
attention/boundary policy in Phase 16. Corpus construction selects no architecture,
mixed precision, CUDA budget or future number of epochs.

## Reproduction

From a checkout with the preserved historical audit inputs available:

```sh
uv sync --locked --extra evaluation --extra data2
uv run --locked --extra data2 python -m latos.data2 acquire --manifest configs/data2/sources-v1.json --raw data/raw/data2-v1
uv run --locked --extra data2 python -m latos.data2 acquire --manifest configs/data2/protected-arc-v1.json --raw data/cache/data2-protected-arc-v1
uv run --locked --extra data2 python experiments/phase-15/run.py --output data/processed/data2-reproduction-base
uv run --locked --extra data2 python experiments/phase-15/finalize.py --corpus data/processed/data2-reproduction-base --output data/processed/data2-reproduction
uv run --locked --extra data2 python -m latos.data2 audit --corpus data/processed/data2-reproduction --output outputs/data2-reproduction-audit.json
RAYON_NUM_THREADS=1 uv run --locked --extra data2 python experiments/phase-15/compare_tokenizers.py --corpus data/processed/data2-reproduction --output artifacts/tokenizers/data2-reproduction
```

A clone alone lacks the original local protected corpora/output panel. Missing
protection is a failure, never a reason to mark an unscreened corpus accepted.
Keep those inputs local under their existing retention rules. Tests use original
disposable fixtures; they do not substitute for the reported real-corpus audit.

The locally assembled Phase 16 directory at `outputs/phase15-input-package` contains
only train/development inputs, the selected verified tokenizer, source notices and
compact evidence. `bundle.json` pins every copied byte. It excludes weights, raw
payloads and reserved/evaluation examples. This is a local input artifact, not a
GitHub release or authorization to redistribute its contents.
