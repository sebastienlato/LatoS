# LatoS byte-level BPE tokenizer

Phase 2 trains an original vocabulary and merge table from the prepared LatoS
training paragraphs using Hugging Face Tokenizers 0.23.2. It does not download an
existing tokenizer, call a hosted service, or train a language model. Tokenizers
supplies the BPE algorithm; the data integration, codec contract,
validation, reports, and CLI are original LatoS code.

## Reproduce

Install the locked environment and prepare the corpus as described in
[DATA.md](DATA.md), then run from the repository root:

```sh
RAYON_NUM_THREADS=1 uv run --locked latos tokenizer train --manifest data/manifests/english-books-v1.json --corpus-dir data/processed/english-books-v1 --config configs/tokenizer/english-bpe-v1.json --output-dir artifacts/tokenizers/english-bpe-v1
uv run --locked latos tokenizer inspect --artifact-dir artifacts/tokenizers/english-bpe-v1
uv run --locked latos tokenizer evaluate --manifest data/manifests/english-books-v1.json --corpus-dir data/processed/english-books-v1 --artifact-dir artifacts/tokenizers/english-bpe-v1 --split validation
uv run --locked latos tokenizer encode --artifact-dir artifacts/tokenizers/english-bpe-v1 --text 'Hello, LatoS!' --bos --eos
```

Use a fresh output directory when repeating a run. The command never overwrites
an existing artifact. For an offline demonstration, use the tiny fixture manifest,
its prepared corpus, `configs/tokenizer/debug.json`, and a separate artifact
directory. CI trains only that fixture, without fetching books or a tokenizer.

The English configuration targets 8,192 entries, requires pair frequency at least
two, and limits learned tokens to 32 byte-alphabet symbols. The debug target is
512 entries. The target includes four reserved entries and all 256 byte symbols;
actual vocabulary size can be smaller if a corpus does not supply enough merges.
The config validator caps the target at 32,768. These choices establish a small
baseline; they were not selected by repeatedly tuning against held-out results.

## Input isolation

Fitting has no split option: it always opens `train.jsonl`. It reads the source
manifest and corpus report, verifies the training-file hash and every record's
source, document/group assignment, text hash, identity, and aggregate counts.
It passes only the `text` field to the BPE trainer. Neither source metadata nor
JSON keys become training text. There is no concatenation across paragraphs.

The fitting code never opens validation or test text. A test removes both files
and guards file access, then verifies identical trained artifacts. Another test
injects a held-out assignment into training and updates the outer file hash;
the record-level split check still rejects it. Source manifests/reports are
reviewed inputs, not authenticated signatures against an adversary rewriting
every file. Text is bounded to 64 MB per prepared split, 50,000 records, and
8,000 characters per record.

`evaluate` opens only its explicitly requested split, verifies it, and measures a
frozen artifact. Phase 2 measured training and validation compression and exact
round trips. The real test split was not evaluated or used for fitting. Keep it
reserved for later final evaluation. Evaluation refuses a different corpus-report
identity from the one recorded at fitting time.

## Text and control-token contract

The tokenizer normalizer is **identity**. It preserves case, accents, Unicode
composition, tabs, newlines, leading/trailing spaces, and other valid Unicode
scalar characters. Phase 1 had already normalized its corpus to NFC and collapsed
paragraph whitespace; the tokenizer does not undo that earlier transformation.
Lone UTF-16 surrogate code points are rejected because they are not valid scalar
strings encodable as strict UTF-8.

ByteLevel pre-tokenization uses its regex with `add_prefix_space=False`. A complete
256-symbol byte alphabet is seeded before training, and the ByteLevel decoder
reverses that representation. No automatic lowercasing, padding, truncation, or
post-processing is enabled.

| Reserved spelling | ID | Meaning |
| --- | ---: | --- |
| `<\|pad\|>` | 0 | Reserved padding ID; the codec itself adds no padding |
| `<\|bos\|>` | 1 | Explicit beginning marker |
| `<\|eos\|>` | 2 | Explicit end marker |
| `<\|unk\|>` | 3 | Reserved fallback; ordinary valid text must never emit it |

`LatoTokenizer.encode(text)` adds no boundary tokens. `add_bos=True` and
`add_eos=True` explicitly prepend/append IDs 1 and 2. Empty text encodes as an
empty list, or `[1, 2]` with both flags. Ordinary input containing a reserved
spelling is encoded as literal text, without emitting a control ID.

`decode(ids)` skips explicitly inserted reserved IDs. Use
`skip_special_tokens=False` to display their spellings. Literal spellings in
ordinary text survive decoding because their encoding uses ordinary byte/BPE
IDs. Invalid, negative, Boolean, or out-of-range IDs are rejected.

Always use `LatoTokenizer.load(artifact_directory)` for this contract. The upstream
`encode_special_tokens=True` setting is **not serialized** in `tokenizer.json`;
LatoS reapplies it on every load. Loading the bare JSON directly with the upstream
API does not restore LatoS's literal-special-token behavior. This difference is
covered by a regression test and means raw format compatibility alone is not a
claim of compatibility with another framework.

The exact-round-trip guarantee applies to complete sequences returned by encoding
valid strings. Individual byte tokens may represent only part of a UTF-8 character;
decoding arbitrary fragments can produce a replacement character. Streaming
generation will need a tested incremental decoder in its later phase. Reserved
tokens define a format; they do not teach a model how to chat or follow instructions.

## Artifact integrity and evidence

The artifact directory contains `tokenizer.json` and `metadata.json`. Metadata
records the contract, config, actual vocabulary/merge counts, their SHA-256 hashes,
the tokenizer-file hash, software versions, pipeline implementation hash, exact
training/corpus identities, and training compression/round-trip results.

Loading verifies the artifact hash, configured pipeline components, fixed reserved
IDs, complete byte alphabet, contiguous vocabulary IDs, merge-count relationship,
and vocabulary/merge hashes. An optional `expected_sha256` lets a caller pin the
tokenizer identity from a separately recorded checkpoint. Metadata is not a
cryptographic signature. Future model checkpoints must retain that identity.

Training validates all training-text round trips, writes a temporary directory,
reloads the artifact, and compares every training encoding before publishing the
local directory. Metadata omits timestamps and absolute paths. The fixed local
environment produced identical tokenizer and metadata files on repeated runs;
cross-version or cross-platform bitwise reproduction is not promised. Rebuilding
Phase 1 data under a newer LatoS package version changes its report identity even
when split text is identical; tokenizer metadata then records that new report
hash. Rebuilding the data under 0.3.0 was checked and produced the same learned
tokenizer JSON as the preserved Phase 1 input, with different provenance metadata.

Compression reports count content tokens only, excluding BOS/EOS. Bytes per token
means UTF-8 text bytes divided by token count; characters per token uses Python
Unicode string length. Neither is a language-model quality score. The English
vocabulary reflects a narrow historical corpus; broad Unicode coverage does not
establish multilingual efficiency or model competence.

The learned files remain under ignored `artifacts/`. Only code, configurations,
tests, and compact evidence are proposed for Git publication. See
[Phase 2 evidence](../experiments/phase-2/REPORT.md) for measured results and exact
artifact identities. No new paid services or pretrained assets were used.
