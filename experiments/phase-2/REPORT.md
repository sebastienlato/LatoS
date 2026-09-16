# Phase 2 tokenizer evidence

Recorded 2026-09-16 on the Apple M4 Max macOS arm64 execution host. LatoS 0.3.0,
Python 3.14.7, Tokenizers 0.23.2; CPU training with `RAYON_NUM_THREADS=1`.
The source checkpoint is the Phase 2 commit containing this report.

## Learned artifact

Trained a new 8,192-entry byte-level BPE vocabulary from the eight training books:
9,171 paragraphs, 5,286,727 Unicode characters, 5,336,384 UTF-8 bytes. The trainer
learned 7,932 merges in addition to 256 byte symbols and four reserved entries.
Requested vocabulary 8,192, minimum frequency 2, maximum token length 32.
The config is [english-bpe-v1.json](../../configs/tokenizer/english-bpe-v1.json).

No validation/test text was opened during fitting. The frozen tokenizer was then
measured once on validation. The real test split remains unused in Phase 2.

| Split | Paragraphs | Content tokens | UTF-8 bytes | Bytes/token | Round-trip failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 9,171 | 1,267,870 | 5,336,384 | 4.2089 | 0 |
| validation | 1,030 | 110,493 | 438,128 | 3.9652 | 0 |

No unknown or other reserved IDs were emitted from ordinary text. No BOS/EOS
markers are included in these compression counts. These are codec measurements,
not language-model quality results. Historical English coverage remains narrow.

## Validation and review

85 tests passed, including the existing 49 checks and new train-only access guards,
corrupt-input/artifact rejection, fixed vocabulary and reserved IDs, literal
reserved spellings, optional boundaries, empty text, whitespace, composed and
decomposed accents, multilingual strings, emoji, controls, long input, invalid IDs,
and 100 seeded random Unicode cases. The full training corpus round-tripped
exactly, and every training encoding agreed after save/load.

Two separate CPU training runs produced byte-identical tokenizer and metadata
files, including a repeat with Python hash seed 77. The measured final training
command, including input checks, training, round-trip metrics, save/load checks,
and publication to a local directory, took **4.16 seconds** wall time. macOS
`/usr/bin/time -l` reported maximum resident set size **107,446,272 bytes**
(about 102 MiB). This is one local measurement, not a cross-platform guarantee.

Separate review checked the complete byte alphabet, native serialization behavior,
control-token injection, source/split verification, codec settings, and failure
paths. The upstream special-token runtime flag does not persist in JSON; LatoS
explicitly restores it and tests the difference. The loader also verifies that
merge counts account for the declared learned vocabulary. Lint and formatting pass.

The built wheel also passed all 85 tests in a fresh locked environment outside
the source directory. It loaded the accepted 8,192-entry artifact by its expected
hash and verified literal-marker/Unicode round trips. Dependency compatibility,
source/wheel member inspection, and the offline tokenizer CLI demonstration passed.

Rebuilding the corpus under package 0.3.0 and training again produced identical
tokenizer JSON. The corpus report records the newer package version, so its hash
and the resulting tokenizer provenance metadata differ; the training-text hash
and learned vocabulary identity do not change. Preserve exact reports when
comparing complete metadata files.

Phase 1 publication was verified at `fe9818b983aac00d5bba8e2a7b7686637f9e4a7a`;
[its Linux CI](https://github.com/sebastienlato/LatoS/actions/runs/35147998448)
passed all 49 tests and the offline data fixture. Phase 2 Linux execution awaits
its publication; its CI definition adds offline tokenizer fitting and evaluation.

## Identities and local storage

| Item | SHA-256 |
| --- | --- |
| Tokenizer JSON (236,074 bytes) | `7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab` |
| Metadata JSON | `eff5bcae670f509a4fc7a72e4e225e61eba5616c5853ffbbf958e30657ab736b` |
| Vocabulary mapping | `979fdb5b76065e459a2040bb0ce5474ac448597dbe9dbd6fba0ff97486f53658` |
| Ordered merges | `8ec9bbca712b51b05a541a89e7b4fac9531d6e3fbed9ba55e1bdc79d6912b64e` |
| Tokenizer implementation | `600cfe333ccd9506cb13db1b44d88ede192a24797952d2abca83f690302b81cd` |
| Training JSONL | `0e0cec6846d133971af2f4919a1a057cec114ed7df42f0ee03f46019df808261` |
| Corpus report | `f6f698a9f1bacf924cbf122a2a5a9657504d5805deaa37899053b9f7c75994ff` |

Accepted learned files: ignored `artifacts/tokenizers/english-bpe-v1/`.
A repeat is preserved at `artifacts/tokenizers/english-bpe-v1-repeat/`.
The learned vocabulary/merges are not proposed for Git publication. Compact
[tokenizer metadata](tokenizer-metadata.json) and [validation metrics](validation.json)
are tracked with the source and configurations. The metadata records exact input
manifest/source identities and software versions.

## Limits and next action

Identity normalization preserves valid Unicode scalar strings; it cannot represent
lone surrogates. Individual byte fragments may decode to replacement characters.
Consumers must use the LatoS loader to restore literal-special-token behavior.
There is no streaming decoder, chat template, model, or model-training engine yet.
No pretrained artifact or paid service was used. Hashes establish recorded
integrity, not an authenticated signature or cross-version reproducibility.

After publication is approved and verified, Phase 3 will implement and numerically
validate the original dense transformer, using this tokenizer identity and actual
vocabulary size. Phase 3 publication requires its own approval.
