# LatoS

LatoS is an original, English-first small language model project. Its goal is to
make the path from documented training data to an evaluated conversational model
reproducible on modest hardware.

**Current capability: Phase 2 tokenizer.** LatoS prepares a documented English
corpus and trains a new byte-level BPE tokenizer on the training split only.
The local baseline has 8,192 vocabulary entries and verified Unicode round trips.
There is no language model, model-training engine, or chat capability yet.
The [roadmap](ROADMAP.md) defines those steps.

## Quickstart

With [uv 0.12.15](https://docs.astral.sh/uv/getting-started/installation/) available,
run these commands from this checkout:

```sh
uv sync --locked
uv run --locked latos --help
uv run --locked latos doctor --device cpu
uv run --locked latos doctor --json
uv run --locked pytest
```

The environment uses Python 3.14.7 and PyTorch 2.14.0. The lock targets macOS
Apple Silicon and Linux x86-64 CPU. Phase 0 checks passed locally on the Mac and
in Linux CPU CI; Phase 1 data CI also passed. Phase 2 tokenization is locally
validated, with its Linux CI run pending publication.
CUDA and other platforms are not validated. See [setup](docs/SETUP.md) and
[recorded environment evidence](docs/ENVIRONMENT.md).

`doctor` reports detected backends and executes a tiny float32 matrix product and
gradient check on the selected device. `auto` prefers CUDA, then MPS, then CPU.
An unavailable explicit device or failed check returns a nonzero exit code.
This establishes basic execution, not model correctness or training performance.

## English data

Run a fully offline fixture with separate raw and prepared directories:

```sh
uv run --locked latos data acquire --manifest data/fixtures/tiny/manifest.json --raw-dir data/raw/tiny
uv run --locked latos data prepare --manifest data/fixtures/tiny/manifest.json --raw-dir data/raw/tiny --output-dir data/processed/tiny
uv run --locked latos data audit --manifest data/fixtures/tiny/manifest.json --output-dir data/processed/tiny
```

Preparation preserves existing outputs; choose a new output directory for another
run. [The data guide](docs/DATA.md) covers the pinned twelve-book corpus, rights,
whole-book split policy, lexical duplicate metric, hashes, and limits. Acquired
books and processed text stay outside Git. The tiny fixture is original test data.

## Tokenizer

After preparing the offline fixture above:

```sh
RAYON_NUM_THREADS=1 uv run --locked latos tokenizer train --manifest data/fixtures/tiny/manifest.json --corpus-dir data/processed/tiny --config configs/tokenizer/debug.json --output-dir artifacts/tokenizers/tiny
uv run --locked latos tokenizer encode --artifact-dir artifacts/tokenizers/tiny --text 'Hello, LatoS!' --bos --eos
```

See [the tokenizer guide](docs/TOKENIZER.md) for the full-corpus command,
Unicode and special-token behavior, artifact verification, and limitations.
Learned tokenizer files stay under ignored `artifacts/`; the
[Phase 2 report](experiments/phase-2/REPORT.md) records their identities and results.

## Development

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv build --no-build-isolation
```

Development advances one phase at a time. Each checkpoint includes implementation,
tests, a separate review, and a local commit; publishing requires the owner's
explicit approval. [PROJECT_STATE.md](PROJECT_STATE.md) records the current handoff.

Original source is MIT licensed. Dependencies keep their own licenses, recorded
in [THIRD_PARTY.md](THIRD_PARTY.md). Future data and model artifacts will have
separate provenance and distribution terms. No model quality or runtime target
has been established yet.
