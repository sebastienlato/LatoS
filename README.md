# LatoS

LatoS is an original, English-first small language model project. Its goal is to
make the path from documented training data to an evaluated conversational model
reproducible on modest hardware.

**Current capability: Phase 1 English data preparation.** The package installs,
checks the local PyTorch environment, and acquires, cleans, splits, deduplicates,
and audits a small English corpus. There is no tokenizer, model, training engine,
or chat capability yet. The [roadmap](ROADMAP.md) defines those steps.

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
in Linux CPU CI. Phase 1 data checks have run locally; its CI run awaits publication.
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
