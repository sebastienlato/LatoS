# LatoS

LatoS is an original, English-first small language model project. Its goal is to
make the path from documented training data to an evaluated conversational model
reproducible on modest hardware.

**Current capability: Phase 0 foundation.** The package installs, provides a CLI,
and checks the local PyTorch environment. There is no tokenizer, model, training
pipeline, or chat capability yet. The [roadmap](ROADMAP.md) defines those steps.

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
Apple Silicon and Linux x86-64 CPU. Only the Mac execution path has been run
locally; the Linux CI workflow awaits publication. CUDA and other platforms are
not validated. See [setup](docs/SETUP.md) and [recorded evidence](docs/ENVIRONMENT.md).

`doctor` reports detected backends and executes a tiny float32 matrix product and
gradient check on the selected device. `auto` prefers CUDA, then MPS, then CPU.
An unavailable explicit device or failed check returns a nonzero exit code.
This establishes basic execution, not model correctness or training performance.

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
