# LatoS

LatoS is an original small language model project built from first principles:
from documented data and an independently trained tokenizer to a dense Transformer,
training, evaluation and local inference. Its model and training implementation
are written directly in PyTorch, with attribution for the established techniques
and libraries it uses.

The current system provides a reproducible engineering foundation. **Useful
instruction following and learned tool use have not been established.**
[Roadmap 2.0](ROADMAP.md) puts evaluation, better data and measured model quality
at the center of the next stage.

## What is implemented

- Reproducible acquisition, cleaning, deduplication and document-level data splits.
- Independently fitted byte-level BPE tokenization with verified artifacts.
- Dense causal language modeling, pretraining, validation and checkpoint recovery.
- Assistant-only supervised fine-tuning, LoRA adaptation/merge and DPO experiments.
- Local terminal chat, streaming, KV caching and explicit context limits.
- A strict, bounded local tool protocol with separate scripted and model evaluations.

These mechanisms have tests and retained experiment reports; their presence does
not establish learned capability. Training data and evaluations so far primarily
use English. The full historical base has **17,308,032 parameters** and was trained
from random initialization. No external pretrained model or tokenizer was imported.

## Architecture

The native decoder uses pre-RMSNorm blocks, rotary positions, causal multi-head
attention, SwiGLU feed-forward layers and tied input/output embeddings. The
historical pilot has eight layers, width 384, six heads, an 8,192-entry vocabulary
and 512-position capacity. Training and language-quality evaluation used windows
up to 256 tokens; capacity alone does not establish quality at longer lengths.
Float32 is the supported baseline. See the [model guide](docs/MODEL.md) for equations,
configuration bounds and [research attribution](THIRD_PARTY.md).

## Install

From a checkout, with uv 0.12.15 available:

```sh
git clone https://github.com/sebastienlato/LatoS.git
cd LatoS
uv sync --locked
uv run --locked latos --help
uv run --locked latos doctor --device cpu
```

The checkout pins Python 3.14.7 and PyTorch 2.14.0. The lock targets Apple Silicon
macOS, Linux x86-64 CPU and Windows x86-64 with CUDA. Installation may download
locked dependencies; the tiny examples below use only repository data. See
[setup](docs/SETUP.md) for prerequisites, PowerShell syntax and scoped platform
evidence. An unavailable explicit device fails rather than silently falling back.

## Try the pipeline

Prepare the original offline fixture, then fit and use its tokenizer:

```sh
uv run --locked latos data acquire --manifest data/fixtures/tiny/manifest.json --raw-dir data/raw/tiny
uv run --locked latos data prepare --manifest data/fixtures/tiny/manifest.json --raw-dir data/raw/tiny --output-dir data/processed/tiny
uv run --locked latos data audit --manifest data/fixtures/tiny/manifest.json --output-dir data/processed/tiny
uv run --locked latos tokenizer train --manifest data/fixtures/tiny/manifest.json --corpus-dir data/processed/tiny --config configs/tokenizer/debug.json --output-dir artifacts/tokenizers/tiny
uv run --locked latos tokenizer encode --artifact-dir artifacts/tokenizers/tiny --text 'Hello, LatoS!' --bos --eos
```

Use fresh output directories; preparation and artifact creation preserve existing
outputs. The [data guide](docs/DATA.md) and [tokenizer guide](docs/TOKENIZER.md)
cover full-corpus usage, provenance, split rules and codec behavior.

For a self-contained training demonstration, the existing acceptance runner builds
its own tiny tokenizer/model, trains on the fixture and verifies checkpoint recovery:

```sh
uv run --locked python experiments/phase-4/validate.py --output-dir outputs/latos-demo
uv run --locked latos chat --model-dir outputs/latos-demo/final/model --tokenizer-dir outputs/latos-demo/tokenizer --device cpu --context-limit 64 --prompt 'Hello.' --max-new-tokens 8 --json
```

The tiny model demonstrates memorization and pipeline mechanics, not assistant
quality. Omit `--prompt` and `--json` for terminal chat. Choose a fresh training
destination. See [training/resume](docs/TRAINING.md), [pretraining](docs/PRETRAINING.md),
[instruction tuning](docs/INSTRUCTION_TUNING.md) and [inference](docs/INFERENCE.md)
for native commands and explicit selection of other model/tokenizer artifacts.

## Stable release

[**v1.0.0**](https://github.com/sebastienlato/LatoS/releases/tag/v1.0.0) is the current
published stable release: source archive, wheel, separate tiny fixture
model/tokenizer and checksums. Use `git checkout v1.0.0` before installation to
reproduce that release; current `main` contains later work and package version 1.3.0.
The [release guide](docs/RELEASE.md) explains installation and reproduction.
Full historical learned models, acquired corpora and raw logs are not bundled
with a clone or the release.

## Measured limitations

The historical base improved held-out language-model loss over random initialization,
but generated text remains limited. SFT and the LoRA/full-tuning comparison each
failed all 32 held-out exact-response tasks. DPO did not establish preference
improvement. Tool experiments produced no successful learned tool use; allowing
more context removed a retry blocker without improving that outcome.

These negative results remain preserved in the [evidence index](docs/INDEX.md),
[model card](docs/MODEL_CARD.md) and [data card](docs/DATA_CARD.md). Factual reliability,
safety alignment, production performance and useful assistant quality are
unestablished. Mac learned-model results, bounded Windows/CUDA checks and hosted
Linux CPU CI are separate evidence; physical Linux validation remains deferred.

## Explore and contribute

[Documentation and experiments](docs/INDEX.md) · [Roadmap 2.0](ROADMAP.md) ·
[Current project state](PROJECT_STATE.md) · [Changelog](CHANGELOG.md) ·
[Engineering decisions](DECISIONS.md)

Development checks:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv build --no-build-isolation
```

[Working instructions](AGENTS.md) describe development and publication safeguards.
Original source uses the [MIT license](LICENSE). Dependencies retain their own
licenses; [THIRD_PARTY.md](THIRD_PARTY.md) records attribution. Dataset and artifact
terms are documented separately in the cards.
