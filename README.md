# LatoS

LatoS is an original, English-first small language model project. Its goal is to
make the path from documented training data to an evaluated conversational model
reproducible on modest hardware.

**Current milestone: Phase 12 published; Windows evidence-path correction reviewed locally,
awaiting publication approval.** External Windows validation stopped at an inventory separator defect
after 299 passing tests and 13 expected skips; Phase 12 CUDA execution was not reached.
The [correction](experiments/phase-12/PORTABILITY_CORRECTION.md) standardizes saved
paths and adds a nested writer/verifier regression. Publication approval and a fresh
Windows/CUDA retest are required before closure or another experiment.

The unchanged [experiment](experiments/phase-12/REPORT.md) compares 256 and 512 tokens.
All 44 previously blocked sessions gained retries, but there was no successful learned
tool use. Removing the capacity blocker is not learned capability improvement.
Runtime/defaults/version remain 1.3.0. See [current state](PROJECT_STATE.md).

Phase 11 closure is published at `e2d61100a7c0c74d759bdc4ded5ddcda89cb112b`.
Its [scoped evidence](experiments/phase-11/CLOSURE.md) preserves original full Mac
negative tool results, distinct owner-reported tiny Windows/CUDA compact-prompt
results, separately inspected hosted Linux CPU evidence and deferred physical Linux.
The Phase 12 experiment uses the original Mac prompt; it does not reproduce the
Windows exercise. Scripted controls establish mechanics only. Learned tool use
remains unestablished; all prior limits and reserved tests remain unchanged.
[Preference learning](docs/PREFERENCES.md) retains the full Mac experiment's
[negative results](experiments/phase-10/REPORT.md): ranking 16/32 → 15/32, exact replies
0/32, and English loss improved versus SFT but still worse than the original base.
Windows tiny ranking stayed 16/32 → 16/32; it did not reproduce that full experiment.
No preference-learning or instruction-following improvement is established.
Phase 8 is published as v1.0.0. [LoRA training and merge](docs/ADAPTATION.md) add
a fixed-budget full-tuning comparison; [results and limits](experiments/phase-9/REPORT.md)
are retained without promoting either adaptation as a useful assistant. LatoS
prepares English data, trains its own tokenizer, and implements an original dense
causal decoder with optimization, validation and resumable checkpoints. A bounded
assistant-only experiment starts from the preserved 17,308,032-parameter Phase 5
base using original synthetic conversations and a shared chat format.
[Phase 6 results](experiments/phase-6/REPORT.md) show lower assistant validation loss
but **0/32 held-out exact answers and worse English validation loss**. Useful chat
or instruction following is not established. The base remains preserved.
A terminal chat interface now streams local replies with optional KV caching,
explicit context limits and cancellation. Select the model artifact explicitly;
see [local inference](docs/INFERENCE.md), [Phase 7 evidence](experiments/phase-7/REPORT.md),
[instruction tuning](docs/INSTRUCTION_TUNING.md) and the [roadmap](ROADMAP.md).

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
Apple Silicon, Linux x86-64 CPU, and Windows x86-64 with the pinned CUDA 13.0 build.
Phase 7 implementation is published as v0.8.0 at
`04031e5ea98da8db495242165a78c216ab1d4cf4`. Mac CPU/MPS validation passed 210 tests
in development and a fresh wheel. Separate inference checks used the full preserved
Mac base/SFT models.
Historical Phase 7 evidence: the owner reports independent Windows/RTX 4070 SUPER
PASS: 202 passed, eight expected skips, fresh wheel suite and actual CUDA inference/CLI. Windows used tiny 256-position
base/SFT artifacts plus a separate synthetic 512-position model; the full Mac learned
artifacts were unavailable and were not validated there. Windows console Ctrl-C
remains unvalidated; callback cancellation passed. GitHub-hosted Linux CPU CI was
separately verified: 203 passed, seven MPS-only skips, tiny inference/CLI/POSIX SIGINT
checks and existing offline workflows. It did not run a fresh-wheel test suite or
validate the full Mac learned artifacts. Physical Linux remains deferred.

See [Phase 7 closure and exact limits](experiments/phase-7/CLOSURE.md). Useful
instruction following, general CUDA determinism, cross-device equality, production
latency, mixed precision and distributed serving remain unestablished. The
[Phase 6 negative SFT result](experiments/phase-6/CLOSURE.md) is unchanged.
Phase 7 closure publication was verified at
`e4fd79ba7ab1fe5bcc19e733c2f6639a2214ba94`; v0.8.0 remains unchanged.
The [release guide](docs/RELEASE.md) provides the tested fresh-checkout path,
[data card](docs/DATA_CARD.md), [model card](docs/MODEL_CARD.md), and
[consolidated experiment report](experiments/phase-8/REPORT.md).
Published v1.0.0 release assets include source, wheel, a clearly separate tiny fixture
model/tokenizer and checksums. Full Mac learned artifacts and acquired corpora stay
local. Phase 10 implementation is published; its separate closure publication
requires explicit approval. See [current state](PROJECT_STATE.md).
See [setup and Windows retest instructions](docs/SETUP.md) and
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

## Transformer checks

```sh
uv run --locked latos model inspect --config configs/model/pilot.json
uv run --locked latos model check --config configs/model/debug.json --device cpu
```

On an available Apple GPU, the model check also accepts `--device mps`.
These commands use random initialization and synthetic IDs; they perform no
optimizer steps. [The model guide](docs/MODEL.md) covers the architecture,
next-token loss, tokenizer binding, local snapshots, and sampling commands.
[Phase 3 evidence](experiments/phase-3/REPORT.md) records numerical and backend
checks. Weights stay in ignored `checkpoints/`.

## Training engine

A fully offline acceptance run creates its own fixture tokenizer and tiny model:

```sh
uv run --locked python experiments/phase-4/validate.py --output-dir outputs/phase-4-demo
```

Choose a fresh destination. The [training guide](docs/TRAINING.md) explains CLI
training/resume, document boundaries, loss weighting, checkpoint contents, and the
same-runtime CPU equivalence guarantee. [Phase 4 results](experiments/phase-4/REPORT.md)
record overfit and recovery evidence. Weights and detailed logs stay ignored.

## English pilot

The [pretraining guide](docs/PRETRAINING.md) explains the fixed run protocol,
reproduction, baselines, local artifact inventory, and recovery. The test split
remains reserved. Full datasets, logs, learned tokenizers, and weights stay outside
Git; reproducing the pilot requires the documented prepared inputs.

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
in [THIRD_PARTY.md](THIRD_PARTY.md). The release cards distinguish fixture artifact terms from acquired-data provenance
and the retained full models. Pilot measurements apply only to the
recorded run, data, and hardware.
