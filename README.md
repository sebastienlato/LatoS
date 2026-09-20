# LatoS

LatoS is an original, English-first small language model project. Its goal is to
make the path from documented training data to an evaluated conversational model
reproducible on modest hardware.

**Current capability: Phase 6 measured instruction-tuning experiment.** LatoS
prepares English data, trains its own tokenizer, and implements an original dense
causal decoder with optimization, validation and resumable checkpoints. A bounded
assistant-only experiment starts from the preserved 17,308,032-parameter Phase 5
base using original synthetic conversations and a shared chat format.
[Phase 6 results](experiments/phase-6/REPORT.md) show lower assistant validation loss
but **0/32 held-out exact answers and worse English validation loss**. Useful chat
or instruction following is not established. The base remains preserved.
See [instruction tuning](docs/INSTRUCTION_TUNING.md) and the [roadmap](ROADMAP.md).

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
Phase 6 implementation is published as v0.7.0 at
`f6636af34933b9678824cc8dd50f6ab6559a2de5`. The full Mac MPS experiment ran 200
updates / 6,188 assistant-target exposures; 187 local tests passed in development
and a fresh wheel installation. The owner reports independent Windows/RTX 4070
SUPER PASS: 184 passed, three MPS-only skips, fresh wheel suite, and a bounded
four-update / 175-assistant-target CUDA exercise with 49 artifact checks and 64
response replays. It did not reproduce the full Mac run or learned artifact; both
tiny Windows models scored 0/32 exact replies. GitHub Linux CPU CI passed separately
with 184 passes/three MPS skips, tiny SFT runner/verifier tests, fixture acceptance
and workflow checks. Physical Linux remains deferred. No useful instruction
following, general CUDA determinism, cross-device equality or mixed-precision/
distributed capability is established. See [Phase 6 closure](experiments/phase-6/CLOSURE.md).
Closure publication awaits approval. **Do not begin Phase 7 in this chat.** After
verified closure publication, use the [fresh-chat handoff](docs/PHASE7_HANDOFF.md).
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
in [THIRD_PARTY.md](THIRD_PARTY.md). Future data and model artifacts will have
separate provenance and distribution terms. Pilot measurements apply only to the
recorded run, data, and hardware.
