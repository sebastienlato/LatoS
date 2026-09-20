# Training and recovery

Phase 4 implements a single-process, single-device float32 training engine. Its
acceptance exercise memorizes an original offline fixture. This is not pretraining
of the English pilot and does not establish useful language quality.
The subsequent [Phase 5 pilot](PRETRAINING.md) uses this unchanged engine on the
English corpus and records its own held-out quality/resource measurements.

## Reproduce the acceptance exercise

From a locked development installation:

```sh
uv run --locked python experiments/phase-4/validate.py --output-dir outputs/phase-4-demo
```

Choose a fresh directory. The script acquires only the local fixture, prepares its
fixed splits, trains a new tiny BPE tokenizer on training text, and creates a
127,808-parameter model from seed 17. It uses the committed
`configs/training/fixture.json`: 400 optimizer updates on one CPU thread, with no
held-out checkpoint selection. The acceptance threshold is final training loss
below 0.25 and below 10% of the initial loss. It saves at update 97 and verifies
that recovery reproduces updates 98–400, all weights, optimizer moments, and the
shuffle stream exactly on the recorded local CPU runtime.

All generated material stays inside the ignored output directory. `acceptance.json`
is the compact result, `metrics.jsonl` contains timing and training losses,
`model-config.json` binds the model to the generated tokenizer, and `interrupted/`
and `final/` are full checkpoints. The script deletes its generated test split
before training to verify that the engine has no test-file dependency. This does
not delete or alter any existing corpus.

## Train through the CLI

After the exercise above has produced its tiny tokenizer and model configuration,
a separate run can use those inputs:

```sh
uv run --locked latos train --manifest data/fixtures/tiny/manifest.json --corpus-dir outputs/phase-4-demo/corpus --tokenizer-dir outputs/phase-4-demo/tokenizer --model-config outputs/phase-4-demo/model-config.json --config configs/training/fixture.json --output-dir outputs/phase-4-cli --device cpu --stop-after 100
uv run --locked latos train --manifest data/fixtures/tiny/manifest.json --corpus-dir outputs/phase-4-demo/corpus --tokenizer-dir outputs/phase-4-demo/tokenizer --resume outputs/phase-4-cli/step-00000100 --config configs/training/fixture.json --output-dir outputs/phase-4-cli-resumed --device cpu
```

`--stop-after` ends at an absolute optimizer-update number without changing the
original 400-update schedule. Resume requires that same training configuration,
including the original total steps. Use a fresh output directory for every segment.
Existing output directories and checkpoints are refused, never overwritten.
`--threads` defaults to 1; `--validate-every` defaults to 10 and
`--checkpoint-every` to 25. Final validation and checkpointing always occur.

The CLI writes an initial recovery checkpoint and a `run.json` before updates;
`summary.json` exists only after a successful segment. Each metrics line records
loss, learning rate, pre-clipping gradient norm, actual target-token and window
exposure, shuffle epoch, elapsed update time, and targets per second. Timings include
forward/backward, optimizer work, and finite-weight checks, but exclude validation
and checkpoint I/O. On failure, recover from the last complete `step-*` directory;
updates since that checkpoint are replayed. Metrics from earlier segments remain
in their original directories. There is no automatic best-checkpoint selection.

## Data and objective contract

The engine verifies prepared split checksums through the existing corpus reader
and binds the tokenizer hash and vocabulary to the model configuration. It reads
only `train` and `validation`; the test split is reserved. Data is materialized in
memory under the existing corpus reader's bounds; there are no worker processes,
streaming corpus shards, or background downloads.

Each prepared record (a retained paragraph) receives BOS and EOS and is isolated
from other records, even within the same source document. Long records are cut
into windows of at most `sequence_length` tokens with one token of overlap. Every
next-token transition, including EOS, occurs exactly once per epoch; context is
reset at a window seam. There is no cross-record attention or EOS-to-BOS training
transition. This deliberately sacrifices long-document context for a simple,
verifiable Phase 4 baseline.

Rows are right-padded with ID 0. Labels remain unshifted for the model's internal
next-token shift; padding labels and the initial label are -100. No padding
attention mask is necessary: valid positions precede padding and causal attention
cannot attend to later padding. Batch loss and validation loss are means over
unmasked target tokens, not means over padded positions or records.

A dedicated CPU generator shuffles all windows without replacement each epoch.
The final microbatch can be short. Accumulation can cross an epoch boundary; each
microbatch's mean loss is multiplied by its target count divided by the whole
update's target count before backward. Thus a short batch or short record is not
overweighted. Reported exposure counts repetitions, and the dataset identity
records the distinct transition count per epoch; this is not unique vocabulary.

## Optimization and validation

AdamW uses the configured betas, epsilon, and decoupled weight decay. Matrix
parameters, including the one shared embedding/output parameter, receive decay;
RMSNorm vectors do not. Foreach and fused optimizers are disabled for the baseline.
Gradients accumulate, are globally clipped once, and are cleared after the update.
Nonfinite loss, gradients, or resulting weights fail the update and invalidate the
in-memory trainer; reload a checkpoint instead of continuing from partial state.

The schedule is indexed by one-based optimizer updates, never microbatches. With
W warmup steps, update s <= W uses peak_lr * s / W. After warmup, cosine decay
reaches peak_lr * min_lr_ratio at the final update. Without warmup the first update
uses peak_lr; a one-update run also uses peak_lr. The saved config and completed
update count completely determine the next learning rate; there is no separate
mutable scheduler object to get out of sync.

Validation runs every target once in inference mode, calculates token-weighted
cross entropy and its exponential, and restores every module's prior train/eval
mode even on error. It does not change weights, optimizer, existing gradients,
shuffle cursor, or RNG state. Training-split evaluation is also exposed for the
explicit overfit check; it is labeled `train` rather than held-out validation.

## Checkpoint format and resume guarantee

A checkpoint directory contains:

- `model/`: the existing verified model-only Safetensors format, usable by sampling.
- `training.safetensors`: AdamW moments and step scalars, current shuffle permutation,
  and the CPU shuffle generator's state. No pickle or executable payload is loaded.
- `state.json`: versioned training configuration, data/tokenizer identities, counters,
  optimizer groups, tensor hashes/sizes, source commit/dirty status when available,
  and runtime/implementation fingerprint.

The current model has no dropout or other stochastic training layers. Its only
training RNG is the dedicated shuffle generator; global Python, NumPy, CPU, and
device RNGs are not consumed by updates and are not restored from checkpoints.
Initialization uses the existing isolated seeded model initializer. Adding
stochastic layers requires a new checkpoint contract and tests.

Checkpoints are allowed only between complete optimizer updates, with no pending
gradients. Saving writes a sibling staging directory, verifies a full reload, then
renames it into place. Failed saves remove their staging output and preserve
previous checkpoints. Use a single writer per output directory. This protects
against incomplete process writes; it is not an fsync/power-loss durability claim.
Hashes detect accidental corruption, not an authenticated publisher identity.

Loading checks shapes, dtypes, finite tensors, nonnegative second moments, optimizer
steps/settings, sampler permutation, token/window counters, and dataset identities.
It refuses changed source implementation, package/runtime versions, platform,
backend, thread settings, or recorded numerical settings. CPU acceptance is
**bitwise equivalence on the same tested host/runtime/settings**, including a
separate-process CLI test. Timing metrics are excluded. No cross-host, cross-version,
or CPU/accelerator bitwise guarantee is made. A fingerprint match alone does not
prove hardware equivalence.

Local MPS update/validation/recovery has a short smoke test with comparison tolerance
atol=1e-6, rtol=1e-5. The owner reports independent Windows/RTX 4070 SUPER
Phase 4 training and separate-process recovery PASS at exact v0.5.0. Exact CUDA
equality was observed in that run, not established as a general determinism
guarantee. GitHub Linux CPU CI completed and its logs were verified separately;
independent physical Linux remains deferred, not performed. See
[closure evidence](../experiments/phase-4/CLOSURE.md) for attribution and scope.
Mixed precision, distributed training, arbitrary model
modifications, mid-microbatch recovery, and changing schedules on resume are outside
this checkpoint contract.

See [Phase 4 results](../experiments/phase-4/REPORT.md) and
[method provenance](../THIRD_PARTY.md#phase-4-optimization-and-reproducibility).
