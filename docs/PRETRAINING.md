# Measured English pilot

Phase 5 adds an experiment runner around the unchanged Phase 4 training engine.
It measures a single float32 pilot from random initialization and preserves its
baselines, configuration, fixed samples, resource measurements, and checkpoints.
See the [predeclared protocol](../experiments/phase-5/PLAN.md) and
[results](../experiments/phase-5/REPORT.md). This is base language modeling, not
instruction tuning or a chat assistant.

## Inputs and execution

Use a locked installation from the repository root. Prepare the documented
[English corpus](DATA.md) and [English tokenizer](TOKENIZER.md) if the retained
local artifacts are unavailable. Their identities must match the committed model
configuration; no weights, acquired text, or learned tokenizer files ship in Git.

On an available MPS host, reproduce the budgeted run in a fresh directory:

```sh
uv run --locked python experiments/phase-5/run.py --config configs/training/english-pilot.json --output-dir outputs/phase-5-reproduction --device mps --validate-every 500 --checkpoint-every 1000
```

The seed-17 initializer creates the starting model independently. If the original
Phase 3 random snapshot is available, add
`--initial-snapshot checkpoints/phase-3-pilot-initial` to require tensor-for-tensor
initial agreement. This snapshot is never used as a trained starting checkpoint.

Measure the host first with `--config configs/training/pilot-benchmark.json
--benchmark` and a different fresh output directory. A benchmark performs only
10 optimization updates; it has no quality claim and is discarded as a model.
The runner also accepts explicit `--device cpu` or `--device cuda`; execution of
the full Phase 5 pilot on those backends is not validated by the MPS result.
Do not infer access to external CUDA hardware from earlier owner-supplied tests.

## Evidence and interpretation

`plan.json` is written before updates and includes runtime/source identities,
configuration, seed, sampling settings, cadence, final-checkpoint selection,
window identities, and the target-exposure upper bound. `runner.py` preserves the
exact experiment script. `baseline.json` records full validation and fixed samples
before optimization. Its add-one unigram baseline counts training targets only,
including EOS, with one pseudocount for every vocabulary entry. It is evaluated on
the exact same validation targets as the model.

`metrics.jsonl` records every update and periodic full validation. Exposure counts
non-padding next-token targets, including repetitions; it is not vocabulary size.
The shuffle visits windows without replacement, so after one epoch all training
positions have been seen. `distinct_target_positions_seen` counts corpus positions,
not distinct token values or distinct n-grams. Context is reset between paragraphs
and window seams. The model supports 512 tokens, but this pilot trains/evaluates
with 256-token windows; it establishes no 512-token quality claim.

`summary.json` records the final metrics, samples, measured runtime, and memory.
The selected checkpoint is always the final configured update, not the lowest
validation loss. The train diagnostic uses only the first 128 training windows;
it is not a corpus-wide train score. Test text is never read or used for selection.
Validation comprises only two held-out books, so its improvement does not establish
broad English knowledge or instruction-following ability.

Update timing synchronizes the accelerator on both sides. Loop time includes
periodic evaluation, checkpoint saving/read-back, logging, and memory measurements;
the timed run span also includes preparation, baseline evaluation, and final
samples. It starts after imports/argument parsing and output-folder creation and
ends before summary serialization and hashing the artifact inventory. Process peak
RSS is a lifetime OS measurement. MPS allocated/driver figures are maximum observed boundary snapshots,
not continuous peaks; do not add them to RSS as if they were separate physical
memory pools on a unified-memory Mac.

`artifacts.json` hashes all completed output files except itself. Preserve it with
the output directory, and retain the recorded weights and optimizer files outside
Git. `failure.json` records an interrupted/failed run when the process can handle
the exception; hard termination cannot guarantee this file. Existing output
folders are refused. Do not modify the only retained checkpoint or regenerate
results into the same directory.

## Recovery and portability

Each `step-*` folder is a full recovery checkpoint, with model-only weights in its
`model/` subfolder. Recovery uses the existing `latos train --resume` interface and
same config/data/runtime requirements in [TRAINING.md](TRAINING.md). Use a new
output folder for a resumed segment and retain both segments. The pilot runner
itself starts a fresh run; it does not merge evidence from resumed segments.

The current run's fixed samples can be generated from the selected `model/` with
the existing model sampling API. Cross-host numerical equality is not promised.
The owner reports Phase 5 Windows/RTX 4070 SUPER PASS for a bounded four-update,
416-target CUDA exercise using the unchanged runner, generation, recovery and
measurement mechanisms. This did not reproduce the full Mac pilot or its scores.
Hosted Linux CPU CI passed tiny runner tests and CPU fixture acceptance separately;
it did not run the full pilot. Physical Linux remains deferred. These results do
not establish general CUDA determinism, cross-device numerical equivalence,
sustained-pilot performance, language quality or instruction following. The earlier
observed CUDA fixture equality was run-specific. See the
[closure evidence](../experiments/phase-5/CLOSURE.md) for exact scope and attribution.

For the retained English input locations, independently verify a completed run:

```sh
uv run --locked python experiments/phase-5/verify.py --run-dir outputs/phase-5-reproduction --output outputs/phase-5-reproduction-verification.json
```

This rehashes artifacts, accounts for every update, verifies seeded initial weights,
recomputes initial/final validation, and replays one update from the penultimate
checkpoint in a new process. The report states the actual scalar loss difference
and tolerance; this check does not establish general bitwise recovery or equality
of all later weights. The original Phase 4 recovery tests remain the engine's
more extensive numerical evidence.
