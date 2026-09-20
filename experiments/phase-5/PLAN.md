# Phase 5 pilot protocol

Recorded before the main run on 2026-09-20. Closure commit
`52b72c9394e022d4cebc85808343bcaee73857c8` was verified on origin/main; annotated
v0.5.0 still resolves to `cb585c321c92f5d774fb59234f76c1d3783a635a`.
The owner explicitly authorized Phase 5 in a fresh session.

## Resource decision

Use the existing Mac M4 Max, 64 GiB unified memory, MPS float32, one CPU thread,
with no new services. CUDA is unavailable on this host. A ten-update resource
probe (`pilot-benchmark.json`, eight windows per update, context 256) processed
10,268 targets in 1.878 synchronized update seconds, including startup:
5,467 targets/s. Maximum sampled MPS driver allocation was 2,251,014,144 bytes;
process lifetime peak RSS was 732,119,040 bytes. These figures support the original
17,308,032-parameter pilot; no smaller model is necessary. The probe's model is
discarded; its measurements remain in `outputs/phase-5-benchmark/`.

## Frozen main-run decisions

- Seed 17, original `configs/model/pilot.json`, trainable float32 parameters.
  Recreate random weights and require exact equality with the untouched Phase 3
  snapshot before moving them to MPS. Do not initialize from the probe or fixture.
- Existing 8,192-entry English BPE and eight-book training corpus. Read only train
  and validation through the verified corpus reader. Never open test text.
- `configs/training/english-pilot.json`: 3,000 updates, batch 8, accumulation 2,
  sequence length 256 (model capacity remains 512), AdamW peak LR 0.0006,
  100-update warmup, cosine to 10%, decay 0.1, clipping 1.0.
- Budget: at most 12,240,000 target exposures; actual exposure is lower because
  paragraph windows vary in length. Expect several passes over 1.28M training
  transitions. The probe suggests roughly 10–20 minutes including overhead, with
  no need for a larger model or a paid run in this phase.
- Full validation before training, every 500 updates, and at completion. Use the
  fixed last checkpoint at update 3,000, irrespective of intermediate loss.
  Save optimizer checkpoints at 0, 1,000, 2,000, and 3,000; retain all of them.
- Compare random-initialization and final cross entropy/perplexity on the same
  complete validation windows, plus an add-one unigram baseline fitted only to
  training targets. A first-128-training-window diagnostic is explicitly a subset.
- Four fixed prompts and sampling settings live in `run.py`: 64 new tokens,
  temperature 0.8, top-k 40, seed 71, before and after training. Keep all samples.
- Record synchronized update time separately from loop time including evaluation
  and saving, and total execution time before artifact inventory. Accelerator
  memory is sampled at boundaries, not claimed as a continuous peak measurement.
- Record failed execution in `failure.json`; preserve completed checkpoints and
  logs. Recovery uses the existing exact-runtime training engine contract.
- Success means a measured, validated pilot and honest comparisons, not a required
  loss threshold or useful assistant behavior. No hyperparameter sweep, test-set
  selection, instruction tuning, or larger-model run is part of this protocol.

Source provenance records the base commit, dirty state, runtime implementation,
lock hash, and exact runner hash. The main runner is copied into ignored outputs.
This experiment adds orchestration and evidence; the Phase 4 engine is unchanged.
