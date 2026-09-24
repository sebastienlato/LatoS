# Project state

Updated: 2026-09-24.

## Current authorized scope

**A separate bounded recovery-mechanics diagnostic is authorized to support a
prospective Phase 17.1 — Base Model 2.0 Corrective Attempt.** Local implementation
and CPU engineering checks prepare its Windows/CUDA handoff. **No actual CUDA
diagnostic comparison has run yet, and no reproducible correction is established.**
Full Phase 17.1 training, Phase 18 and all remote publication are unauthorized.

Published Phase 17 closure is permanently preserved at
`09c1935b085218fbbc9ea88703d8ba12e63c516e`: interrupted/failed, replay gate failed,
recovered model unaccepted, fixed development/final evaluation absent and exit gate
unmet. Its result is not reopened or repaired retrospectively. Verified publication
supersedes the earlier tracked pending-publication wording. No new tag/release/assets.

## Frozen diagnostic

[Plan](experiments/recovery-diagnostic/PLAN.md),
[machine-readable bounds](experiments/recovery-diagnostic/plan.json) and
[Windows guide](experiments/recovery-diagnostic/WINDOWS.md) govern this distinct study.
The owner's ceilings remain 60 GPU minutes, two pairs, 4,096 physical diagnostic
updates, 6 GiB new artifacts, zero paid resources and no held-out/final access.

The reviewed design uses **2,964 selected-scale physical updates maximum**: each
of two profiles runs a fresh 997-update reference, saves at 512, then replays the
same checkpoint for 485 updates in a new process. The prefix is shared, not trained
twice. All engineering preflight tests forbid optimizer updates. The conservative
3,600-second supervisor limit includes preparation/preflight/hashing and checkpoint
I/O as well as GPU work. A 1 GiB internal reserve covers administrative/return copies
inside the 6 GiB artifact ceiling. No retries, extra profiles or budget reset.

Use the exact selected 34,087,424-parameter native model, CUDA BF16/float32 AdamW,
batch two × accumulation eight, ending with two microbatches. Inputs are fresh
synthetic token IDs at fixed seeds, shaped by training-only window-length metadata;
no historical model or actual training/held-out text is loaded. The short diagnostic
schedule is explicitly separate from the old full run and future Phase 17.1.

Compare legacy numerical settings to prospectively deterministic algorithms,
cuDNN determinism and a fixed cuBLAS workspace. Import original LatoS unchanged
(implementation hash `76cf4a198ee6acde566fdcc70c9420c0cae8ab6cc0d6b4d7e4c9c0cba5c8f09c`).
Require exact model/optimizer/sampler/RNG state at checkpoint roundtrip/restore.
A demonstrated prospective correction requires a complete legacy continuation
failure and a corrected continuation with exact per-update states and no loss
excess above the unchanged `1e-5`. Otherwise stop as inconclusive/failed.

## Evidence available now

[Historical scalar inspection](experiments/recovery-diagnostic/historical-trace.json)
locates the first visible loss/gradient-norm differences at original update 7,124;
the first frozen-tolerance excess remains 7,125. This does not establish a cause.
The original first I/O failure's PowerShell involvement remains unproven.

[Data checks](experiments/recovery-diagnostic/data-validation.json) reproduce the
15,940 synthetic windows / 5,021,949 targets byte-for-byte and match the intended
variable-length shuffle. No model or optimizer was created by that data construction.
[Local validation](experiments/recovery-diagnostic/validation.json) and
[separate review](experiments/recovery-diagnostic/REVIEW.md) record engineering scope.
There is no local connection to the RTX machine; actual diagnostic evidence must
come from the bounded Windows handoff. No new CI or physical Linux result is claimed.

## Next action and stop boundaries

Prepare the exact reviewed local diagnostic commit and checksummed allowlisted
transfer. Windows Work uses the existing pinned runtime, runs only the new CPU
preflight and one bounded study, preserves every outcome and returns evidence.
Old Windows source, datasets, ledgers and tensors remain untouched. The compact
return is not a backup of diagnostic tensor payloads retained on Windows.

After independent Mac verification/review, if the diagnostic establishes a
reproducible prospective correction, **stop and prepare a concise proposal for a
fresh Phase 17.1 Mac Work chat**. That future attempt must start from random
initialization, use accepted Phase 15 data and Phase 16 model/training configuration,
incorporate only justified prospective engineering corrections, and perform fixed
acceptance evaluation. It needs separate explicit owner authorization. Never promote
or initialize it from the failed recovered model. If diagnostic evidence is
inconclusive/failed within the budget, stop and report it; do not start Phase 17.1.

Phase 18 remains blocked until a valid future base passes the required process.
This approval authorizes no push, tag, release, model asset or other remote write.
