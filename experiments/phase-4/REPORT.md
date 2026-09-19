# Phase 4 — Training engine

Implemented and published as v0.5.0 on 2026-09-19 at
`cb585c321c92f5d774fb59234f76c1d3783a635a`. This report records the original local
implementation checks; [closure evidence](CLOSURE.md) adds owner-reported Windows/CUDA
PASS and directly verified GitHub Linux CPU CI. Closure publication is pending. The owner
started Phase 4 in a fresh Work chat after read-only verification that closure
commit `294d2e0af2712266b398151eb7e335fa7ccdc31e` was on `origin/main`; v0.4.2
still resolved to `1102b64714b58c1f2289f80863fa032cc192c47b`.

## Deliverable

Original single-device float32 training integration: record-isolated batching,
AdamW, warmup/cosine decay, token-weighted gradient accumulation, global norm
clipping, metrics, validation, and resumable tensor-only checkpoints. The CLI
runs new or resumed segments into fresh output directories. Checkpoints preserve
model, optimizer, learning-rate position, shuffle permutation/RNG/cursor, exposure,
data/config identities, and runtime/source provenance.

See [the training contract and commands](../../docs/TRAINING.md). Existing pinned
dependencies are unchanged; only the package's root version advances to 0.5.0.

## Fixed offline acceptance

[acceptance.json](acceptance.json) is the compact output of [validate.py](validate.py)
on macOS 26.6.2 arm64, Python 3.14.7, PyTorch 2.14.0, float32 CPU, one compute
thread. The host reports 16 logical CPUs and 64 GiB physical memory. CUDA is
unavailable. Source was the recorded closure checkout plus local Phase 4 changes;
the package implementation SHA-256 identifies the exact code tested. The published
implementation commit is recorded above.

The experiment fits its own fixture-only tokenizer (328 actual entries), then
initializes a 127,808-parameter, two-layer model with width 64 and context 64.
It trains on all prepared training records: six windows, 329 target transitions
per epoch. Validation has eight windows and 391 transitions. The generated test
file was removed before training; no test-file read is needed by the engine.

The fixed criterion was training loss <0.25 and <10% of initial after exactly
400 optimizer updates. No held-out tuning or best-checkpoint selection was used.

| Evidence | Result |
| --- | --- |
| Initial → final training loss | 5.8146379088 → 0.0100304467; PASS |
| Initial → final validation loss | 5.8069872746 → 11.2458015257 |
| Training target exposure | 131,600 targets, 2,400 windows; repeated fixture exposure |
| Resume | Update 97 → 400; all 303 remaining non-timing metric records identical |
| Final recovery comparison | All model/AdamW tensors and sampler state bitwise equal on CPU |
| Validation isolation | Weights, optimizer, RNG and cleared gradients unchanged; mode restored |
| Update-loop time | 1.683 seconds, approximately 78,204 target tokens/second |
| Total exercise time, including resumed replay | 3.423 seconds |
| Peak process RSS | 334,512,128 bytes, including runtime and comparison copies |

These timings describe only this tiny fixture and host, excluding validation/save
from update-loop throughput. They are not pilot-runtime estimates. The severe
validation regression is retained explicitly: this demonstrates memorization,
not improved general English. No English pretraining or Phase 5 work occurred.

Local artifacts are retained at `outputs/phase-4-acceptance-v2/`, including full
metrics, generated tokenizer, interrupted checkpoint, and final checkpoint.
Model tensor SHA-256:
`21c3b92efd50eeacd60ff046a84874d9b0a1edc032e9c7eca09ea63b91fa5c35`.
Training tensor SHA-256:
`4d63537beb90c76c3a205cffb765247960a0188777e733674a03866730e9a5a6`.
Only this compact report/configuration and validation script are tracked.

## Tests and separate review

- Full local suite: **161 passed**, including 29 training cases and both MPS tests.
- Training cases cover target alignment/padding, every chunk-seam transition,
  token-weighted accumulation against a combined batch, clipping before AdamW,
  decay grouping and tied weights, schedule endpoints, overfit, and exact recovery
  at steps 0/1/3/7/12 across partial batches and epoch boundaries.
- A real CLI integration test launches three separate processes and compares final
  weight, optimizer/sampler, and metadata files byte-for-byte after recovery.
- Validation checks weights, optimizer, existing nonzero gradients, RNG, counters,
  and mixed per-module modes; exception paths restore modes as well.
- Failure checks reject data/config/runtime drift, bad counters, corrupted tensor
  bytes, bad optimizer shapes, split/context mismatch, and output overwrite.
  Failed save read-back removes staging output and preserves the prior checkpoint.
- Local MPS smoke: update, read-only validation, save/load, and next-update comparison
  at atol=1e-6, rtol=1e-5. [MPS diagnostics](mps-environment.json) passed with CPU
  fallback disabled. This short test is not full MPS overfit or a bitwise guarantee.
- Lint/format and locked environment checks passed. Clean wheel installation and
  archive inspection are recorded in [packaging evidence](packaging.json).

A separate local review pass inspected the batching math, scheduler endpoints,
optimizer grouping/state restoration, checkpoint failure handling, CLI boundaries,
and publication scope after implementation. It removed a redundant model load on
CLI resume and made optional Git provenance tolerate unavailable Git while keeping
implementation hashes. Added regression coverage verifies actual clipping, decay,
failed-save preservation, optimizer-shape rejection, and split/config rejection.
The relevant checks were rerun after these fixes; no actionable review finding
remains open. No independent reviewer or external hardware run was claimed by
that local review.
Subsequent external validation is attributed separately in the closure report.

One draft test initially failed collection because of invalid comprehension syntax;
it was corrected before the first passing training suite. A draft report script
had a lint-only line-length error, fixed before final lint. An earlier local
acceptance run is retained under `outputs/phase-4-acceptance-v1/`; the v2 report
above is the final reviewed-source run with source/memory provenance.

## Preservation and limits

The existing English corpus audit passed: 12,595 records, no exact or near
cross-split duplicates. The accepted tokenizer still has SHA-256
`7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
The Phase 3 random pilot still loads and has weight SHA-256
`99dce7bf0356bf7d641dff4b29d6616772dfbf9779877e0c085bdea9e368b5f2`.
No existing corpus, tokenizer, or pilot checkpoint was overwritten. During the
local implementation checks, no new paid service, remote write, dataset upload,
or weight upload occurred; implementation publication followed owner approval.

Subsequent Phase 4 Windows/CUDA PASS is owner-reported at exact v0.5.0. Completed
GitHub Linux CPU CI has now been directly verified at the same commit. Both are
recorded separately in [closure evidence](CLOSURE.md); the original acceptance and
packaging JSON remain historical records of the pre-publication checks. Independent
physical Linux remains deferred, not performed. Observed exact CUDA equality in the
external validation is not a general CUDA determinism guarantee.

The exact-resume guarantee is limited to the tested same-host CPU execution.
There is no mixed precision, distributed training, streaming corpus loader,
stochastic model layer, or recovery inside a partial optimizer update. Checkpoint
hashes are integrity checks, not authentication; runtime matching does not promise
cross-hardware reproducibility. The 17.3-million-parameter pilot remains untrained.
