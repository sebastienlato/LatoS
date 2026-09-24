# Project state

Updated: 2026-09-24.

## Active checkpoint

**Phase 17 — LatoS Base Model 2.0 is explicitly authorized and in progress.**
Mac implementation prepares the frozen one-pass training path and bounded Windows
handoff. **No serious Phase 17 model has been initialized/trained on Mac, and no
Phase 17 learned-quality result exists yet. Actual RTX 4070 SUPER execution and
returned evidence are required. Phase 18 is explicitly prohibited.**

Phase 16 publication is verified at `ce9de1b734a8e23fd54e9d137cf269f484631765`:
local/remote main were clean and synchronized, with eleven unchanged tags and the
existing release/four asset records unchanged. This supersedes its historical
pending-publication wording. Package remains 1.3.0; v1.0.0 still targets
`10d9ef7bf0618b364f887ff9d279408e3cbc33c9`. No new remote write is authorized.

## Frozen scope and implementation

[Training contract](docs/TRAINING_2.md), [plan](configs/training2/phase17-plan.json),
selected model and Phase 14 evaluation protocol remain byte-identical. The plan's
historical `execution_authorized: false` is superseded solely by the owner's explicit
Phase 17 start; no numeric bound changes. [Execution plan](experiments/phase-17/PLAN.md)
and [Windows handoff](docs/PHASE17_WINDOWS_HANDOFF.md) define the active work.

- Fresh seed 160, native 8-layer / 34,087,424-parameter model, context 512, CUDA BF16,
  batch two × accumulation eight; final update uses two microbatches. One attempt,
  one pass, 37,811,418 targets / 119,748 windows / 7,485 updates. No probe reuse.
- Compact verified full-data cache and explicit one-permutation trainer, with
  token-weighted final accumulation and strict versioned checkpoint recovery.
- Source/input/runtime-bound Windows supervisor, checkout/wheel preflight, actual
  CUDA tail/numerical controls, immutable checkpoints, interruption ledger/process
  locks, hard active time and artifact bounds, and independent evidence verifier.
- Two active training/preparation/recovery hours plus one shared development/final
  evaluation hour. 6 GiB planned artifacts, 20 GiB hard cap, 8 GiB host memory,
  85% reserved VRAM. At most two exact external-interruption resumptions.

## Validation and limitations

[Mac cache evidence](experiments/phase-17/cache-validation.json) reproduces the exact
Phase 16 full train/development sequence hashes and counts, with zero group overlap.
Training cache creation/readback took 37.91 seconds, development 0.83 seconds;
process peak RSS 488,357,888 bytes. This was data validation, not optimization.
[Validation](experiments/phase-17/validation.json) and
[separate review](experiments/phase-17/REVIEW.md) record final checks and fixes.
Synthetic mechanics remain separate from learned training. No new CUDA execution,
CI inspection or physical Linux validation is claimed; physical Linux is deferred.

The local transfer includes the exact historical Phase 5 model/tokenizer and pinned
fixed evaluation inputs for paired CUDA evaluation. Original LM test payload is
opaque-copied/hashed only for later gated final access. No reserved scoring or
content inspection occurs during preparation. No old instruction test is included.
New final model bytes must return to Mac; original optimizer/intermediate tensors
stay on Windows with checksummed inventories. The return subset is not their backup.

Fixed gates and historical negatives remain: 1.872944 BPB, ARC-Easy 157/570,
Challenge 66/299, all five historical models 0/96 instructions. Feasibility does
not predict learned success. Development/resource failure preserves a negative
result and blocks final scoring and Phase 18. Development success requires a
separate selection/contamination review before one final comparison; no final tuning.

## Next action and publication

Mac implementation and separate review/validation are complete: **424 passed / two
CUDA skips** in both development and installed-wheel environments. The reviewed
local source is ready for the checksummed execution transfer; its exact commit and
archive identity are recorded in the handoff manifest and ignored local record.
There is no connected Windows/CUDA executor in this Mac task; the bounded handoff must run in the owner's Windows Work session.
Return actual training/development evidence for Mac verification and integration.
If development succeeds, complete the already-authorized final review/comparison
within the remaining frozen budget. Phase 17 is incomplete at this external boundary.

All phase source work remains local. No push, remote branch, PR, tag, release or
asset upload. After actual result integration and separate review, prepare the
final exact commit and request Phase 17 publication approval to existing
`https://github.com/sebastienlato/LatoS.git`, destination main only. A handoff commit
is not the phase-completion publication proposal. Do not begin Phase 18, including
after any Phase 17 publication. Private context and large/raw artifacts stay ignored.
