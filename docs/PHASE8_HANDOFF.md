# Phase 8 fresh-chat handoff

Preparation only. **Do not begin Phase 8 in the Phase 7 closure chat.** First obtain
explicit approval for the reviewed closure commit, publish that exact state and
verify remote main and unchanged tags. Then require an explicit owner start in a
fresh Work chat. Closure push approval alone does not override this gate.

## Starting point and evidence boundaries

Read AGENTS.md, PROJECT_STATE.md, ROADMAP.md, this handoff, and local private context
and blueprint when present. Private planning stays outside tracked content and
publication metadata. Verify the actual closure commit against its approval and
publication record; the implementation tag alone is not evidence of closure.

Phase 7 implementation is published as annotated **v0.8.0** at
`04031e5ea98da8db495242165a78c216ab1d4cf4`, tag object
`485aaf36b7ed81a95c948434e1c14634b220be07`. Keep that tag fixed. Phase 7 closure is
documentation-only and retains package 0.8.0. See its
[closure evidence](../experiments/phase-7/CLOSURE.md) and
[structured summary](../experiments/phase-7/closure.json).

- Mac CPU/MPS: 210 tests passed in development and a fresh non-editable wheel;
  full preserved Mac base/SFT inference was measured through capacity 512. Same-
  backend tolerances: CPU atol/rtol=1e-5, MPS atol/rtol=1e-4. Numerical capacity is
  not language quality; earlier training/evaluation windows were at most 256.
- Independent Windows / RTX 4070 SUPER: **owner-reported PASS**, 202 passed,
  eight expected skips, zero failures, plus fresh wheel suite and actual CUDA CLI.
  **Retained tiny base/SFT artifacts through capacity 256**, plus a **separate
  synthetic 512-position/batch-two model**. Same-CUDA full-vocabulary logits met
  atol=1e-5, rtol=1e-5; fixed cached/uncached/non-streaming greedy IDs agreed.
  Cache/streaming/stopping/context/callback cancellation checks passed. The full
  Mac learned artifacts were unavailable and **not validated on Windows**.
  **Windows OS-level Ctrl-C/console-event delivery remains unvalidated**; the
  POSIX SIGINT test was skipped. Raw Windows logs/hashes/latency values were not
  supplied here, and latency must not be compared directly with Mac measurements.
- GitHub-hosted Linux CPU: directly inspected
  [run 35529617332](https://github.com/sebastienlato/LatoS/actions/runs/35529617332)
  at exact v0.8.0: **203 passed, seven MPS-only skips**. Tiny CPU inference/CLI,
  POSIX SIGINT and earlier SFT/recovery fixtures; locked setup, lint/format,
  offline workflows and builds. No full learned-artifact validation, CUDA/MPS,
  Phase 7 retained-artifact latency matrix or fresh-wheel-installed suite.
- Physical Linux remains **deferred, not performed**. Useful instruction following,
  general CUDA determinism, cross-device equality, production latency, mixed
  precision and distributed serving remain unestablished.
- Full Mac SFT quality result remains negative: 200 updates / 6,188 assistant-target
  exposures, assistant loss 8.354879 → 5.815233, exact replies 0/32 → 0/32,
  English loss 4.731898 → 5.653422. Keep the fixed final update 200; do not promote
  the experimental SFT checkpoint as an improved base or useful assistant.

## Preserved local inputs

A clone does not contain ignored artifacts. Verify availability and hashes before
using them; do not overwrite, silently substitute, refit, or assume remote backup.

- Base: `outputs/phase-5-english-pilot/step-00003000/model/`, weights SHA-256
  `f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa`.
- Experimental SFT: `outputs/phase-6-instruction-reviewed/step-00000200/model/`,
  weights SHA-256 `62b890214e5544292cc54b725a4e505730fca0d501b3337cdd3f051a622d09ae`.
- Tokenizer: `artifacts/tokenizers/english-bpe-v1/`, 8,192 entries, SHA-256
  `7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
- Evidence: Phase 5/6/7 reports and inventories, all prior attempts/failures, random
  baseline `checkpoints/phase-3-pilot-initial/`, and `outputs/phase-4-*` through
  `outputs/phase-7-*`. Phase 7 measurements/source snapshots/logs are under
  `outputs/phase-7-validation/`, with an 83-entry
  [inventory](../experiments/phase-7/artifacts.json). The Phase 6 reviewed run has
  its separate 49-entry [inventory](../experiments/phase-6/artifacts.json).
- English corpus `data/processed/english-books-v1/` and instructions
  `data/processed/english-instructions-v1/`: **both test payloads remain reserved**.
  Do not consume them for routine reproduction, tuning or artifact selection.
- Shared formatter: `src/latos/chat.py`, `CHAT_CONTRACT` version 1. Preserve segmented
  encoding, BOS/EOS, role order and generation-prefix identity. Read
  [INFERENCE.md](INFERENCE.md), [INSTRUCTION_TUNING.md](INSTRUCTION_TUNING.md),
  [TRAINING.md](TRAINING.md) and [DATA.md](DATA.md).

Model-only loading and optimizer recovery are distinct. Recover earlier optimizers
only with their recorded original runtime/source; do not rewrite retained artifacts
for a new package. Recheck the actual host, available backends, memory and disk.
Prior local host was M4 Max/64 GiB, not a promise about the next session. Local uv:
`.private/tools/bin/uv`, runtime `.venv`. No new paid services are authorized.

## Phase 8 objective — only after verified closure and explicit fresh-chat start

Prepare the roadmap's reproducible release: a tested fresh-checkout guide,
data/model cards, consolidated experiment report, and an inspected plan/package
for permitted artifacts. These are future deliverables, not work completed here.

- Distinguish tiny mechanism reproduction from the full English/SFT experiments.
  Use available resources and record commands, source/runtime identities, actual
  outcomes and missing-input/hardware blockers. Do not claim a fresh clone contains
  ignored learned artifacts or that external checks reproduced the Mac experiments.
- Document data provenance and applicable redistribution terms, tokenizer/model
  hashes, selected checkpoints, training exposures, negative results and platform
  boundaries. Preserve the base and experimental SFT identities separately.
- Inspect any proposed release bundle using explicit permitted files; never archive
  the entire working folder. Keep private context, credentials, environments,
  acquired data, logs and disallowed artifacts out. Git history must not gain weights
  or datasets. Do not change visibility, incur charges or upload before approval.
- Keep evaluation reservations and inference/chat compatibility intact. Do not add
  LoRA, preference tuning, tools or serving extensions to this release phase.
- Complete appropriate validation and a separate review/fixes; prepare the actual
  local commit and concrete release contents/destination before the Phase 8 push
  checkpoint. Disclose any proposed v1.0.0 tag/release/assets in that approval request;
  neither Phase 7 approval nor this handoff authorizes remote release publication.

## Concise prompt — copy only after closure publication is verified

> Start LatoS Phase 8 in this fresh Work chat after verifying the published Phase 7
> closure commit against its approval/publication record and confirming v0.8.0
> remains at 04031e5ea98da8db495242165a78c216ab1d4cf4. Read AGENTS.md,
> PROJECT_STATE.md, ROADMAP.md, docs/PHASE8_HANDOFF.md and local private context/
> blueprint when present. Preserve the base, experimental SFT, tokenizer, shared
> chat format, prior evidence and both reserved test sets. Windows/CUDA PASS used
> tiny 256-position artifacts plus a synthetic 512-position model, not the full Mac
> learned artifacts; Windows console Ctrl-C remains unvalidated. Linux CPU CI passed
> separately within its documented scope; physical Linux remains deferred. Retain
> all quality, determinism, cross-device and performance limits. Prepare the
> reproducible release guide, data/model cards, experiment report and permitted
> artifact package using existing resources without new paid services. Complete
> validation and separate review/fixes, prepare the local commit and concrete
> publication proposal, then stop for explicit Phase 8 publication approval.
