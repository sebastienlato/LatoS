# Phase 9 closure — scoped external validation

Recorded 2026-09-21. **Phase 9 is formally closed locally within the documented
scope; closure publication awaits explicit approval. Phase 10 has not begun and
must not begin in this chat.** The owner-supplied independent Windows/CUDA result
satisfies the external-validation wait. Publish and verify the reviewed closure
only after approval, then stop. Phase 10 requires an explicit fresh-chat start.

Validated implementation: `0fbedd31f746d30bb5f071fbda30defa9036a451`, published and
verified on remote main. **There is no Phase 9 tag.** All eleven existing tags stay
fixed; annotated v1.0.0 still targets `10d9ef7bf0618b364f887ff9d279408e3cbc33c9`,
object `1acb6b94f0adfbca85014ad6601dbfc716e3c651`.

This closure changes documentation/evidence only. Package remains 1.1.0; source,
tests, configs, lock, workflow, experiment scripts and original results/inventories
are unchanged. The original [report](REPORT.md) and [validation](validation.json)
remain historical implementation snapshots. This closure and [closure.json](closure.json)
supply subsequent publication/platform status without rewriting earlier evidence.

## Evidence by environment

| Environment | Result and attribution | Scope |
| --- | --- | --- |
| Mac CPU/MPS | Previously recorded: 236 passed in development and fresh non-editable wheel | Full retained Mac base/adapter/control experiment; independent merge/cache through 512, separate from the test suite |
| Windows / RTX 4070 SUPER | **PASS**, owner's independent validation summary at the exact implementation commit | 226 passed, 10 expected skips, zero failures; fresh wheel suite; actual CUDA tiny LoRA/full-tuning exercise |
| GitHub-hosted Linux CPU | **PASS**, run/job metadata, logs and exact-commit workflow/tests directly inspected | 228 passed, 8 MPS-only skips; tiny CPU LoRA/full-tuning tests and existing offline workflows |
| Physical Linux | **DEFERRED — NOT PERFORMED** | Hosted CI is separate evidence |

## Owner-reported Windows/CUDA PASS

The owner reports:

- Fresh exact-commit checkout and locked installation passed. **226 tests passed,
  10 expected platform/hardware skips, zero failures**; the complete suite also
  passed from a fresh non-editable wheel installation.
- Actual LoRA and full-tuning training ran on the **NVIDIA RTX 4070 SUPER**. Both
  methods completed **4 updates / 175 assistant-target exposures**, with matching
  data, update/exposure budget, schedule and evaluation conditions.
- All **7,856 tiny-base parameters remained unchanged** during LoRA. Exactly
  **192 adapter parameters were trainable and updated**. Adapter initialization,
  frozen-base behavior and optimizer membership passed.
- The trained tiny ratio is approximately **2.44399% of base parameters**. The
  independently checked **0.851951279%** ratio belongs to the full pilot
  architecture (147,456 / 17,308,032), not this tiny trained model.
- Exact adapter save/load round-trip passed. CPU and CUDA adapter-versus-merged
  checks passed the unchanged amended **atol=rtol=1e-4** contract. Cached/uncached
  checks retained the separate non-MPS **atol=rtol=1e-5** criterion.
- Independent CPU float64 merge-reference diagnostics passed. The original
  pre-amendment CPU 1e-5 merge failure remains preserved and was not reclassified.
- Unsupported adapter optimizer checkpoint/resume was correctly rejected; no
  LoRA optimizer-resume capability is claimed.
- Base, LoRA and full-tuned tiny models all remained **0/32 exact held-out replies**.
  Reserved test payloads remained unused. This is validation evidence, not reserved
  test-set evaluation.
- All **1,513 prior evidence/report files** and **173 tracked files** were preserved;
  the working tree remained clean. CLI, diagnostics, lint, formatting, dependency
  integrity, offline workflows, packaging and privacy checks passed. Nothing was
  committed or pushed from Windows.

Attribution: owner-supplied independent report, not execution by this Mac session.
Raw Windows logs, exact artifact hashes/configuration, numerical error maxima and
Windows driver/runtime details were not supplied here; none are invented. The
1,513-file count belongs to that Windows installation and must not be combined
with Mac preservation counts as though they were one verified inventory.

This is a **small faithful Windows/CUDA mechanism exercise**, not reproduction of
the full Mac learned-base/adapter experiment. **Full learned 512-position merge
behavior was not validated on Windows.** The summary does not supply a new Windows
console Ctrl-C validation; that earlier limitation remains. No useful instruction
following, LoRA superiority, general CUDA determinism, cross-device equality,
mixed-precision/distributed behavior or production-performance claim follows.

## GitHub Linux CPU CI — separately inspected

[Run 35617504810](https://github.com/sebastienlato/LatoS/actions/runs/35617504810),
CPU [job 106391902352](https://github.com/sebastienlato/LatoS/actions/runs/35617504810/job/106391902352),
was triggered by the implementation push at
`0fbedd31f746d30bb5f071fbda30defa9036a451`. The job and run completed successfully
on **2026-09-21 at 15:14:56 UTC**. Logs identify Ubuntu **24.04.5**, Linux x86-64,
Python **3.14.7**, PyTorch **2.14.0+cpu**, **228 passed / 8 skipped in 32.95 seconds**
from 236 collected tests. All eight skips were MPS-only. Raw metadata/logs are
retained locally with hashes in the structured closure record.

The exact-commit workflow and test bodies establish these boundaries:

- `tests/test_lora.py`: **19 CPU cases passed, one MPS case skipped**. Tiny models
  cover equation/gradients, zero initialization, frozen base, optimizer membership,
  read-only validation, adapter round trip, merge/cache, corruption and identity
  rejection, rejected adapter optimizer checkpointing, and repeatable CPU updates.
  The tiny unit merge case passes 1e-5; this does not erase the full learned CPU
  1e-5 failure or replace the amended 1e-4 contract.
- `tests/test_phase9.py`: **three CPU integration cases passed**. A tiny random
  base, capacity 256, runs two-update LoRA/full tuning under matching conditions;
  adapter reload, merge at 1e-4, merge CLI, rejection of incomplete comparisons,
  failure retention, wrong-base and overwrite checks pass. Both fixture test
  payloads are removed before these runs. This exercises comparison machinery,
  not the full 200-update learned-model experiment or its performance measurements.
- Existing CPU inference/CLI tests, including POSIX SIGINT, and prior SFT/training
  recovery tests passed. These do not validate Windows console events.
- Locked installation, Ruff lint/format, CPU doctor/debug-model checks, offline
  data acquisition/preparation/audit, source/wheel build, offline tokenizer training
  and validation, and the Phase 4 offline training acceptance all passed.
- The explicit Phase 4 acceptance completed **400 updates / 131,600 targets**.
  Training loss 5.814638 → 0.010031; validation worsened 5.806987 → 11.252522.
  Recovery from update 97 matched 303 subsequent non-timing metrics, weights,
  optimizer and sampler exactly within this CPU run. This is dense fixture
  memorization/recovery, **not LoRA optimizer resume** or useful language quality.

CI did **not** run CUDA/MPS, reproduce full Mac learned artifacts, run full learned
512-position merge verification, invoke the standalone Phase 9 full-artifact or
float64 reference scripts, perform the learned-model performance comparison,
validate physical Linux, or run a fresh non-editable wheel test suite. Building
a wheel is distinct from installing it and retesting. Packaging-related unit tests
are not a standalone archive/privacy audit or a prior-Mac-artifact preservation
audit. Fixture tests do not evaluate the reserved real English/instruction tests.

## Unchanged numerical and quality limits

The CPU float32 merge threshold was explicitly amended after a real 1e-5 failure
to **atol=rtol=1e-4**. Original failure evidence, diagnostic and protocol amendment
remain unchanged. CPU/CUDA cache tolerance remains separately scoped at 1e-5,
MPS cache at 1e-4. The original full Mac float64 diagnostic is a reference check,
not mixed-precision training or a platform-wide determinism guarantee.

Full Mac Phase 9 remains **0/32 exact replies for base, LoRA and full tuning**.
Both adaptations saw 200 updates / 6,188 targets; English loss was base 4.731898,
LoRA 4.773844, full 5.653425. The original Phase 6 SFT is still negative:
200 updates / 6,188 targets, assistant loss 8.354879 → 5.815233, exact 0/32 → 0/32,
English loss 4.731898 → 5.653422. Preserve its fixed final update 200 and the base
separately; neither tiny success nor lower teacher-forced loss promotes a model.

Useful instruction following, factual reliability, safety alignment, broader
English quality, quality beyond 256-token windows, general accelerator determinism,
cross-device equality, production latency, mixed precision and distributed serving
remain unestablished. Adapter optimizer resume remains unsupported. Both reserved
test sets, tokenizer, shared chat contract, full base/SFT, new adapter/merge/control
artifacts, random baseline and prior attempts/failures remain preserved locally.

## Review and publication

[Separate closure review](CLOSURE_REVIEW.md) checks attribution, fractions,
numerical contracts, CI scope, unchanged runtime/evidence and the fresh-chat gate.
**743 distinct local files rehashed unchanged**, covering the earlier 507-file set,
225 Phase 9 inventory entries and original Phase 9 evidence/scripts. Windows counts
remain separately attributed. Both reserved payloads were hashed only as opaque
bytes. JSON, links, staged diff, whitespace, history/privacy and publication scope
were checked. No new runtime tests, training, inference benchmark or build was run
for this documentation-only closure.

Propose only the exact reviewed closure commit to existing public
`https://github.com/sebastienlato/LatoS.git`, branch `main`. **No tag, tag movement,
release, artifact upload, visibility change or paid service.** After explicit
approval, publish and verify the closure commit and unchanged tags, then stop.
The [Phase 10 handoff](../../docs/PHASE10_HANDOFF.md) is preparation only; it may be
used in a fresh chat only after closure publication is verified.
