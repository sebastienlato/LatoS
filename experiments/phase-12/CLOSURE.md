# Phase 12 closure — scoped external validation

Recorded 2026-09-23. **Phase 12 is formally closed locally within its documented
bounded scope; this reviewed closure requires explicit publication approval.** The
owner-reported corrected Windows/CUDA PASS satisfies the external-validation wait.
Do not begin another experiment in this chat. After approved closure publication,
verify the exact remote commit and all eleven unchanged tags, then stop. The next
extension needs an explicit fresh-chat start; [handoff](../../docs/PHASE13_HANDOFF.md).

Validated correction: `b222c1fa844289247531b922d5cb18134e99c7b1`, published and verified,
package 1.3.0. Original experiment: `5b72872a2032f8999aae9c28ff0ab5c704e3d893`.
**No Phase 12 tag exists.** All eleven tag objects and targets remain fixed, including
v1.0.0 at `10d9ef7bf0618b364f887ff9d279408e3cbc33c9`, tag object
`1acb6b94f0adfbca85014ad6601dbfc716e3c651`. Verified publication supersedes historical
pending snapshots. Original experiment and correction evidence remain unchanged.

## Windows/CUDA — owner-reported corrected PASS

The owner reports a completely fresh checkout, environments, report directory and
integrity snapshots. The checkout and fresh non-editable wheel suites each passed
**301 tests, 13 expected skips, zero failures**. These results are owner-reported;
this Mac session did not execute Windows or CUDA. Packaging, lint, formatting,
dependencies, CLI, builds, privacy and general regression checks passed.

The native Windows nested inventory writer/verifier now passes. Missing entries,
extra files and changed nested contents remain rejected. The prior failure at the
original experiment commit remains preserved: the old checkout suite passed 299
with 13 skips, then the path-representation mismatch stopped validation before any
Phase 12 CUDA execution. It was not a dependency, CUDA or model-quality failure.
See the unchanged [correction report](PORTABILITY_CORRECTION.md) and
[structured correction evidence](portability-correction.json).

Actual CUDA inference executed on the NVIDIA RTX 4070 SUPER. KV-cache/model execution
reached all 512 positions. Cached versus full logits passed the existing
**atol=rtol=1e-5** contract; maximum observed absolute difference was approximately
**1.78814e-7**. Position 513 and over-budget inputs were rejected without truncation,
and exact 256/512 output-reservation boundaries behaved correctly. This does not
establish general CUDA determinism, cross-device equality or performance guarantees.

### Synthetic initial blocks are not learned retry transitions

The pinned learned Mac base/SFT/DPO artifacts and English tokenizer were unavailable
on Windows and were not substituted. The bounded Windows exercise used the
repository's original **random synthetic 512-capacity fixture**. The same fixed
model/configuration was used at both lengths; only the session budget changed.

All **16 synthetic sessions initially blocked at 256** could generate at 512,
producing **1,024 actual CUDA-generated tokens**. Each had **zero replies at 256
and one reply at 512**. Those extended sessions yielded **0/16 valid JSON**, no
calls, **0/12 normal tasks and 0/4 learned failure-handling successes**. This is
removal of initial context blocks, not reproduction of the Mac's 44 learned retry
transitions. Scripted controls still pass **12/12 tasks and 4/4 expected failures**;
scripted retry controls demonstrate mechanics, not model learning.

Windows preservation is also owner-reported: **232 tracked files unchanged** and
**3,129 prior regular report/evidence files preserved**, including the failed run.
Both reserved payloads are unchanged and opaque-hashed only; CUDA access guards
recorded zero attempts. Nothing was modified, committed, pushed, tagged or uploaded
from Windows. Raw Windows logs, artifact/harness/prompt/tokenizer hashes, individual
skip breakdown and new driver/runtime versions were not supplied locally; no such
identities are invented. The numerical maximum above is the owner's approximate report.

## Hosted Linux CPU — separately inspected PASS

Directly inspected corrected-commit [run 35880051473](https://github.com/sebastienlato/LatoS/actions/runs/35880051473),
[job 107245869446](https://github.com/sebastienlato/LatoS/actions/runs/35880051473/job/107245869446),
full logs, metadata, exact-commit workflow and relevant test/fixture bodies.
Push run completed 2026-09-23 15:15:37 UTC, hosted Ubuntu 24.04.5 x86-64,
Python 3.14.7 / PyTorch 2.14.0+cpu. **303 passed / 11 MPS-only skips / zero failures**,
314 collected, 50.20 seconds. Phase 12 contributes **eight CPU passes and one MPS skip**.

Phase 12 CPU coverage includes scripted token-length overflow/replay, budget-error
propagation, tiny-model capacity rejection, flat inventory corruption/missing entry
rejection, forged-generation-error rejection, and real tiny CPU inference beyond
256 compared with direct generation. Both final nested inventory variants pass:
native POSIX and simulated Windows relative-path serialization. They exercise
complete nested membership, repeat writes and rejection of changed nested content.
The scripted multi-turn fixture is not a learned model. Native Windows was not run.

Existing synthetic CPU cache tests reach 512 positions; existing tool/protocol and
other runtime regressions pass. Additional workflow steps pass: locked install,
lint/format, CPU diagnostics/debug model, offline tiny data/tokenizer workflows and
source/wheel builds. Dense fixture acceptance performs 400 updates / 131,600 target
exposures, checks update-97 recovery across 303 updates with equal weights, optimizer,
sampler and non-timing metrics, and unchanged validation state. This remains dense
fixture recovery, not tool learning or supported adapter/DPO optimizer resume.

This workflow does **not** run the standalone full Phase 12 experiment/replay, the
full learned Mac artifacts, the Windows 16-session synthetic CUDA exercise, CUDA/MPS,
a fresh installed-wheel suite, full learned-artifact/reserved-payload preservation
audit or CUDA access guards. The older verifier unit fixture uses scripted data
under base/SFT/DPO labels, not those learned checkpoints. Fixture data workflows are
not evaluation of the reserved full-project payloads. **Physical Linux is deferred.**
Log/metadata/workflow hashes are recorded in [closure.json](closure.json); raw copies
remain local. No additional Linux model-quality claim is inferred.

## Original Mac research and capability limits

| Evidence | Transition actually observed | Learned outcome |
| --- | --- | --- |
| Mac learned base/SFT/DPO, English tokenizer | 44 previously blocked retries gained replies: 12/16/16 by model | No valid JSON/calls/tasks/learned failure handling |
| Windows random synthetic fixture | 16 initial blocks removed: zero replies → one reply per session | 0/16 JSON, zero calls, 0/12 tasks, 0/4 failure handling |
| Scripted controls | Predetermined correct responses and retries | Mechanics only; not learned evidence |
| Hosted Linux CPU | Scoped synthetic tests and workflows | No full research replication or capability result |

The original [Mac report](REPORT.md), [plan](PLAN.md), results/samples/inventories,
both attempts and correction history remain unchanged. At 256 the baseline exactly
reproduces Phase 11. At 512, base/SFT/DPO parsing remains **0/41, 0/64, 0/64 emitted
replies**; each has 0/12 tasks and 0/4 learned failure handling. Token counts remain
595/48/51 at 256 and 1,834/192/195 at 512, 2,915 per run across 96 sessions. Two
research runs, zero training; no correction or closure research rerun. Original Mac
suites passed 312 each; correction Mac suites passed 314 each. H1 (retry room) passed;
H2 (at least one normal task success) failed. Capacity relief is not capability gain.

Preserve prior negative SFT/LoRA/control/DPO results, including full Mac DPO ranking
16/32 → 15/32 and exact 0/32, English loss still worse than original base; distinct
tiny Windows and hosted Linux scopes remain in [Phase 10 closure](../phase-10/CLOSURE.md).
[Phase 11 closure](../phase-11/CLOSURE.md) retains distinct negative full Mac and
compact-prompt tiny Windows results and separately scoped Linux evidence.

Preserve all models/tokenizers, chat contract 1, both reserved tests and failures.
Merge atol=rtol=1e-4 and the original CPU 1e-5 failure remain; cache CPU/CUDA 1e-5,
MPS 1e-4. Adapter/DPO optimizer resume unsupported; dense recovery version-bound;
Windows OS-level Ctrl-C unvalidated. No learned tool use, useful instruction following,
factual reliability, learned long-context quality, general accelerator determinism,
cross-device equality, safety alignment, production performance, mixed precision
or distributed serving is established.

## Closure checks and publication

Documentation/evidence only: runtime, tests, configs, lock, workflow, package version
and original research/correction evidence unchanged. **78,433 local files rehashed
unchanged**, including earlier preservation inventories and original/corrected Phase 12
source snapshots, builds, logs and evidence. Both reserved tests opaque-hashed only.
Windows preservation counts remain separately owner-reported. JSON, local links,
source identity, CI provenance/counts, full staged diff and privacy checks pass;
see [separate review](CLOSURE_REVIEW.md). No new runtime suite, training, benchmark
or build was run for this closure.

Propose the reviewed closure to existing `https://github.com/sebastienlato/LatoS.git`,
main only, after explicit approval. No tag, release, assets or visibility change.
After approved publication verify exact main and all eleven unchanged tags, then stop.
No next method, dataset, baseline or configuration has been selected; use the
[concise fresh-chat handoff](../../docs/PHASE13_HANDOFF.md) only after verification.
