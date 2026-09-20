# Phase 7 closure — scoped external validation

Recorded 2026-09-20. **Phase 7 is formally closed locally; closure publication
awaits explicit approval. Phase 8 has not begun and must not begin in this chat.**
The independent Windows/CUDA validation wait is satisfied. The owner requires
reviewed closure publication and verification before an explicit Phase 8 start in
a fresh Work chat. Approval to publish closure does not authorize that transition.

Validated implementation: annotated **v0.8.0**, commit
`04031e5ea98da8db495242165a78c216ab1d4cf4`. Remote main and the peeled tag were
verified again before closure. Tag object:
`485aaf36b7ed81a95c948434e1c14634b220be07`. Keep all ten existing tags unchanged.
This closure changes documentation/evidence only; package remains 0.8.0. Runtime
source, tests, configs, lock, experiment scripts, CI workflow and original measured
results/inventories remain unchanged. [closure.json](closure.json) records the
attributed evidence separately from the original Mac [report](REPORT.md).

## Evidence by environment

| Environment | Result and attribution | Scope |
| --- | --- | --- |
| Mac CPU/MPS | Previously recorded: 210 passed in development and a fresh non-editable wheel, plus separate artifact checks | Full preserved Mac base/SFT artifacts; same-backend cache/logit checks through 512 positions and fixed latency workloads |
| Independent Windows / RTX 4070 SUPER | **PASS**, owner's external summary at exact v0.8.0 | 202 passed, eight expected skips, zero failures; fresh wheel suite; actual CUDA on retained tiny artifacts through capacity 256, plus a separate 512-position synthetic model |
| GitHub-hosted Linux CPU | **PASS**, exact-commit run/job metadata, workflow, tests and logs directly inspected | 203 passed, seven MPS-only skips; tiny CPU inference/CLI/SIGINT tests and existing offline workflows |
| Independent physical Linux | **DEFERRED — NOT PERFORMED** | Hosted CI is separate evidence |

### Owner-reported Windows/CUDA PASS

The owner reports the following at the unchanged validated checkpoint:

- Fresh checkout and locked installation passed. **202 tests passed, eight expected
  platform/hardware skips, zero failures**; the complete suite also passed from a
  fresh non-editable wheel installation.
- Actual cached and uncached inference ran on the NVIDIA RTX 4070 SUPER. CUDA KV
  construction, growth, reuse, reset, invalidation, capacity and error handling passed.
- Same-CUDA full-vocabulary logits met **atol=1e-5, rtol=1e-5**. Fixed greedy token
  IDs agreed across cached, uncached and non-streaming paths. This is scoped numerical
  agreement, not a general determinism or bitwise/cross-device equality guarantee.
- Streaming text, stopping, callback cancellation, history rollback, Unicode
  buffering/final flush and context-boundary checks passed.
- Retained **tiny base/SFT artifacts were exercised through their 256-position
  capacity**. A **separate synthetic model** exercised 512-position and batch-two
  cache mechanics. These are distinct from the Mac learned pilot artifacts.
- Real installed CLI chat ran on CUDA in both development and fresh-wheel environments.
- Scoped RTX 4070 SUPER latency measurements were collected, without a speedup
  requirement or direct comparison against the Mac results.
- Artifact, packaging, privacy and reserved-data preservation checks passed. All
  **143 tracked files** and **694 prior Phase 5/6 evidence files** remained unchanged;
  nothing was committed or pushed from Windows.

Attribution: this is the owner's independent validation summary, not execution by
this Mac session. Raw Windows logs, exact tiny-artifact hashes/configurations,
latency numbers and driver/runtime details were not supplied here; none are
invented. The 694-file count belongs to that Windows installation and is not a
substitute for the separate Mac preservation inventories.

**The full Mac learned base/SFT artifacts were unavailable on Windows and were not
validated there.** Windows OS-level Ctrl-C/console-event delivery was also **not
validated**: the POSIX process-SIGINT test is intentionally skipped on Windows.
Callback cancellation PASS must not be described as Windows console-event PASS.
Useful instruction following remains unestablished. No general CUDA determinism,
cross-device equality, production-latency, mixed-precision or distributed-serving
claim follows. These limits remain part of the accepted result.

### GitHub Linux CPU CI — separately verified

[Run 35529617332](https://github.com/sebastienlato/LatoS/actions/runs/35529617332)
was triggered by the implementation push at
`04031e5ea98da8db495242165a78c216ab1d4cf4`. The successful CPU job
[106127787033](https://github.com/sebastienlato/LatoS/actions/runs/35529617332/job/106127787033)
completed on **2026-09-20 at 18:37:55 UTC**; the run was completed/success at
18:37:56 UTC. The workflow targets Ubuntu 24.04; logs identify Linux x86-64,
Python **3.14.7**, PyTorch **2.14.0+cpu**, and **203 passed / seven skipped in
38.58 seconds** from 210 collected tests. Every skip was MPS-only.

Successful steps: locked install, Ruff lint/format, pytest, CPU doctor and debug
model checks, offline fixture acquire/prepare/audit, source/wheel build, offline
tokenizer fitting/validation, and offline CPU training acceptance.

The exact-commit workflow and tests establish the following boundaries:

- `tests/test_inference.py`: **19 CPU cases passed, four MPS cases skipped**.
  Tiny synthetic CPU models exercise incremental/chunked logits through capacity
  512, batch-two mechanics, causal masking, cache reset/error rollback/invalidation,
  cached/uncached/old-sampler agreement, RNG isolation, Unicode, stops, context and
  history. Subprocess tests exercise installed development-package CLI chat, JSON,
  reset/EOF/errors and **POSIX SIGINT**; the latter is Linux evidence only.
- The three `tests/test_phase6.py` cases passed: a two-update SFT fixture initialized
  from a tiny random CPU base, with both fixture test payloads removed, plus
  in-process verifier replay, preservation/failure/overwrite and wrong-base checks.
  This does not reproduce the full Mac SFT experiment or run a standalone Phase 6
  verifier CLI. Prior training tests include separate-process CPU recovery.
- The explicit Phase 4 acceptance fixture passed at this source version:
  **400 updates / 131,600 target exposures**, 127,808 parameters. Training loss
  5.8146375958 → 0.0100304254; validation loss 5.8069872746 → 11.2483315773.
  Recovery from update 97 matched the remaining 303 non-timing metrics, weights,
  optimizer and sampler exactly within that run. This is fixture memorization,
  not useful language or instruction capability.

CI did **not** run CUDA/MPS, use the full learned Mac base/SFT artifacts, run the
Phase 7 retained-artifact latency matrix, validate Windows console signals, or
validate physical Linux. It built a wheel but did **not** run the full test suite
from a fresh non-editable wheel installation. No standalone archive/privacy audit
or prior-Mac-artifact preservation audit exists in this workflow. Local fixture
preparation/audit is not reserved English/instruction test-set evaluation.

## Closure decision and unchanged quality evidence

The inference engineering deliverable and requested independent Windows/CUDA
validation passed within their stated scopes. No unresolved failure was reported.
The full Mac Phase 6 negative result remains: **200 updates / 6,188 assistant-target
exposures**, assistant loss **8.354879 → 5.815233**, exact replies **0/32 → 0/32**,
and English loss **4.731898 → 5.653422**. Preserve final update 200, the accepted
base separately, and all prior attempts. The experimental SFT is not an improved
base or useful assistant; streaming and cache speed do not change that finding.

Model capacity/mechanism checks do not establish language quality beyond the prior
at-most-256-token training/evaluation windows. External validation grants this
session no access to the Windows GPU. Physical Linux remains deferred. No new paid
service, artifact upload or Phase 8 release work is part of closure.

## Separate closure review and publication

A separate documentation review checked evidence attribution, exact refs/counts,
Windows 256-position tiny artifacts versus synthetic 512-position mechanics,
Windows console-signal exclusions, actual Linux CI scope, quality limits and the
fresh-chat gate. Review clarified that the Mac retained-model checks were separate
from the 210-test suite. **357 prior Mac artifact/data files**, **83 Phase 7 evidence
entries** and all **49 reviewed SFT inventory entries** were rehashed unchanged;
counts describe their own inventories and are not added as unique files. Both
reserved test payloads were integrity-hashed only, never parsed or evaluated.
Source/lock identity, links, JSON, whitespace and publication privacy were checked.
No actionable finding remains. **No new runtime tests, training, inference benchmark
or package build was run for this documentation-only closure.**

After explicit approval, publish only the exact reviewed closure commit to existing
private `https://github.com/sebastienlato/LatoS.git`, branch `main`. **No new tag,
tag movement, release or artifact upload.** Preserve v0.8.0 and package 0.8.0.
Verify the remote closure commit and unchanged tags, record verification locally,
and **stop**. Phase 8 requires an explicit fresh-chat start after verified closure
publication. The prepared [handoff](../../docs/PHASE8_HANDOFF.md) is planning only.
