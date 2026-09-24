# Phase 17 first Windows interruption — independent Mac review

The return archive matches the owner's SHA-256
`b4d959194b29f68ae02f7c3340e70f02d88d6aeb0d10e7b9ab4eebe7d0b5338e`.
The independently checked [record](interruption-review.json) binds all 3,761 included
files and the executed 62-file package snapshot to the authoritative local training
commit `0fa94ad580110fd2dc7aaa2aa3560950abe73a02`. Returned administrative Python was
not executed. The original return ZIP and receipt remain preserved, unmodified.

## What the evidence establishes

Both original Windows suites have 411 passes, 15 skips and zero failures. Both
required CUDA numerical/tail-recovery tests executed in checkout and isolated
wheel. Wheel runtime files match the exact source. The 15 skips are the documented
MPS, POSIX SIGINT and unavailable Windows symlink paths, not missing CUDA controls.

All 7,485 contiguous updates independently reconstruct from the fixed seed-160
permutation and full-corpus cache. They executed 37,811,418 targets over 119,748
windows, with two microbatches / 1,520 targets in the final update. Epoch remains
zero; learning rates match the frozen schedule. Logged values are finite. The
latest retained checkpoint is step 7,000: 35,384,765 targets and 112,000 windows.
The 485 later updates executed 2,426,653 targets, but their updated state was lost.
No final checkpoint, training completion receipt or fixed evaluation exists.
Step 7,000 cannot substitute for the planned final candidate.

Peak **logged** reserved GPU memory is 1,178,599,424 bytes (1.10 GiB); host working
set is 2,545,315,840 bytes (2.37 GiB). An abrupt stop prevented a final worker peak
receipt, so these are the retained observations, not a claim about unobserved final
moments. No numerical, quality or resource violation is recorded. Receipt disk
accounting remains below 20 GiB; acquired text, caches and tensors stay outside Git.

Eight complete checkpoint inventories are consistent with returned metadata, and
the full-cache metadata/binary identities match Mac's independently built cache.
The tensor bytes remain on Windows. Mac verified their inventory bindings, not
absent bytes or a numerical checkpoint replay. The recovery preflight must rehash
those originals on Windows, and the original strict CUDA loader must accept them.

## Failure and causation

The preserved main ledger is stale at 1,560.9963497 seconds and still says `running`.
Its separate temporary file records `supervisor-failure` at 1,561.6473704 seconds,
with Windows access denied while replacing the main ledger. The verified source's
exception path kills/waits for the worker, then its final ledger replacement also
fails. The returned process inspection shows no surviving supervisor or worker.
The worker was interrupted during post-update monitoring before saving step 7,485.

The concurrent PowerShell read is a plausible contributor, not an established
cause. The lock owner was not instrumented. Windows sharing modes can prevent
rename/delete access while a read handle is open; this explains a possible
mechanism, not this incident's actual attribution. See Microsoft's
[CreateFile sharing semantics](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew).
The new Windows-only fixture will test that mechanism on disposable files; it will
not retroactively prove who held the production handle.

## Recovery decision

**A same-attempt checkpoint replay is viable, conditionally on Windows integrity
and strict recovery checks. A stock resume is not the appropriate action.** Do not
use the stale `running` status, relabel the failed job, edit either ledger, promote
step 7,000, invent a final checkpoint or start a fresh run.

The unchanged optimizer worker was terminated by its supervising process for a
non-numerical administrative I/O failure. The existing interruption allowance can
recover that complete checkpoint without changing the mathematical experiment.
This is a separately reviewed corrective execution path under the owner's request,
not an assertion that an unrelated external process caused the failure. It consumes
**the first of the two allowed same-attempt resumptions**. This handoff permits one
corrective launch only; another failure returns to Mac review, without automatic
use of the second allowance.

Patching the imported `latos` package would change its implementation hash and
invalidate the source/runtime recovery contract. The correction therefore lives
**outside** that package and outside the preserved source checkout. A separately
hashed administrative launcher calls the exact original worker in the original
checkout environment. It leaves both failed ledgers, old logs, source and checkpoints
unchanged. New progress uses a persistent append-only journal; no active ledger is
replaced. OS locks exclude concurrent or orphaned workers. There is no monkeypatch,
metadata migration, checkpoint rewrite or override of the strict loader.

The only optimization is replay of updates 7,001–7,485 from the full step-7,000
optimizer/sampler checkpoint, with the same source/runtime/configuration, data,
precision, schedule and final partial accumulation. No model reinitialization,
additional epoch, seed or checkpoint selection. The planned logical trajectory
still has 7,485 updates and 37,811,418 targets. Total physical work after a successful
replay would be 7,970 update executions and **40,238,071 target executions**, of which
2,426,653 are explicitly repeated work. They are never called new unique data.

Charge **1,562 seconds** for the original attempt, rounding its latest failed-ledger
time upward. Recovery verification, restart, replay, final monitoring/readback and
post-run verification together have at most **5,638 additional active seconds**.
The fixed evaluation reserve remains **3,600 seconds**, shared by development and
any later permitted final comparison; the three-hour total is unchanged. The hard
artifact/host/GPU limits stay unchanged. A conservative extra artifact reserve
covers recovery tooling and return copies rather than increasing the cap.

The administrative preflight rehashes every original attempt file, including all
checkpoint tensors, and the full source/input transfer. The unchanged worker checks
the exact runtime and loads the pinned checkpoint without relaxation. Replay
counters and schedule must match exactly; logged losses must remain within the
existing 1e-5 same-backend recovery tolerance. This does not establish bitwise
equality to absent lost final weights. A discrepancy stops before acceptance.
After a valid final checkpoint and complete corrective receipt, run the already
planned paired fixed development evaluation. A negative score is retained. Passing
development still requires Mac selection/contamination review before final access.

The concrete [recovery guide](recovery/RECOVERY.md) and separately verified local
bundle supply this bounded path. Actual recovery remains unexecuted on Mac; the
Windows sharing fixture is untested here. Phase 17 remains incomplete. No Phase 18,
publication, source replacement or scientific-contract amendment is authorized.
