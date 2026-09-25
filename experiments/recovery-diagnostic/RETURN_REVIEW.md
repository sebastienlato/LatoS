# Independent Mac review of the completed CUDA diagnostic

The frozen diagnostic acceptance rule **passes**. This supports the combined
prospective deterministic execution policy on the tested RTX 4070 SUPER/runtime
and synthetic shape schedule. It does not repair Phase 17, accept its recovered
model, identify the exact historical kernel, or demonstrate learned language quality.
Published Phase 17 remains permanently interrupted/failed at
`09c1935b085218fbbc9ea88703d8ba12e63c516e`.

## Source and evidence verification

The owner's return SHA-256 is
`340b57d7f201436e47b575b7a792cb208cb74dfe47cd5f835322d313304c47d9`.
The 23,129,151-byte ZIP and receipt agree. All 318 included files rehash and match
their extracted bytes. The 11 diagnostic payloads bind to the original reviewed
commit `0e5b55f516b6dd53a48f53928df2616872ca5f82`. Four executed native-package
snapshots, each containing 62 Python files, match that commit exactly. No returned
Python source was executed by the Mac review. The new local
[reviewer](review_return.py) independently reconstructs the result from records.

Fourteen omitted tensor/cache payloads remain inventory-bound on Windows; their
bytes were not rehashed, loaded or numerically replayed on Mac. Included checkpoint
metadata binds the omitted weights and optimizer files. Deterministic final model
and training payload identities match between reference and replay. The return
is not a backup of those retained tensors.

The 10 CPU tests passed with zero skips/failures; their code forbids optimizer
updates. The journal has one preparation job, four completed CUDA jobs, one start
and one terminal completion. There are no retries or failure records in the return.
This is evidence for the recorded study, not an audit of every process on Windows.

## Reconstructed comparison

| Measure | Legacy | Prospective deterministic |
| --- | ---: | ---: |
| Reference / replay updates | 997 / 485 | 997 / 485 |
| First full-state difference | 674 | none |
| First loss excess over `1e-5` | 675 | none |
| Loss excess count | 303 / 485 | 0 / 485 |
| Different full-state updates | 324 / 485 | 0 / 485 |
| Maximum absolute loss difference | 0.0004386934420383959 at 741 | 0 |

The fresh initial states agree across profiles. Both split checkpoint roundtrips
and both new-process restorations exactly match their own reference state. Final
roundtrips also match. Every update's full state digest reconstructs, with matching
configuration, dataset identities, optimizer parameter mapping, fixed global RNG,
sampler order/counters and input signatures. All same-Windows learning rates match
exactly, across both profiles and both continuations. Exact state agreement is
required in addition to the unchanged loss gate.

Physical work is 2,964 optimizer updates and 14,897,204 synthetic target executions.
Each fresh trajectory consumes 5,021,949 targets; each replay repeats 2,426,653.
These are mechanics exposures, never new language data or a candidate base model.

Supervised duration, including preflight, is 1,458.462660 seconds. The included
administrative receipt ends at 1,545.051568 seconds before packaging; the owner's
later 1,572-second total is retained as an owner report, not independently timed
on Mac. Both are below the authorized 3,600 seconds. Peak recorded host memory is
2,610,409,472 bytes (2.43 GiB); reserved GPU memory is 1,237,319,680 bytes (1.15 GiB).
The supervisor's conservative peak artifact charge is 3,626,208,730 bytes, including
its 1 GiB reserve. Inventoried retained bytes plus the ZIP total about 2.40 GiB.
All measured bounds pass. No new paid resource, held-out score or final access is
present in the verified study source/evidence.

## Mac verifier finding and separate review

The original [verifier](verify_return.py) failed at its cross-host learning-rate
recomputation check. It remains unchanged, and this failure is not presented as a
passing invocation. Eight distinct cosine-schedule outputs differ between Windows
and Mac; five exceed its one-output-ULP check, with a maximum of three ULPs. These
same values match exactly in every Windows execution at the corresponding step.

The bound source uses `math.cos`; CPython documents that its math functions mostly
wrap the platform C math library ([Python documentation](https://docs.python.org/3/library/math.html)).
Propagating either immediately adjacent cosine result through the unchanged
floating-point expression reproduces every observed difference. This supports a
cross-platform arithmetic explanation; it is not proof of a particular libm routine.
The original check assumed that one ULP at cosine implied one ULP at the final rate,
which this arithmetic does not guarantee.

The separate reviewer requires exact same-Windows rates at every corresponding
update and retains the cross-host calculation as an explicitly scoped consistency
check. No CUDA loss tolerance, exact-state requirement, optimizer schedule, input,
profile or outcome was changed. Cross-device bitwise schedule equality was not the
frozen experimental acceptance criterion. Changing the Mac verification method
after observing this verifier limitation is disclosed here, with a negative test
that rejects even a one-ULP same-Windows schedule difference.

A separate pass strengthened the reviewer to check initial random/empty-optimizer
state, checkpoint sidecars against trace endpoints, full checkpoint metadata and
omitted tensor identities, complete per-stage inventories, resource records,
preflight bindings and job order. The first draft of the new reviewer omitted the
canonical JSON trailing newline and failed digest reconstruction; correcting the
serialization implementation resolved that local verifier error. No evidence was
edited. Focused rejection tests cover the unchanged loss boundary, state drift
despite passing loss, missing replay, altered input/schedule, nonfinite metrics,
digest tampering and a passing legacy control that must remain inconclusive.

The accepted result is in [RETURN_VERIFICATION.json](RETURN_VERIFICATION.json).
[Review validation](return-validation.json) records actual local checks. Historical
training code, diagnostic execution code/plan/bundle, Phase 15/16/17 evidence,
dependency lock and acceptance gates remain unchanged.

## Prospective decision

Recommend the tested combination: deterministic algorithms with errors enabled,
cuDNN determinism, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Torch initialization,
while preserving all other measured controls. Bind that policy to all future
checkpoints and restores; avoid monitored ledger replacement by an append-only
supervisor. Neither change is applied retrospectively to Phase 17.

The [Phase 17.1 proposal](../phase-17-1/PROPOSAL.md) is prespecified planning for a
fresh random-initialization attempt. Full training, Phase 18 and publication remain
unauthorized. Stop for separate owner authorization.
