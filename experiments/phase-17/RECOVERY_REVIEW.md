# Separate review of the administrative recovery

This pass followed implementation and the first targeted/full Mac checks. It is
an in-task review, not an independent external reviewer or a claim of CUDA recovery.
The review was confined to the administrative correction and evidence integration.

- Confirmed the original worker source/configuration/strict checkpoint loader stays
  byte-identical. Adding a controller outside `src/latos` keeps the implementation
  hash at `76cf4a198ee6acde566fdcc70c9420c0cae8ab6cc0d6b4d7e4c9c0cba5c8f09c`.
  The Windows checkout must remain at the original commit. No state hash is migrated.
- The old `running` ledger is stale and cannot authorize a stock resume. Bind both
  original ledgers by digest, retain the supervisor-failure status, use a separate
  journal and charge the larger attempted elapsed time rounded upward to 1,562 s.
- All original attempt records, including cache/optimizer/model tensor identities,
  are rehashed before and after. Mac independently checks the return's source,
  JUnit, wheel, full-cache identities, exposure and checkpoint metadata; absent
  tensors must still be rehashed and strictly loaded on Windows.
- A result file alone was initially sufficient for starting development, which
  could miss a late journal failure. Require its matching terminal completion event
  and absence of a failure record. The regression rejects an incomplete journal
  and a receipt accompanied by a later failure.
- Windows byte-range locks prevent reopening the locked region through another
  handle, including in the locking process. Read the original active-lock bytes
  through the held handle, and explicitly release the lease before closing. The
  existing nine-test suite checks held-handle verification and lock exclusion.
  [Microsoft LockFile semantics](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-lockfile)
  informed this correction; the Windows execution check remains pending on Mac.
- The append-only journal never replaces either original ledger. A synthetic
  forced-replace failure leaves the journal operational. An actual Windows-only
  disposable-file test will demonstrate reader-sharing refusal of rename; it is
  not evidence establishing the original lock owner or blaming PowerShell.
- A real sleeping child is killed at its test deadline; a failing child does not
  become a success. Cumulative time is preserved across the reviewed stages.
  No automatic retries, second replay, final scoring or Phase 18 path is exposed.
- Replay checks exact counters/schedule and the existing 1e-5 loss recovery tolerance,
  preserving repeated physical work separately from logical one-pass exposure.
  Divergence, failed loading or another execution fault returns to Mac review.
- Reviewed bundle/return handling includes only the needed administrative files,
  original/new evidence and final model if it exists. The original return and all
  Windows checkpoints remain intact. A compact return is not a full tensor backup.

Formatting/import-order findings were fixed. First targeted run: eight passes,
one actual-Windows-only skip. Full Mac checkout and installed-wheel results, final
hashes and preservation results are in `recovery-validation.json`. No serious
optimization, fixed scoring, CUDA execution, new dependency, remote write or
Phase 18 occurred during this review. Recovery remains conditional on the specified
checks on the original Windows machine; absent learned outcomes are not inferred.
