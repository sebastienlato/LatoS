# Phase 16 Mac review — external checkpoint only

A separate local review followed the initial implementation and focused/full test
runs. No delegated or external reviewer is claimed. CUDA behavior remains pending
actual Windows execution; code inspection and CPU tests cannot certify BF16.

Findings addressed before the external handoff:

- Device completion errors could leave `ready=true` after a CUDA synchronization
  failure. Mark an update checkpointable only after synchronization succeeds.
  A forced completion-failure regression verifies that saving remains blocked.
- Tiny recovery tests alone would not exercise real candidate checkpoint costs.
  The probe now saves/reloads at its midpoint and compares development loss before
  continuing. Full checkpoint readback duplication is included in VRAM peaks.
- A successful result without imported source bytes weakens later reproduction.
  Every worker and suite now retain a source snapshot plus portable relative-path
  hashes. Timeouts remain incomplete even if some files exist; no automatic retry.
- Device failures initially retained only exception text. Retain available host
  and CUDA peak metrics too. Unavailable device measurements remain unavailable.
- The new experiment directory needed explicit inclusion in source distributions.
  Add it to the reviewed package allowlist; wheel training modules package normally.
- Context boundaries were reviewed independently: adjacent chunks rejoin verbatim,
  numeric paragraph/chunk order is enforced, gaps receive separators, overlap-one
  windows count transitions once, and documents never share attention. Tests compare
  against an independently constructed expected transition stream and perturb a
  second document/future token to verify isolation and causality.

The first formatting checks found long lines/import order and a closure lint issue;
these were fixed before running tests. No numerical tolerance, scientific bound or
quality gate was lowered. The actual data probe required no document exclusions.
All nonfinite update attempts in tests are deliberate original synthetic fixtures.

Review limits: CUDA-specific autocast, driver behavior, Windows working-set API,
actual GPU throughput/memory/headroom and candidate learned loss remain unexecuted
on this Mac. The first external protocol fixes batch/context and varies depth;
it is not an exhaustive architecture/performance search. No activation
checkpointing or FP16 was added without measured need. Any failed external gate
must be preserved and reviewed before a bounded revision. Phase 16 stays open.
