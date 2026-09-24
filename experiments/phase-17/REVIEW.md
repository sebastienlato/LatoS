# Phase 17 Mac review — execution handoff only

A distinct review pass followed the first implementation and complete Mac test
suite. It was performed within this task, without delegated/external reviewers.
This review cannot establish CUDA correctness, training completion or learned quality.

## Findings addressed

- Generic accumulation would cross the final epoch boundary. Add an explicit
  single-pass mode with an exact ceil-divided update budget, no sampler wrap and
  the actual tail target denominator. Independent summed-cross-entropy gradients
  match the partial update; odd tails and one-update datasets are covered.
- Legacy recovery derives cursor from full accumulation. The distinct v3 checkpoint
  clamps microbatch count to the one-pass total and verifies the seed-derived order
  and RNG, exact target/window counters and sampling policy. Initial/intermediate/
  final restore paths are tested; old checkpoint formats retain their default mode.
- Finite weights alone would miss a corrupted optimizer moment. Single-pass updates
  check all optimizer tensors and leave failed updates non-checkpointable. A forced
  post-step infinite-moment test verifies the failure boundary.
- A self-reported cache checksum is insufficient. Recompute the semantic sequence
  hash from actual read-only little-endian data/offsets, validate IDs, lengths and
  target counts against the independent Phase 16 accounting. Tampering with tokens
  and updating the cache's own hash still fails the frozen sequence hash.
- An unbounded worker or reset ledger could exceed the approved attempt budget.
  The supervisor preserves cumulative active time, enforces subprocess stop limits,
  forbids terminal failure retries, retains rolled-back target counts, and allows
  at most two documented external-interruption resumptions. OS-held supervisor and
  worker locks refuse overlap, including an orphaned worker. A real sleeping child
  is killed in the hard-timeout test; its exhausted ledger remains.
- Initial disk accounting covered only the attempt directory. The supervisor now
  includes validation artifacts and conservatively reserves two uncompressed
  transfers to cover source/inputs and the ZIP. Runtime environments are separately
  reported, as documented; retained cache/checkpoint/failure bytes all count.
- The shared CUDA environment helper originally reapplied the 20 GiB free-disk
  start requirement after training had retained its checkpoints. Parameterize only
  that prerequisite; preserve its existing default for Phase 16, enforce 20 GiB
  before the initial Phase 17 attempt and retain the hard artifact cap thereafter.
  A synthetic disk-state regression covers the distinction.
- Initial final-stage entry could consume its attempt before discovering a missing
  declaration. Validate development/resource gates and the unchanged final-access
  declaration before starting the final worker. Tests verify failed development and
  exhausted budgets cannot proceed. Final feedback never enables another attempt.
- Require matching passing checkout/wheel test receipts, bind them to source and
  retain/recheck original JUnit bytes. Both actual CUDA numerical/tail tests must
  execute; a skipped CUDA case cannot satisfy preflight. The receipt remains audit
  evidence, not an authentication boundary or replacement for Work's log review.
- Retain imported source snapshots, immutable checkpoints and complete failure logs.
  Transfer/return utilities use explicit file allowlists, integrity readback and
  safe-path checks. The returned final model is included; omitted optimizer/cache/
  intermediate bytes have identities and remain on Windows, not claimed rehashed.

## Actual attempts and limits

The first focused run failed nine new cases because of a missing test import;
43 other cases passed and two CUDA cases skipped. The import was fixed, then
52 focused cases passed with the two CUDA skips. Later regression runs passed
418 and 421 cases while review added additional checks. Formatting/import-order
findings were fixed. All attempt logs remain ignored locally. An environment setup
command briefly removed the editable package from the local development environment;
it was restored with the unchanged lock before final tests. No dependency version
change or training resulted. Final counts belong in `validation.json`.

Mac full-data construction matches all pinned document/sequence/count identities.
No model was initialized by that accounting. Actual sustained CUDA throughput,
resource use, one-pass convergence and fixed learned-quality outcomes await the
reviewed Windows run. The source must not change silently there. Any source defect
returns to Mac with the failed attempt preserved before a reviewed correction.
No thresholds, frozen plan/model/protocol bytes, historical evidence or reserved
scores were changed. No GitHub write or Phase 18 work occurred.
