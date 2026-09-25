# Project state

Updated: 2026-09-24.

## Current authorized scope

The bounded recovery-mechanics diagnostic has completed on the RTX 4070 SUPER and
passed independent Mac review of its **prespecified synthetic acceptance rule**.
The owner authorized review and preparation of a fresh Phase 17.1 proposal only.
**Full Phase 17.1 training, Phase 18 and all publication remain unauthorized.**

Published Phase 17 is permanently interrupted/failed at
`09c1935b085218fbbc9ea88703d8ba12e63c516e`: recovery equivalence failed, recovered
model unaccepted, fixed development/final acceptance absent, exit gate unmet. Neither
the diagnostic nor a future attempt repairs or replaces that historical result.

## Completed diagnostic and reviewed evidence

Measured diagnostic source: `0e5b55f516b6dd53a48f53928df2616872ca5f82`.
Return ZIP SHA-256:
`340b57d7f201436e47b575b7a792cb208cb74dfe47cd5f835322d313304c47d9`.
[Independent review](experiments/recovery-diagnostic/RETURN_REVIEW.md),
[verification](experiments/recovery-diagnostic/RETURN_VERIFICATION.json) and
[local checks](experiments/recovery-diagnostic/return-validation.json).

All 318 included files and four 62-file executed source snapshots verify. Fourteen
omitted tensor/cache files remain inventory-bound on Windows, not rehashed or
numerically replayed on Mac. Ten Windows CPU checks passed, zero skips/failures and
zero optimizer updates. Four CUDA jobs completed: two 997-update references, each
sharing its checkpoint at 512 with a separate-process 485-update replay. Total
2,964 physical updates / 14,897,204 synthetic targets, no retries in the journal.

Legacy full-state divergence starts at 674; first loss excess at 675, 303 excesses,
maximum 0.0004386934420383959. Deterministic continuation has exact full-state and
loss equality across all 485 updates; maximum loss difference zero. Fresh initial
states, checkpoint roundtrips, restoration, input signatures and Windows schedules
match. No loss tolerance or exact-state gate was relaxed.

Supervised time is 1,458.462660 seconds; all recorded resource bounds pass. The
owner-reported 1,572-second total includes later administrative work. An original
Mac verifier check failed on cross-platform cosine rounding (up to three output
ULPs). It remains unchanged. The disclosed independent review requires exact
same-Windows schedules and separately reconstructs that arithmetic discrepancy;
it does not reinterpret the failed historical CUDA loss gate. No held-out score,
exact historical kernel cause, general reproducibility or language quality follows.

## Phase 17.1 proposal checkpoint

[Reviewed proposal](experiments/phase-17-1/PROPOSAL.md),
[prespecified contract](experiments/phase-17-1/plan.json) and
[preparation handoff](experiments/phase-17-1/HANDOFF.md) are local only.

Propose one fresh seed-160, 34,087,424-parameter attempt on accepted Phase 15 data
and the Phase 16 recipe: 7,485 logical updates / 37,811,418 targets, CUDA BF16,
batch two × accumulation eight with final accumulation two. Apply the tested
deterministic algorithms/cuDNN/workspace combination prospectively. Add immutable
checkpoint policy bindings and an append-only supervisor; preserve the native core.
Keep fixed quality gates, two-hour training/preflight/recovery and one-hour cumulative
evaluation limits, 20 GiB new-artifact cap, and zero new paid resources.

This is a scientific/engineering specification, **not an implemented full-training
controller or executable Windows bundle**. Separate explicit owner authorization
is required before implementing the proposed execution controller and conducting
its bounded preflight/one serious run. After authorization, review and freeze that
source/transfer before execution; material plan changes need renewed approval.
No failed Phase 17, probe or diagnostic weights may initialize the attempt.

The proposed tiny adapter preflight is fully specified, at most 52 CUDA fixture
updates across checkout/existing wheel; no such updates have run. Full-cache and
source checks use zero optimizer updates. Conditional external-interruption recovery
allows at most two same-attempt resumptions after Mac review, within all cumulative
bounds and exact full-state replay checks. No automatic or numerical-failure retry.

## Next action and publication status

**Stop for separate owner authorization of the reviewed Phase 17.1 proposal.**
After any authorized run, return paired development evidence to this authoritative
Mac task before final access; passing development/resource gates and a reviewed
selection/contamination declaration are mandatory. Phase 18 remains blocked.

The owner manually transfers checked bundles to a separate fresh Windows Work
session; no Windows remote executor is connected to this Mac. The returned evidence
and original diagnostic handoff remain preserved locally; tensor originals remain
on Windows. No new CUDA, CI or physical Linux execution occurred on Mac.

Current work is a local diagnostic-review/proposal checkpoint on `main`, following
the unpublished diagnostic preparation commit. Intended eventual remote is existing
`origin` (`https://github.com/sebastienlato/LatoS.git`), branch `main`; no tag proposed.
Exact local commit is reported with the checkpoint and retained in local continuity.
No push, tag, release, assets or other remote write is authorized or performed.
