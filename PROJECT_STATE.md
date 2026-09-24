# Project state

Updated: 2026-09-24.

## Authoritative disposition

**The bounded Phase 17 experiment is closed as interrupted/failed. Its corrective
replay-equivalence gate failed; its learned-quality exit gate is unmet and unmeasured.**
There is no accepted LatoS Base Model 2.0. **Phase 18 remains blocked.** No further
retry/recovery, paired development scoring, final acceptance or model promotion is
permitted under the existing frozen contract. Closure publication awaits approval.

[Closure](experiments/phase-17/CLOSURE.md),
[independent verification](experiments/phase-17/closure-verification.json) and
[next-decision proposal](experiments/phase-17/NEXT_DECISION.md) are authoritative.
Earlier execution/recovery guides describe completed historical handoffs, not active
instructions to launch more work. Their original evidence remains unchanged.

## Six separate facts

1. Original optimizer execution completed all **7,485 updates / 37,811,418 targets**.
2. The supervisor failed before saving the original final checkpoint; that state
   remains missing. Step 7,000 was the last retained original checkpoint.
3. The reviewed step-7,000 recovery replayed **485 updates / 2,426,653 targets**,
   consuming the first allowed resumption. Counters/schedule/final two microbatches match.
4. **351 replayed updates failed the unchanged `1e-5` loss tolerance**. First excess
   step 7,125; maximum **0.0007099797320835322**, step 7,255. The corrective gate failed.
5. A recovered final model exists and its bytes/finite float32 structure verify.
   It remains **unaccepted**, not the missing original final checkpoint or an accepted base.
6. Fixed paired development/final acceptance **never ran**. The monitoring loss
   3.964917 / perplexity 52.715883 is not an acceptance score. Learned quality is unknown.

Logical pass: **37,811,418 targets**. Physical execution including replay:
**40,238,071 targets / 7,970 update executions**. Repeated work is not new unique data.
Charge **1,682.9380624 active training/recovery seconds**. Evaluation used zero;
its 3,600-second reserve remains unused. Resource bounds passed; neither spare time
nor the unused interruption allowance permits rerunning a failed correctness gate.

## Evidence and preservation

The recovery return SHA-256 is
`a0d521a0c9c4e7bf12d25b8bca19cc389b4ddd77928d73b3f2221cb4c0b0e0f9`.
All 3,847 included files verify; two executed package snapshots match original
training commit `0fa94ad580110fd2dc7aaa2aa3560950abe73a02`. Administrative code binds
to `1436ec27a9ef42c6a72c7625ba9b4414cdf82daf`. Original return records, ledgers,
source and tensors retain their identities. Nine Windows administrative tests passed
without skips. No returned code was executed on Mac.

Unaccepted recovered model SHA-256:
`d37141054b26f777403e855ee5903f45a8b5d22205d48d6ab37f9e73fc39f4e6`.
Its 136,355,352 tensor bytes are retained on Mac and Windows. Mac performed CPU
artifact loading only, with no forward pass, generation, optimization or scoring.
Optimizer/intermediate tensors remain on Windows; their absent bytes were not
rehashed on Mac. Keep originals; compact returns are not full checkpoint backups.

Imported training source, frozen plan/model/evaluation gates and lockfile are
unchanged. The original implementation hash remains
`76cf4a198ee6acde566fdcc70c9420c0cae8ab6cc0d6b4d7e4c9c0cba5c8f09c`.
The replay divergence cause is unestablished; original PowerShell involvement in
the first filesystem failure also remains plausible, unproven. Tiny control passes
do not establish full-scale recovery fidelity. Historical negatives stay preserved.

[Closure validation](experiments/phase-17/closure-validation.json) and
[separate closure review](experiments/phase-17/CLOSURE_REVIEW.md) record actual Mac
checks. No new CUDA, hosted CI or physical Linux execution is claimed.

## Publication and future boundary

Phase 16 remains published at `ce9de1b734a8e23fd54e9d137cf269f484631765`.
All Phase 17 implementation, corrective review and failed-experiment closure commits
remain local pending explicit publication approval. Existing remote is
`https://github.com/sebastienlato/LatoS.git`, destination main only. No tag, release,
model asset, remote backup or PR is proposed; package remains 1.3.0.

Prepare the reviewed closure commit and exact publication proposal. Approval would
publish this failed result and preserved source/evidence, not certify success.
After approved publication and verification, **STOP**. Publication alone does not
authorize another experiment. Reopening improvement work needs explicit owner
approval of a new bounded prospective corrective plan; no such execution is
currently authorized. A future accepted base must still pass unchanged quality
gates before Phase 18. Roadmap 2.0's capability destination remains incomplete.
