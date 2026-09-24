# Phase 17 bounded experiment closure — interrupted/failed

**The authorized Phase 17 experiment is closed with a failed corrective recovery
gate. The Phase 17 learned-quality exit gate is unmet and unmeasured. No LatoS Base
Model 2.0 is accepted, and Phase 18 remains blocked.** This records the outcome; it
is not successful roadmap completion. Publication remains subject to explicit approval.

The recovery return ZIP matches the owner's SHA-256
`a0d521a0c9c4e7bf12d25b8bca19cc389b4ddd77928d73b3f2221cb4c0b0e0f9`.
[Independent reconstruction](closure-verification.json) verifies 3,847 included
files, two executed runtime snapshots against training commit
`0fa94ad580110fd2dc7aaa2aa3560950abe73a02`, and administrative bundle contents against
review commit `1436ec27a9ef42c6a72c7625ba9b4414cdf82daf`. All 4,043 records of the
first return, including declared omitted files, retain the same identities. Mac
rehashed included bytes and retained inventory bindings for tensors still on Windows;
it did not execute returned scripts or reconstruct absent optimizer tensors.

## Six distinct outcomes

| Evidence | Authoritative interpretation |
| --- | --- |
| Original 7,485 updates / 37,811,418 targets / 119,748 windows executed | Successful optimizer execution of the original planned pass; not a saved accepted model |
| Supervisor ledger replacement failed before final checkpoint | Original final model state was lost; checkpoint 7,000 was the last retained recovery point |
| Bounded step-7,000 replay executed 485 updates / 2,426,653 repeated targets | First permitted resumption consumed; counters, schedule and two-microbatch final update match |
| Replay-equivalence tolerance `1e-5` failed | Corrective continuation failed closed; remaining allowance/time does not authorize another retry |
| Recovered final checkpoint was produced | Valid serialized experimental artifact, currently unaccepted; not a substitute for the missing original final state |
| Fixed paired development/final acceptance never ran | Learned quality is unknown; no acceptance score, improvement or failure on those quality metrics can be claimed |

Logical exposure remains **37,811,418 targets**. Physical work across the original
run and replay is **40,238,071 target executions / 7,970 update executions**. The
2,426,653 repeated targets are not new unique data or an additional epoch in the
recovered model's logical trajectory. All prior runs, ledgers and failures remain.

## Frozen corrective gate

The nine administrative Windows checks passed without skips, including the actual
file-sharing fixture. The new append-only supervisor completed its worker dispatch.
The journal's final event is `worker-complete`, exit zero, followed by a separate
`recover-failure.json`. There is no successful corrective completion receipt or
terminal administrative `complete` event. The worker's `training-result.json`
with `status: complete` describes its own work and cannot override that failure.

Independent comparison of every replayed update gives:

- **351 of 485** absolute loss differences exceed the unchanged **`1e-5`** bound.
- First excess: **step 7,125**, original 3.994038991161755, replay 3.993776696707521,
  absolute difference **0.00026229445423409103**.
- Largest excess: **step 7,255**, original 3.9539700086654386, replay 3.954679988397522,
  absolute difference **0.0007099797320835322** (about 71 times the bound).
- All logged counter/schedule comparisons match; final accumulation is two
  microbatches. These matching properties do not waive the numerical mismatch.

The [original interruption review](INTERRUPTION_REVIEW.md) made the corrective
replay conditional on that tolerance. The [frozen plan](../../configs/training2/phase17-plan.json)
requires recovery mismatches to fail closed. The allowance of at most two
same-attempt resumptions applies to qualifying external interruptions; it is not
an entitlement to reroll a completed recovery that failed its correctness gate.
No second replay, fresh attempt, alternative checkpoint or paired scoring is
permitted under the existing contract. No threshold is rounded, relaxed or replaced.

The cause of the replay divergence is not established by the available scalar
logs. Matching source/runtime/counters and passing tiny CUDA controls do not prove
full-scale numerical equivalence. Do not assert that BF16 rounding is harmless or
attribute a specific kernel, checkpoint defect or race without further evidence.
The original PowerShell monitor's contribution to the first I/O failure also remains
plausible but unproven. The later disposable sharing fixture established a possible
mechanism, not the original production lock owner.

## Preserved, unaccepted recovered model

Model SHA-256: `d37141054b26f777403e855ee5903f45a8b5d22205d48d6ab37f9e73fc39f4e6`.
The 136,355,352-byte tensor file is now verified and retained on Mac as well as in
the Windows originals. CPU model-only loading confirms the frozen configuration,
34,087,424 parameters, expected tensor names/shapes and finite float32 values. This
was artifact validation only: no forward pass, generation, optimization or scoring.
Final/historical optimizer tensors remain on Windows with their recorded hashes;
the return is not their full backup. Preserve all originals until a verified backup
exists. No model bytes or acquired text enter Git or a release.

Final **training-monitoring** validation loss is **3.964916790444685** and perplexity
**52.715882741983336**, over 804,337 targets / 2,552 windows of the new development
split. These are not fixed Phase 14 paired development or final acceptance scores.
No cross-tokenizer perplexity comparison or learned-quality promotion follows.
See the [model-card addendum](../../docs/MODEL_CARD.md).

## Resources and next decision

Original time is conservatively charged at 1,562 seconds, plus 120.9380624 seconds
for recovery: **1,682.9380624 seconds** total. Evaluation used zero; its 3,600-second
reserve is untouched. Recovery peak reserved GPU memory was 1,193,279,488 bytes;
host peak 2,733,400,064 bytes. Peak controller-charged artifact bytes were
4,985,525,605. Recorded resource limits passed. Spare resources do not lift a failed
correctness gate or authorize additional work.

Roadmap 2.0 requires retaining failures, blocking dependent phases and recording a
bounded corrective proposal. [Next decision](NEXT_DECISION.md) supplies a proposal,
not an executable handoff or authorization. A new prospective execution plan and
explicit owner authorization are needed to reopen improvement work. Preserve the
existing learned-quality gates; changing required roadmap goals/gates/non-goals
would separately require an owner-approved roadmap amendment. No such change is
made here. Publication of this closure alone authorizes no diagnostic run, new
attempt, model promotion, Phase 18 or later release claim.
