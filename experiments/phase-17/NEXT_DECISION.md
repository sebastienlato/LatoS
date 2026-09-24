# After the failed Phase 17 experiment — proposal only

The current contract is exhausted by a terminal recovery mismatch, not by resource
usage. Its numerical gate remains failed and its learned-quality exit gate remains
unmeasured/unmet. One unused interruption allowance and unused evaluation time are
not permission to retry, score an unaccepted checkpoint or proceed to Assistant 2.0.
No executable continuation is prepared from these allowances.

[Roadmap 2.0](../../ROADMAP.md) calls for a bounded corrective proposal when a gate
fails. The owner can stop with the preserved failure or explicitly authorize a
separate prospective correction. The following is a proposed scope, not authorization:

1. Investigate recovery mechanics separately from learned quality. Before execution,
   freeze exact source/runtime, synthetic inputs, checkpoints, seeds, comparisons,
   tolerances and stop conditions. Use the existing RTX 4070 SUPER and zero new paid
   resources. Proposed ceiling: 60 active GPU minutes, at most two predeclared
   comparison pairs, each at most 1,024 uninterrupted updates versus the same 1,024
   updates with a checkpoint/restore (at most 4,096 physical updates total), and
   6 GiB of new artifacts. Include a replay horizon at least as long as the observed
   485-update tail, the selected model scale and variable-length batching; tiny
   short controls alone are insufficient. Do not read
   held-out acceptance data, tune from learned-quality scores or change the existing
   `1e-5` criterion for this failed experiment. Any execution correction is original,
   separately reviewed and versioned; no live mutation of the preserved run.
2. If that investigation supports a reproducible correction, prepare a new serious
   pretraining proposal with explicit attempts, source/data/model identities,
   schedule, exposure, artifact/time bounds, recovery criteria and paired evaluation
   policy fixed before launch. Starting it needs separate explicit owner approval;
   passing the diagnostic does not automatically grant another serious attempt.
   Do not silently reclassify the present recovered model as the new baseline.
3. Only a valid future base that actually passes the existing Phase 14 development
   and gated final acceptance requirements can unblock Phase 18. Maintain the
   historical base and current failed experiment as distinct preserved evidence.

These bounds are a reviewable proposal, not added budget for the closed attempt.
No investigation, source correction, extra optimization or evaluation was performed
by writing this document. The specific cause of divergence remains open. More work
cannot be justified merely by the small absolute size of a failed precommitted bound.

A bounded execution-plan revision does not require changing the roadmap destination
or lowering quality gates. If the owner instead proposes changing required roadmap
goals, quality gates or explicit non-goals, the roadmap requires an explicit approved
amendment, with this failed result retained. No amendment is made or assumed here.
A push approval for the failed-experiment closure grants publication only; after
publication, stop until the owner supplies the next authorized scope.
