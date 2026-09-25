# Project state

Updated: 2026-09-25.

## Authoritative disposition and current gate

**Phase 17.1 completed its authorized fresh execution but failed fixed development
quality acceptance. The bounded experiment is closed; no Base Model 2.0 is accepted.**
Final acceptance is prohibited for this candidate. Phase 18 remains blocked.
The owner explicitly authorized the zero-training exposure/domain design review
after publication. That review is now complete, with one prospective study for
owner/master-planning review. **No experiment implementation, training, retry,
final scoring, new phase or publication is authorized.**
Publication approval, if later granted, would authorize that exact publication only;
it would not reopen improvement work or lift this stop gate.

Published Phase 17 remains permanently interrupted/failed at
`09c1935b085218fbbc9ea88703d8ba12e63c516e`; its recovered model stays unaccepted.
The earlier positive synthetic diagnostic is separate mechanics evidence. Neither
historical result is rewritten by Phase 17.1.

## Verified Phase 17.1 evidence

[Closure](experiments/phase-17-1/CLOSURE.md),
[independent reconstruction](experiments/phase-17-1/closure-verification.json),
[separate review](experiments/phase-17-1/CLOSURE_REVIEW.md), and
[local validation](experiments/phase-17-1/closure-validation.json).

Executed controller: `7cb5afd54d33fda4d18b84c5f0645d661b76e002`.
Returned ZIP SHA-256:
`66c2ddee8e85a289998461deb0e86945022b9e368a142f86ca9a68ba3a96e358`.
All 1,366 included files, twelve native-source snapshots and the fixed development
comparison verify. Eighty-one omitted tensor/cache records are inventory identities
only; their bytes remain on Windows and were not replayed on Mac.

Windows CPU checks: 35/35 in both runtimes, no skips/failures or optimizer updates.
All four CUDA preflight pairs passed (52 tiny updates). Fresh seed-160 serious run:
7,485 updates / 119,748 windows / 37,811,418 targets; no serious replay/interruption,
final accumulation two, nine retained checkpoints. Every serious input signature,
state digest, counter and exact Windows schedule binding reconstructs. Total
supervised active time 4,721.880432 seconds; recorded resource bounds pass.

New final candidate SHA-256:
`969034537d6b9740e217b436d5bfffbab51db3bbb3d9be79801a71f2ae336331`.
Its included 136,355,352-byte finite float32 tensors match the final training
fingerprints and 34,087,424-parameter count. It is retained on Mac and Windows as
an **unaccepted experimental artifact**, not promoted to a baseline or release.

## Quality outcome

Seven mandatory development gates fail: matched BPB ratio 1.0954109564 (required
<=0.90); book ratios 1.1923261860 / 1.0615539175 (<=1.02); ARC-Easy 28.9474%
(>=35%) with +1.4035-point gain (>=5); repetition 0.4988729779 (<=0.25) and
increase 0.1106861299 (<=0.10). Other numeric gate passes do not offset failures.

Matched BPB 2.0516431115 is 9.54% worse than the paired historical base. New-corpus
monitoring loss falls 9.8115374518 to 3.9639621581, showing learning under that
objective without the required generalization. Inference mechanics pass (6.1934 ms
first token, 148.5148 tokens/s under the fixed synthetic-prompt measurement).
No final scoring/declaration or retry occurred. No final score is inferred.

## Completed design review and next gate

[Exposure/domain review](experiments/phase-17-1/EXPOSURE_DOMAIN_REVIEW.md),
[derived evidence](experiments/phase-17-1/exposure-domain-analysis.json), and
[separate review/validation](experiments/phase-17-1/EXPOSURE_DOMAIN_REVIEW_CHECKS.md).

Recommendation: one prospective unchanged-model/tokenizer/data exposure study:
fresh one-pass control and fresh four-pass candidate, with a prespecified candidate
first-pass schedule diagnostic. Two trajectories, one seed, no checkpoint search.
The domain hypothesis remains untested; larger models and a data-mix intervention
are not justified as the immediate next experiment. No study code/configuration,
new dataset or execution handoff has been implemented.

Prospective cost: about 6.6 hours on the existing RTX 4070 SUPER; hard 11-hour active
ceiling, 32 GiB new-artifact limit, zero paid services. Proposed serious exposure
37,425 updates / 189,057,090 targets, plus at most 52 existing tiny CUDA preflight
updates; no serious replay. These are proposal limits, **not authorization**.
A failed candidate ends this unchanged-corpus exposure direction for master planning.
All fixed acceptance references/gates remain unchanged. Even a future development
pass would require separate selection/contamination and owner review before final
access. Phase 18 remains blocked.

**Stop for owner/master-planning review before any proposed study implementation.**
The earlier review-start gate is satisfied by explicit owner authorization in this
session; the experiment-start and publication gates remain closed.

Current review read saved metrics and metadata only: zero optimizer updates, model
forward passes, new generation/scoring, new data acquisition or paid resources.
The 7,485-row trace hash and arithmetic were checked; the prior 1,366-file closure
reconstruction and tensor inspection are historical evidence, not rerun here.
No CUDA/CI/physical Linux execution. Private planning remains ignored.

## Publication and local continuity

Phase 17.1 closure `c2277e7104e92408581db41cd0f9852c8f7cb2ee` was published;
local HEAD and live remote `main` matched it at review entry. This supersedes the
closure snapshot's old pending-publication wording. No historical result was changed.

This design review is local documentation/evidence only, with a reviewed local
commit recorded in the task checkpoint and private continuity. Existing remote:
`https://github.com/sebastienlato/LatoS.git`, branch `main`. No remote write occurred;
no publication, tag, release, weights or assets are authorized or proposed for action
in this task. Owner review concerns the prospective study, not a push checkpoint.
