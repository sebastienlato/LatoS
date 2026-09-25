# Project state

Updated: 2026-09-25.

## Authoritative disposition and current gate

**Phase 17.1 completed its authorized fresh execution but failed fixed development
quality acceptance. The bounded experiment is closed; no Base Model 2.0 is accepted.**
Final acceptance is prohibited for this candidate. Phase 18 remains blocked.
The owner explicitly authorized implementation, validation and Windows execution
preparation for the reviewed small-subset diagnostic. Mac implementation and
zero-update validation are complete; **stop at the Windows handoff checkpoint**.
Actual Windows execution/results remain pending. No Mac GPU training, full Base
Model attempt, final scoring, Phase 18 or publication is authorized.
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

## Active checkpoint — small-subset Windows handoff

The owner rejected the former 6.6-hour/four-pass study; it remains prohibited.
The [authorized small-subset design](experiments/phase-17-1/SMALL_SUBSET_PROPOSAL.md)
is unchanged: A 34.09M one pass, B 34.09M two passes with fixed B1 observation,
C 21.24M two passes. Fixed existing Phase 16 subset: 1,943,015 targets / 6,105
windows. Fresh seed 160; 1,910 serious updates / 9,715,075 target executions;
at most 52 existing tiny CUDA conformance updates, no retries or resumptions.

[Windows instructions](experiments/phase-17-1/small-subset/WINDOWS.md),
[separate review](experiments/phase-17-1/small-subset/REVIEW.md),
[validation](experiments/phase-17-1/small-subset/validation.json).
Standalone controller, pass-tail adapter, exact diagnostic checkpoint readback,
source/cache binding, saved-evidence verifier and timed return packaging are ready.
Native source, historical plans/results, fixed evaluation gates/reference, dependency
lock and proposal bytes are unchanged. The subset was reconstructed on Mac with
exact existing identities; no new selection, tokenizer or acquired data.

**One absolute 45-minute Windows wall clock**, including verification/extraction,
preparation, all arms, all evaluation and packaging. Target 33 minutes; stage caps
180/1,440/480/300 seconds. Controller cutoff 2,690 s and independent Windows Job
Object guardian cutoff 2,695 s reserve closure time before 2,700 s. All descendants
are contained; a real zero-update guardian test must pass inside preparation.
Missing controls, corruption, interruption or budget exhaustion fails closed.
Zero paid compute; no later packaging run or separate per-arm clock.

Mac validation uses only CPU scripted/accounting/storage checks; zero optimizer
updates, model forward/scoring calls or GPU training. Tiny random CPU models are
used only for checkpoint storage validation. Windows Job behavior/CUDA conformance,
actual C training, measured study time and learned results are **pending**, not
validated on Mac. Fixed A1/B1/B2/C2 development scores will be diagnostic only;
Phase 5 remains the primary acceptance reference. No promotion or final access.

**Stop here for the authorized manual Windows handoff.** Exact local transfer files,
checksums and prepared kickoff are recorded in the task checkpoint and private
continuity. No further work/phase follows from successful execution or spare budget.
Return the existing evidence for Mac/owner review. Keep context concise; no fan-out,
full historical optimizer test suite or repeated reconstruction without need.

## Publication and local continuity

Phase 17.1 closure `c2277e7104e92408581db41cd0f9852c8f7cb2ee` was published;
local HEAD and live remote `main` matched it at review entry. This supersedes the
closure snapshot's old pending-publication wording. No historical result was changed.

This implementation/handoff is local, with its reviewed commit and exact artifact
identities recorded in the task checkpoint and private continuity. Existing remote:
`https://github.com/sebastienlato/LatoS.git`, branch `main`. No remote write occurred;
no publication, tag, release, weights or assets are authorized or proposed for action
in this task. The current checkpoint is Windows execution preparation, not a push request.
