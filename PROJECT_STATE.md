# Project state

Updated: 2026-09-25.

## Authoritative disposition and current gate

**Phase 17.1 completed its authorized fresh execution but failed fixed development
quality acceptance. The bounded experiment is closed; no Base Model 2.0 is accepted.**
Final acceptance is prohibited for this candidate. Phase 18 remains blocked.
The owner reports that the authorized small-subset diagnostic stopped on a packaging
exception after 21m13s, with no verified return/receipt/completed record. Actual
Windows artifacts and A/B/C quality are **not yet independently reviewed on Mac**.
The owner authorized investigation and, if valid, a zero-update packaging correction.
Mac correction preparation is complete; **stop at its Windows checkpoint**. No
training, new model scoring, Phase 18 or publication is authorized.
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

## Active checkpoint — packaging-only correction

[Diagnosis, limits and corrective instructions](experiments/phase-17-1/small-subset/packaging-correction/README.md),
[validation](experiments/phase-17-1/small-subset/packaging-correction/validation.json).

The prepared packager incorrectly called the base-acceptance function for A1→B1
and other descriptive contrasts. Its correct Phase 5 identity guard rejects A1.
This defect is reproduced using historical saved outputs with a valid reference;
it does not establish missing Phase 5 data or any A/B/C quality result. The new
helper retains all four Phase 5 acceptance comparisons and uses the existing paired
bootstrap directly for descriptive contrasts. No acceptance gate changes.

Conditional Windows admission requires the preserved traceback to identify this
exact defect, complete saved jobs/development outputs, original source/cache identity,
matching existing verification, and sufficient time. A different failure or missing/
invalid reference stops without training, inference, data repair or new evaluation.
The original controller, scorer, proposal, study files and failed packaging evidence
remain unchanged. New outputs use a separate permanent `small-subset-packaging-fix1`
directory; before/after inventories must match. Never overwrite `run/verification.json`
or manufacture an original `completed.json`.

**Carry the original budget:** debit D=max(1273, ceil(original guardian seconds)).
Total remaining is at most 1427 seconds. Also debit failed packaging time
P=ceil(D−original outcome.elapsed_before_package). Corrective cap is
min(2700−D, 300−P), including all startup/checking/packaging/readback/closure.
No timer reset, GPU work, optimizer updates, scoring, paid resources or second attempt.
The existing pinned process-tree guardian enforces the additional interval.

Only one corrective script plus receipt/kickoff is transferred; no weights, data,
runtime or training worker. Exact local commit/artifact identities are in the task
checkpoint and private continuity. Regression tests use saved historical rows and
scripted artifacts, never this unreceived Windows study's scores.

**Stop for the authorized Windows corrective handoff and evidence return.** Its
actual prerequisites, duration and success remain pending. The rejected long study
stays prohibited; the original A/B/C design is not reopened. No accepted base or
Phase 18 follows, regardless of later diagnostic results. No publication authorized.

## Publication and local continuity

Phase 17.1 closure `c2277e7104e92408581db41cd0f9852c8f7cb2ee` was published;
local HEAD and live remote `main` matched it at review entry. This supersedes the
closure snapshot's old pending-publication wording. No historical result was changed.

This implementation/handoff is local, with its reviewed commit and exact artifact
identities recorded in the task checkpoint and private continuity. Existing remote:
`https://github.com/sebastienlato/LatoS.git`, branch `main`. No remote write occurred;
no publication, tag, release, weights or assets are authorized or proposed for action
in this task. The current checkpoint is Windows execution preparation, not a push request.
