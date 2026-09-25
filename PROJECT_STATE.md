# Project state

Updated: 2026-09-25.

## Authoritative disposition and current gate

**Phase 17.1 completed its authorized fresh execution but failed fixed development
quality acceptance. The bounded experiment is closed; no Base Model 2.0 is accepted.**
Final acceptance is prohibited for this candidate. Phase 18 remains blocked.
The owner authorized evidence review, closure and a prospective next-decision proposal
only. **No further training, retry, final scoring, new phase or publication is authorized.**
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

## Next action and publication state

**Stop for owner/master-planning direction.** The concise
[next-decision proposal](experiments/phase-17-1/NEXT_DECISION.md) recommends a separately
authorized, zero-training design review of exposure and domain coverage, not an
immediate retry. These are hypotheses, not established causes. Roadmap 2.0 goals
and all numeric gates remain unchanged; no future experiment is selected or launched.

Closure work uses saved records and final tensor storage inspection only: no new
optimizer update, model forward pass, generation, final payload scoring, CUDA replay,
CI or physical Linux execution. Historical source/contracts/artifacts are preserved.
The complete Windows tensor inventory is not backed up by the compact return.

The reviewed closure commit is local on `main`; its exact hash is recorded with
the checkpoint and local continuity. Existing remote is
`https://github.com/sebastienlato/LatoS.git`, destination `main`, no tag/release/weights
proposed. Publication needs explicit approval. No remote write has occurred.
