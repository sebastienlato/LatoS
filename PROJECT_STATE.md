# Project state

Updated: 2026-09-25.

## Authoritative disposition and current gate

**The final seed-161 subset confirmation is verified and closed. E4 fails eight
fixed development gates. The prospective stopping rule is met: the fixed-subset
exposure direction is STOPPED.** Do not propose additional passes, another seed or
another exposure-only experiment on this subset, regardless of an improving trajectory.
No Base Model 2.0 is accepted. No final scoring, Phase 18 or publication is authorized.

Return to owner/master-planning review of data/training design. The current authority
covers evidence closure and analysis only: no new experiment implementation, data
acquisition or execution. Preserve all earlier negative results and unaccepted weights.
The earlier subset diagnostic remains closed; Phase 17.1 remains execution-complete
but development-quality-failed (seven gates; BPB 9.54% worse), not rehabilitated by
these studies. No final score exists or is inferred.

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

## Verified small-subset closure and next decision

[Results and recommendation](experiments/phase-17-1/small-subset/RESULTS.md),
[derived verification](experiments/phase-17-1/small-subset/results.json), and
[separate review](experiments/phase-17-1/small-subset/RESULTS_REVIEW.md).

Return ZIP `db0092d459051e5ddf613ebac09d6ffa8b19a32804f629608d7720cd37a50acb`
verifies: 1,667 payloads plus manifest, 17 native-source snapshots, 1,910 serious
updates / 9,715,075 targets, 52 separate conformance updates, eight checkpoint
boundaries, four included finite float32 models and all saved comparisons. Forty-three
omitted tensor/cache/bytecode records remain inventory identities, not rehashed Mac
payloads. The original failure and 1,703-file preservation inventory remain intact;
no original completed record was fabricated. Correction added zero optimization/inference.

Cumulative charged Windows time is **1,284.933927 seconds (21m24.934s)**, including
**20.933927 seconds cumulative packaging**, under both original limits. The original
1,272.764974 seconds are charged upward to 1,273; correction adds 11.933927 seconds.
The packager's erroneous cross-arm base-gate call is preserved and corrected only
in evidence handling. Original training/scoring/configurations/gates remain unchanged.

A1/B1/B2/C2 BPB: 2.554079 / 2.535825 / 2.472051 / 2.469469. Repetition:
79.28% / 71.68% / 63.29% / 63.36%. Every endpoint fails eight mandatory development
gates. A1→B2 improves both books and repetition (18/24 prompts improve), supporting
an early relative exposure/schedule benefit. C2 is numerically close to B2 on observed quality
but trains 39.2% faster. ARC intervals include zero; no established accuracy gain,
domain-causality claim, useful instruction following or accepted base follows.

## Final confirmation closure — fixed-subset direction stopped

[Authoritative closure](experiments/phase-17-1/subset-confirmation/CLOSURE.md),
[reconstructed evidence](experiments/phase-17-1/subset-confirmation/closure-verification.json),
[separate review](experiments/phase-17-1/subset-confirmation/CLOSURE_REVIEW.md), and
[owner/master-planning analysis](experiments/phase-17-1/subset-confirmation/MASTER_PLANNING.md).

Return `81bba1b8200dc382ef241afeea6fe110eae517a0226bd59e79c21791d0aa4722` verifies:
1,421 payloads plus manifest, 15 native-source snapshots, 2,292 serious / 2,344
physical updates, 11,658,090 target executions, eight checkpoint boundaries and
three included model files with 90 exact finite float32 tensor fingerprints.
Thirty-seven omitted tensors remain inventory-bound on Windows, not rehashed Mac
bytes. No serious retry/replay; original seed/data/model/schedule/endpoints intact.

Guardian total **1,011.561073 seconds (16m51.561s)**, exit zero, no deadline termination.
All stage caps and the 30-minute complete-study cap pass, including packaging.
D2/E2/E4 matched BPB: **2.466165 / 2.410365 / 2.417791**; repetition:
**65.21% / 56.62% / 55.10%**; ARC-Easy: **159 / 162 / 155 of 570**.
All three endpoints fail eight fixed development gates. E4 is 29.09% worse than
Phase 5 on matched BPB. No alternate checkpoint is promoted; no final scoring occurred.

At equal two-pass exposure, the longer schedule improves observed book/repetition
metrics. E2→E4 then lowers encyclopedia monitoring loss but worsens both book BPBs;
ARC-Easy falls seven answers with a paired interval including zero. Mean repetition
improves slightly, while fourteen of 24 prompts worsen. This does not identify a
unique failure cause, but it does not justify overriding the prospective stop.

Next decision is a materially revised data/training design brief or broader stop/
rethink, **not another subset exposure run**. Domain coverage/quality is a priority
hypothesis, not a proven cause. Keep schedule/objective accounting explicit and
model scale a measured tradeoff; the efficient 21.24M model is not an accepted base
or proved optimal architecture. Any future experiment needs separate authorization,
unchanged references/gates and an inclusive **45-minute maximum Windows GPU/computer
budget**, preferably targeting substantially less, with zero paid resources.
No new dataset, training recipe, executable experiment or handoff was created here.

**Stop for owner/master-planning review.** Mac work used saved metadata/rows and
CPU tensor-storage inspection only: zero optimization, model construction/forward,
generation or new model scoring. External contextual notes remain private.

## Publication and local continuity

Phase 17.1 closure `c2277e7104e92408581db41cd0f9852c8f7cb2ee` was published;
local HEAD and live remote `main` matched it at review entry. This supersedes the
closure snapshot's old pending-publication wording. No historical result was changed.

This faithful study closure is local, with its reviewed commit and evidence identities
recorded in the task checkpoint and private continuity. Existing remote:
`https://github.com/sebastienlato/LatoS.git`, branch `main`. No remote write occurred;
no publication, tag, release, weights or assets are authorized or proposed for action
in this task. The current checkpoint is owner/master-planning review, not a push request.
