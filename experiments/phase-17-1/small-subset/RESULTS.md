# Small-subset diagnostic — verified results and next decision

2026-09-25. **The three-arm diagnostic and its packaging-only correction are
verified. Every scored experimental endpoint fails eight fixed development gates.
No Base Model 2.0 is accepted; Phase 18 remains blocked.**

Recommendation: the observed exposure benefit justifies **one final, short
independent-seed confirmation on the same subset**, using the cheaper 21.24M
configuration. It does not justify a full-corpus rerun, a larger model, an accepted
base claim, or an open-ended sequence of additional passes. The prospective study
below is not implemented or authorized. Stop for owner/master-planning review.

## Evidence and budget

Return ZIP SHA-256:
`db0092d459051e5ddf613ebac09d6ffa8b19a32804f629608d7720cd37a50acb`
(552,504,674 bytes). The independent corrective reader verified all 1,667 included
payload files plus the return manifest, rebuilt all 1,910 serious input signatures,
state digests/counters and the saved comparisons, and bound the original tools to
`15cf9d96e8baa00cdc29e76840e283c2ab4bc814` and correction to
`0eca1d836f24b2b80ad93730db0719367598bc68`.

The second review binds configuration/seed/schedule on every update, verifies all
17 imported native-source snapshots, checks the four included model tensor files
as finite float32 against their checkpoint fingerprints and exact parameter counts,
and confirms the scored checkpoint identities. It constructs/runs no model. A/B
initial model/sampler states match, with empty optimizers; their first 19 warmup
updates have identical model/optimizer/sampler/RNG fingerprints. B/C have identical
recorded schedules and two-pass input sequences. All eight checkpoint boundaries
are accounted for; no serious resumption or alternate checkpoint selection occurred.

The corrected archive preserves the original packaging exception, failed guardian,
logs and pre-packaging outcome. The 1,703-file preservation inventory matches every
included original byte and binds **43 omitted tensor/cache/bytecode records**.
Those omitted bytes remain on Windows and were not rehashed on Mac. Before/after
identity equality is supported by the exact reviewed correction's execution and
its preservation record; this compact return is not a full Windows backup.

| Time accounting | Verified seconds |
| --- | ---: |
| Original preparation, including conformance | 36.759 |
| All three training jobs/stage | 821.603 |
| Five fixed development observations/stage | 406.280 |
| Original guardian elapsed at packaging failure | 1,272.765 |
| Conservatively charged original time | 1,273 |
| Packaging-only correction, including closure | 11.934 |
| **Cumulative charged study** | **1,284.934 = 21m24.934s** |
| **Cumulative charged packaging** | **20.934** |

The original failed packaging is charged nine seconds, not erased or credited back.
The correction used zero additional optimizer updates and no new model inference,
generation or evaluation. The 2,700-second total and all original stage caps pass.
Combined recorded original artifacts, corrected ZIP/reports and reserve are about
2.655 GiB, below 8 GiB. Native worker resource peaks pass the prescribed host/VRAM
bounds. This is scoped evidence for the supervised jobs, not every Windows process.
Unused time does not authorize another action in this closed study.

Detailed values, tensor identities, source/runtime observations and derived
contrasts are in [results.json](results.json). [Review](RESULTS_REVIEW.md) records
verification scope, initial supplementary-check corrections and limitations.

## Fixed development outcomes

Lower matched bits per byte (BPB) and repetition are better. ARC columns are raw
correct counts. B1 is the prespecified first-pass observation inside B, not an
independently trained fourth arm or an eligible replacement checkpoint.

| Endpoint | Exposure / parameters | Matched BPB | BPB / Phase 5 | Repetition | ARC-Easy /570 | ARC-Challenge /299 | Failed gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 5 reference | Preserved historical base | 1.872944 | 1.000000 | 38.82% | 157 | 66 | Reference |
| A1 | 1 pass / 34.09M | 2.554079 | 1.363671 | 79.28% | 143 | 65 | 8 |
| B1 | 1 pass / 34.09M, long schedule | 2.535825 | 1.353925 | 71.68% | 145 | 62 | 8 |
| B2 | 2 passes / 34.09M | 2.472051 | 1.319875 | 63.29% | 151 | 62 | 8 |
| C2 | 2 passes / 21.24M | 2.469469 | 1.318496 | 63.36% | 159 | 66 | 8 |

All four fail the matched BPB ratio, both book guardrails, ARC-Easy minimum,
ARC-Easy gain, ARC-Easy chance lower bound, repetition maximum and repetition
increase. C2's book ratios are 1.461919 / 1.268392 (required <=1.02), ARC-Easy
27.8947% (required >=35%), and gain +0.3509 percentage points (required >=5).
Its Wilson lower bound 0.243703 is below the fixed chance threshold. Matched BPB
must be <=0.90 of the reference and repetition <=0.25 as well as satisfy its
reference-relative limit. Other numeric passes cannot offset these failures.

All have zero empty continuations; all five models remain 0/96 on the descriptive
instruction checks. External raw/byte regression guards pass. These observations
are not useful instruction-following, reasoning or general assistant capability.
The Phase 5 quality aggregates and generated IDs repeat the previous pinned Windows
reference exactly; runtime timing is separately measured, not expected identical.
No final/reserved scoring occurred.

## What the controlled comparisons establish

**Schedule at equal exposure (A1→B1):** same model, initial state, first-pass data
and warmup; only schedule horizon differs. BPB falls 0.715%, both books improve,
and repetition falls 7.60 percentage points. Fifteen of 24 prompts improve on
repetition, eight worsen and one ties. This supports a schedule effect for this
seed/protocol; extra exposure cannot explain that first-pass difference.

**Additional exposure on the long schedule (B1→B2):** BPB falls 2.515%, both books
improve and repetition falls 8.39 points (15/24 improve, eight worsen, one ties).
Monitoring NLL falls 5.92550→5.64605. Exposure and the later learning-rate path are
still coupled; this is not a schedule-free causal estimate of repetition alone.

**Practical one-versus-two-pass recipe (A1→B2):** BPB falls 3.212%, both books
improve, and repetition falls 15.99 points (18/24 improve, five worsen, one ties).
This satisfies the proposal's qualitative requirement of consistent book/repetition
direction for useful planning evidence. It does not pass any failed acceptance gate.
ARC-Easy gains eight answers, but its paired 95% interval is −0.526 to +3.333 points,
including zero. B1→B2's Easy interval is −0.702 to +2.807 points; Challenge is unchanged.
No convincing external-accuracy gain is established.

**Depth at matched exposure/schedule (B2→C2):** C2 improves BPB by only 0.104%, with
slightly worse repetition (+0.074 points; 11 prompts improve and 12 worsen).
Its eight-answer Easy advantage also has a paired interval including zero
(−0.526 to +3.333 points). Thus the smaller model is not established as a better
language model. It is a measured efficiency option: complete training jobs took
388.412 seconds for B versus 235.977 for C, **39.2% less time**, with nearly tied
observed development quality. This is a descriptive near-tie, not a formal
equivalence/noninferiority result. It is equal-exposure evidence, not equal-compute
superiority.

The subset contains only 1,943,015 distinct target positions; B/C each execute
3,886,030 positions, and all arms together 9,715,075. Repeated tokens are not new
unique data. All training and monitoring remain encyclopedia-only. Domain coverage
was not varied. These results cannot prove that insufficient exposure or domain
coverage caused Phase 17.1's failure, predict a saturation point, or establish
seed-robust performance. Its full one-pass BPB 2.051643 remains better than every
subset endpoint, under a different exposure/schedule/unique-data budget; this is
not a matched experiment proving that repeated small data beats fresh coverage.

## One prospective next decision, not an implementation

**Recommend one final fixed-subset exposure confirmation rather than stopping the
unchanged-corpus direction immediately.** The reason is the prespecified coherent
book/repetition benefit, not spare runtime or a claim that the gates are close.
Use C's lower measured cost as the practical choice; do not claim quality superiority.

- Two fresh 21,238,272-parameter models, both new seed **161**, empty optimizers,
  unchanged tokenizer and exact existing 1.94M-target subset. No old weights reused.
- Control: two passes / 764 updates. Candidate: four passes / 1,528 updates.
  Keep 19-update warmup, original AdamW/precision/deterministic settings and matched
  permutations; cosine ends at each arm's declared horizon. Flush every pass tail.
- Retain the candidate's fixed two-pass observation to separate schedule effects
  at equal exposure. Score only Phase 5, the control endpoint, that diagnostic
  observation and the fixed four-pass endpoint after training. No adaptive change,
  extra seed, checkpoint search, continuation, new corpus or full-corpus rerun.
- Total prospective serious work: 2,292 updates / 11,658,090 target executions;
  unique positions remain 1,943,015. Allow at most the same 52 conformance updates
  (2,344 physical updates maximum); no retries or additional optimizer fixtures.
  The new two-pass control also checks whether
  C2's early result persists under another seed, without claiming a two-seed proof.

Measured-job projection is about **17.03 minutes**: three times C's two-pass job
(707.931 s), three C-sized development observations plus reference (256.189 s),
preparation (36.759 s), and observed cumulative packaging (20.934 s). This linear
projection is not a promised runtime, and actual preparation/validation must fit.
Recommend a **20-minute target and 30-minute hard end-to-end cap**, including preparation,
the capped conformance checks, evaluation and packaging.
The owner's standing 45-minute ceiling remains binding; it is not an extension
allowance beyond a newly approved smaller cap. New paid compute remains zero.

Treat this as the **last exposure diagnostic on this fixed subset**. If the fixed
four-pass endpoint still fails the unchanged development gates, stop this route
and rethink data/training design rather than add passes again. A pass still needs
separate owner selection/contamination review and unchanged final-access policy;
it would not itself accept Base Model 2.0 or authorize Phase 18. This stop rule is
a prospective resource decision, not proof that all unchanged-corpus models fail.

No next-study controller, configuration, data preparation or handoff was created.
The original failed Phase 17 and development-failed Phase 17.1 remain permanent,
separate records. **Stop now for owner/master-planning review; no publication.**
