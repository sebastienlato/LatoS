# Final fixed-subset exposure confirmation — authorized design

The owner explicitly authorized implementation, validation and a bounded Windows
handoff after reviewing the small-subset results at
`dffe041b2126783dc77dc01b77c66c174ba761cc`. Mac stops at the execution checkpoint.
This is the **final authorized experiment for this fixed-subset exposure direction**,
not a Base Model 2.0 acceptance run. No publication, Phase 18 or paid resources.

## Fixed comparison

Two fresh native 21,238,272-parameter models: four layers, width 512, eight heads,
FFN 1,408, vocabulary 16,384, context 512. Use **seed 161** for both model
initialization and CPU shuffle; new empty optimizers. Never load prior trained,
probe, diagnostic or failed-study weights. D/E initial tensors/samplers must match.

| Arm / observation | Exposure | Updates | Status |
| --- | ---: | ---: | --- |
| D2, control endpoint | Two complete passes | 764 | Short-schedule control |
| E2, candidate midpoint | Same first two passes | 764 | Prespecified diagnostic only |
| E4, candidate endpoint | Four complete passes | 1,528 | Sole terminal endpoint |

Use exactly the existing Phase 16 subset: 4,096 training documents, 6,105 windows,
1,943,015 targets/pass; monitor 128 separate development documents, 184 windows,
57,036 targets. The existing cache/tokenizer/selection identities remain unchanged.
No new data, re-selection, re-tokenization, repacking, added domains or corpus expansion.

BF16 autocast with float32 weights/gradients/AdamW state; unchanged deterministic
policy. Batch two, accumulation eight, 19-update warmup, peak rate 0.0003, cosine
to 0.1× at each arm's declared final update, betas 0.9/0.95, epsilon 1e-8, decay
0.01, clipping 1. All other model/runtime settings remain unchanged. Seed 161 is
independent of the previous seed-160 study, not a sweep or a chosen best seed.

Persistent CPU sampler RNG generates successive complete permutations. Flush
accumulation at every 382-update pass boundary: 3,053 microbatches/pass, final update
five microbatches and final microbatch one window. The first two permutations in
D and E are identical; E then continues to its third and fourth permutations.
No schedule restart, missing tail, cross-document stitching or resumed optimizer.

Exactly **2,292 serious updates / 11,658,090 target executions**, including repeats;
only 1,943,015 distinct training positions. At most the existing four tiny CUDA
conformance pairs (8+5 updates each), **52 additional updates**, once. Maximum physical
updates 2,344. Those fixture model seeds/seed-160 samplers remain their frozen
mechanics controls and never initialize the seed-161 serious arms. No other optimizer
fixtures, extra seed, retries, resumptions or full-corpus training.

## Fixed observations and terminal rule

Monitor the fixed encyclopedia development subset and save immutable diagnostic
checkpoints at initialization and every pass end: D at 0/382/764, E at
0/382/764/1,146/1,528. Retain all eight checkpoints, counters, source/runtime identities,
per-update full-state/input fingerprints, timing, resources and failures. Checkpoint
readback must be exact; the pass-tail adapter uses a no-resume diagnostic envelope.

Run the entire unchanged fixed development protocol only after both arms finish,
once each for **Phase 5 reference, D2, E2 and E4**. No E1/E3 scoring or checkpoint
search; no intermediate fixed-suite result may influence training. Preserve raw
rows and native paired uncertainty. Primary acceptance comparisons always use
Phase 5 (`f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa`),
with unchanged protocol/gates. Cross-arm contrasts use verified saved rows directly,
never the base gate with an experimental reference.

D2→E2 tests schedule horizon at equal exposure; E2→E4 describes extra exposure along
a fixed long schedule, with its later learning-rate evolution still coupled to
exposure; D2→E4 compares the complete recipes. The previous C2 seed-160 result may
be examined as historical context on Mac, never substituted as the primary reference
or rescored on Windows. A second seed does not establish broad seed robustness.

**If E4 fails any fixed development gate, stop this fixed-subset exposure direction
and return to owner/master-planning review of data/training design. Do not propose
more passes merely because loss or quality is improving.** D2/E2 cannot replace E4.
Even if E4 passes, no model acceptance, final scoring or Phase 18 is authorized;
stop for owner review. Any interrupted/incomplete study stops without inventing an
E4 score or authorizing a retry. No outcome automatically permits further exposure.

## One complete 30-minute Windows clock

Observed-job projection: approximately 17.03 minutes. Target **20 minutes**.
**Absolute cap: 1,800 seconds**, including Python startup, archive checks/extraction,
preparation, conformance, both training jobs, all evaluations, verification,
packaging/readback and closure. The old study's spare time is not reused.

| Stage | Hard cap |
| --- | ---: |
| Preparation and all conformance | 180 s |
| Both serious training arms together | 1,080 s |
| Four fixed development observations together | 300 s |
| Verification and packaging | 180 s |
| Stage caps plus evidence-closure reserve | 1,740 + 60 = 1,800 s |

The kickoff captures Windows QPC before Python starts. The controller cutoff is
1,790 s; an independent kill-on-close Windows Job Object guardian cuts the entire
process tree at 1,795 s. These margins do not extend the 1,800-second cap. Stage and
whole-study clocks never reset; no later packaging invocation or corrective attempt
is automatically authorized. On failure, preserve existing evidence and stop.

Use the existing RTX 4070 SUPER and pinned environments only. Host worker RSS <=8 GiB,
reserved VRAM <=85%, new artifacts <=8 GiB including return/transport reserve,
at least 12 GiB free initially. No paid compute, dependency/driver upgrade, fallback
backend or concurrency with another GPU workload. Missing identities/controls,
nonfinite state, resource/time limit, corruption or guardian failure ends the attempt.

## Current checkpoint

Mac implements and tests the standalone tools using zero-update CPU checks and
saved-row fixtures; no Mac training/inference. The exact source and transfer must
be frozen before Windows invocation. Run only the supplied kickoff and preserve
all earlier failures, source, data and artifacts. Return evidence for independent
Mac review. Stop at the Windows execution checkpoint in this task.
