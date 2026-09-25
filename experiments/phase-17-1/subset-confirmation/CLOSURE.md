# Final subset confirmation — closed; fixed-subset exposure direction stopped

**E4 fails eight fixed development gates. The prospective stopping condition is met:
this fixed-subset exposure direction is closed. No additional passes, another seed,
or exposure-only experiment on this subset is proposed or authorized.** Improving
training or monitoring loss cannot reopen it. No Base Model 2.0 is accepted; no final
scoring or Phase 18 follows. Return to owner/master-planning review of data/training
design. This closes a diagnostic, not the Roadmap 2.0 base-quality milestone.

## Independently verified execution and return

Returned archive: 302,819,663 bytes, SHA-256
`81bba1b8200dc382ef241afeea6fe110eae517a0226bd59e79c21791d0aa4722`.
Receipt, completion and guardian records bind the same return. All **1,421 payload
files plus the return manifest** verify. Thirty-seven omitted tensor records remain
inventory-bound on Windows; their absent bytes were not rehashed on Mac. Preserve
them: this compact return is not a full backup.

The known handoff binds to reviewed commit
`ba262bdb598d164da322972c1208e7fed55c9335` and the frozen plan/authorization.
Fifteen actual native-source snapshots match the unchanged implementation. The
reviewed local reader reconstructs every saved serious input signature, permutation,
counter, state digest, checkpoint identity and numerical comparison, including the
terminal stop disposition. No returned source was executed and no model was run.

- D: fresh seed 161, 21,238,272 parameters, 764 updates / 3,886,030 targets.
- E: same fresh model/seed, 1,528 updates / 7,772,060 targets.
- Total: **2,292 serious / 2,344 physical updates**, including the 52 separate
  prescribed conformance updates; **11,658,090 serious target executions**.
- Same 1,943,015-position subset; repetitions are not new unique data. All pass tails
  contain five microbatches, with one window in the final microbatch. No serious retry,
  replay, recovery, seed change, omitted update or alternative checkpoint selection.
- Eight prescribed checkpoint boundaries are accounted for. D/E initial model and
  sampler fingerprints match, initial optimizers are empty, and the common first
  19 warmup updates match exactly. Recorded initial model identity differs from the
  prior seed-160 initializer; absent initialization tensor bytes remain inventory-bound.
- D2/E2/E4's three included tensor files contain **90 finite float32 tensors /
  63,714,816 elements**, matching the exact logged checkpoint fingerprints, model
  counts and evaluator weight identities. This is CPU storage inspection, not inference.

| Complete Windows time accounting | Seconds |
| --- | ---: |
| Preparation/conformance | 32.922 |
| Both training jobs | 702.885 |
| Four fixed development observations | 265.321 |
| Packaging through outer closure | 10.433 |
| **Guardian total** | **1,011.561 = 16m51.561s** |

Exit is zero, with no deadline termination. Completion time is 1,011.544 seconds;
packaging/readback precedes completion and the outer guardian record. All stage
caps and the **1,800-second inclusive cap** pass, without a later correction.
Training peaks are approximately 2.13 GiB host RSS and 0.775 GiB reserved GPU; recorded
bounds pass. Reconstructed original/returned artifact accounting with reserve is
about 2.05 GiB, below 8 GiB. Resource claims cover supervised workers, not every
Windows process. Unused budget does not authorize further work on this direction.

Detailed arithmetic/identities are in [closure-verification.json](closure-verification.json);
[separate review](CLOSURE_REVIEW.md) records the verification scope and limits.

## Fixed development results

Lower matched bits per byte (BPB) and repetition are better. E2 is a prespecified
observation within E, not a selectable replacement for E4.

| Endpoint | Matched BPB | Ratio to Phase 5 | Repetition | ARC-Easy /570 | ARC-Challenge /299 | Failed gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 5 reference | 1.872944 | 1.000000 | 38.82% | 157 | 66 | Reference |
| D2 | 2.466165 | 1.316732 | 65.21% | 159 | 65 | 8 |
| E2 | 2.410365 | 1.286940 | 56.62% | 162 | 59 | 8 |
| **E4** | **2.417791** | **1.290904** | **55.10%** | **155** | **63** | **8** |

All three endpoints fail matched BPB, both book guardrails, ARC-Easy minimum,
ARC-Easy gain, ARC-Easy chance lower bound, repetition maximum and repetition
increase. For E4 specifically:

- Matched BPB ratio **1.290904**, required <=0.90: **29.09% worse** than Phase 5.
- Book ratios **1.436479 / 1.240048**, each required <=1.02.
- ARC-Easy **27.193%**, required >=35%; gain **−0.3509 points**, required >=+5.
  Wilson lower bound **0.237019**, below the fixed chance threshold **0.249971**.
- Repetition **0.551007**, required <=0.25; increase **0.162821**, required <=0.10.

External raw/byte regression guards and empty-continuation limits pass; those passes
do not offset failures. All four models remain 0/96 on the descriptive instruction
checks. There is no useful instruction-following or broad reasoning claim. The
historical reference's quality aggregates reproduce exactly. All evaluator records
are development-only, without final access or an acceptance declaration.

## Scientific interpretation within the diagnostic

**D2→E2, equal exposure:** the longer horizon improves BPB by **2.263%**, both books
improve, and mean repetition falls **8.59 percentage points**. With model, initial
state and first two permutations held fixed, this supports a schedule effect in
this seed/protocol. It is not evidence of an optimal general learning-rate recipe.

**E2→E4, the two additional scheduled passes:** encyclopedia monitoring loss improves
**5.42888→5.23005**, yet matched book BPB worsens **0.308%**, including both books
(+0.387% / +0.276%). Mean repetition improves only **1.52 points**; ten prompts improve
and fourteen worsen. ARC-Easy falls by seven correct answers. Its paired 95% interval
is **−3.333 to +0.702 points**, including zero; do not assert a conclusive general
accuracy decline. Exposure and the later schedule remain coupled, so the result
cannot identify a pure exposure effect or prove an optimizer defect/overfitting cause.

**D2→E4, complete recipe:** book BPB improves **1.961%** and repetition falls **10.11
points**, but the book improvement was already present at E2; all fixed gates above
still fail. Relative improvement cannot override the prospective stopping rule.

The prior same-size two-pass C2 (seed 160) and D2 (seed 161) are close on BPB
(2.469469 versus 2.466165) and both score 159/570 on ARC-Easy, while repetition differs
by +1.85 points. This is limited consistency of the weak two-pass result, not broad
seed robustness. The earlier one→two-pass signal did not establish that repeated
exposure would continue transferring to books or meet the gates.

**E2 must not replace E4.** No post-hoc checkpoint choice, new seed, revised gate,
new decoding rule or additional subset exposure is permitted. The stop is a faithful
execution of the predeclared rule, not a claim that every possible model/data recipe
is incapable of improvement. Preserve the permanent Phase 17 failure, the separate
Phase 17.1 quality failure and both subset studies as distinct evidence.

See [owner/master-planning analysis](MASTER_PLANNING.md) for what must change before
another base-model attempt. No next experiment, data acquisition, implementation,
Phase 18 or publication was performed or authorized by this closure.
