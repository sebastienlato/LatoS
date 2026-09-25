# Base Model 2.0 exposure/domain design review

Date: 2026-09-25. Recovered source: `c2277e7104e92408581db41cd0f9852c8f7cb2ee`.
Status: **REJECTED by the owner: the 6.6-hour/four-pass proposal must not be implemented.**

The owner now caps the entire next experiment at 45 minutes. The analysis below is
historical; its recommendation and budgets are superseded by
[the small-subset proposal](SMALL_SUBSET_PROPOSAL.md). No part of the old study
may be executed using separate 45-minute allocations.

The owner explicitly authorized this existing-evidence review after publication.
That authorization supersedes the earlier wait for permission to review, and nothing
else. Local HEAD and live origin/main matched the recovered source at entry. The
tracked closure's pending-publication wording is historical; that closure is published.

**Recommendation: one two-trajectory, unchanged-data exposure study, with a fixed
schedule diagnostic. Do not enlarge the model or start a domain-mix experiment now.**
There is enough ongoing in-distribution learning and local resource headroom to
justify this limited information-gathering experiment, but no evidence that it will
produce an accepted base. If its fixed endpoint still fails development acceptance,
stop this unchanged-corpus exposure direction and return to master planning. No
automatic extra epochs, seed search, data revision or subsequent experiment follows.

Phase 17 remains permanently interrupted/failed. Phase 17.1 completed every planned
target without serious replay but failed seven development gates; matched BPB was
9.54% worse. No final scoring occurred, no Base Model 2.0 is accepted, and Phase 18
remains blocked. This review does not revise historical evidence or acceptance.

## Evidence and what it can establish

The source records are [closure and verification](CLOSURE.md),
[closure-verification.json](closure-verification.json), [next decision](NEXT_DECISION.md),
[Phase 15](../phase-15/REPORT.md), [Phase 16 closure](../phase-16/CLOSURE.md),
[Phase 16 candidate measurements](../phase-16/windows-verification.json),
[Phase 5](../phase-5/REPORT.md) and [Phase 14](../phase-14/REPORT.md).
[Derived arithmetic and source hashes](exposure-domain-analysis.json) record this
review's calculations. Only saved records were read: no model was constructed,
loaded, run or scored, and no optimizer or data preparation pipeline was invoked.
Private planning input was read locally and is not reproduced here.

### Trajectory and checkpoints

The nine retained checkpoint boundaries have monitoring losses below. These are
Data 2.0 development monitoring measurements, not the fixed book BPB or ARC suite.
All 7,485 saved update records were read, their trace hash checked against the
returned inventory, and target totals independently summed. Earlier checkpoint
weights were not loaded or scored; many tensor payloads remain on Windows only.

| Update | Cumulative targets | Monitoring NLL | Learning rate at boundary |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 9.81154 | initialization |
| 1,000 | 5,052,854 | 5.27354 | 0.000294885 |
| 2,000 | 10,143,264 | 4.77446 | 0.000266670 |
| 3,000 | 15,200,880 | 4.46975 | 0.000218927 |
| 4,000 | 20,262,702 | 4.25757 | 0.000160825 |
| 5,000 | 25,312,099 | 4.11841 | 0.000103525 |
| 6,000 | 30,346,690 | 4.03107 | 0.000058034 |
| 7,000 | 35,384,765 | 3.97870 | 0.000033088 |
| 7,485 | 37,811,418 | 3.96396 | 0.000030000 |

Monitoring improves by 0.08734 over updates 5,001–6,000, 0.05237 over
6,001–7,000, and 0.01474 over the last 485 updates. Per million targets these
last intervals improve by approximately 0.01735, 0.01039 and 0.00607 nats.
The curve is still falling, with diminishing gains while the learning rate decays.
There is no observed saturation proof, extrapolatable scaling law or book-quality
curve. Earlier checkpoints cannot be inferred to pass the failed final gates.

The final interval's token-weighted online training loss is 3.99773 versus final
monitoring loss 3.96396. These differ in examples and model times; subtracting them
is not a valid generalization-gap estimate. Gradients were clipped in 92.71% of
updates and all updates after 4,000; the last interval's median preclip norm is
1.31137. This is a reason to retain optimization diagnostics, not evidence that
clipping caused failure or authorization to tune its threshold. The original failed
Phase 17 recovery's similar monitoring loss is neither a valid replicate nor a
quality comparator: its equivalence gate failed and fixed development was unscored.

### Exposure and historical baseline

The actual coalesced-document layout contains 37,730,940 content positions plus
80,478 EOS targets = **37,811,418 positions per pass**, 119,748 windows, 7,485
updates. One pass supplies **1.10925 targets per parameter** for 34,087,424
parameters. Positions are not unique vocabulary, n-grams or independent information.
Phase 15's 37,226,334 isolated-record content tokens and 37,559,116 targets use a
different layout; they must not replace the executed totals. Joining retained
chunks/newlines explains the layout distinction, not extra acquired documents.

The historical 17,308,032-parameter base used eight books, an 8,192-entry tokenizer,
256-token windows, MPS float32, peak learning rate 0.0006 and weight decay 0.1.
It saw 5,761,229 targets over 4.511389 passes of 1,277,041 corpus positions
(about 0.333 executed targets/parameter), and its book monitoring plateaued near
2,500 updates. Phase 17.1 used the new 16,384-entry tokenizer, 512-token windows,
CUDA BF16, peak 0.0003 and weight decay 0.01, among other differences.

Therefore the historical comparison does **not** isolate exposure or domain.
Phase 17.1 already had about 6.56 times the total target executions and many more
corpus positions, yet worse matched book BPB. Cross-tokenizer position ratios are
budget descriptors, not equal-information comparisons. The baseline's repeated
literary exposure makes domain alignment plausible; it does not establish that
four passes of encyclopedia text will reproduce its advantage. No universal
required tokens-per-parameter ratio is asserted from these records.

### Data composition and domain hypothesis

Accepted pretraining is **100% encyclopedia prose**, not merely mostly encyclopedia:
80,478 training documents from two pinned Wikipedia shards, 158,795,397 retained
UTF-8 bytes (159,300,003 under the coalesced layout). It is 29.76 times the old
training text by bytes, but larger volume does not establish balanced coverage.
The new development monitor is encyclopedia too. The fixed matched-LM development
suite contains two literary books; both regressed. This is consistent with domain
mismatch, but gives no randomized or controlled causal estimate.

The separate 14,259 training instruction conversations (12,156 Dolly and 2,103
OASST) were not pretraining exposure. Counting them as base-training breadth would
be incorrect. Mixing them in now changes format, objective/use and repetition
structure, with implications for later instruction evaluation. It is not a clean
substitute for general-English pretraining or permission to begin Phase 18.

The existing eight historical training books are a possible future genre control,
not a balanced modern-English corpus. Reweighting this small stock toward two
already-observed literary gates risks narrow development specialization. A mix
would need fresh cross-corpus/source-family and protected-output audits, matched
exposure, recorded unique-versus-repeated positions, and frozen proportions before
training. Existing Data 2.0 clearance does not automatically clear a merged corpus.
No protected books/authors, near-duplicates, evaluation outputs or reserved splits
may be admitted. No such mixture or audit was built here. Increasing Wikipedia
repetitions cannot test missing-domain coverage, and a failed exposure study would
not prove that adding books is the remedy.

### Development distance and model-scale evidence

Matched BPB 2.051643 versus 1.872944 requires about **17.84% reduction from the
failed candidate** to reach the fixed 0.90 ratio (approximately 1.685649 with the
saved paired reference). Book ratios 1.192326 / 1.061554 exceed 1.02. ARC-Easy
165/570 needs at least 200/570 for the 35% gate: 35 more correct answers. The
observed eight-answer gain has a paired 95% interval of about −2.46 to +5.61
percentage points. Repetition 0.498873 must fall to at most 0.25 as well as satisfy
the reference-relative bound. These are material gaps, not marginal rounding
failures. Every unchanged gate must pass; aggregate improvements cannot trade off
against a failed gate. Current instruction scores remain descriptive 0/96.

Phase 16's common BF16 exposure was only 322,035 targets per candidate, about
0.85% of one full pass. Its final losses were 7.17584 / 7.23181 / 7.25245 for
34.09M / 53.36M / 72.63M parameters. All were within the frozen 2% band; the rule
selected the smallest eligible model, not a proven quality winner. Corresponding
update throughputs were 28,983 / 17,433 / 12,205 targets/s, reserved VRAM about
1.09 / 1.63 / 2.14 GiB. Larger models fit, but cost roughly 1.66x / 2.37x in this
short protocol and have no demonstrated long-run quality advantage. Holding the
selected model fixed is warranted. Compression gains likewise do not prove the
new tokenizer improves literary language modeling; retraining it now would add
another unisolated intervention.

## One prospective study: exposure with a schedule diagnostic

This is a proposed protocol, **not executable authorization**, not a new phase,
and not a continuation of a failed model. Its question is narrow: does a bounded
four-pass recipe on the unchanged accepted corpus substantially improve the fixed
development outcome, and is any early difference already explained by its schedule?
It cannot estimate the causal effect of broader domains or establish seed robustness.

| Trajectory / observation | Exposure | Schedule | Role |
| --- | ---: | --- | --- |
| A: fresh control endpoint | 1 pass; 37,811,418 targets; 7,485 updates | Original 375-update warmup, cosine to 0.1× at 7,485 | Contemporary unchanged-data/unchanged-recipe control |
| B1: fixed diagnostic in fresh B | Same first pass and order as A | 375-update warmup, cosine horizon 29,940 | Same-exposure schedule comparison; never eligible for promotion |
| B4: fixed candidate endpoint | 4 passes; 151,245,672 targets; 29,940 updates | Same B schedule, minimum at 29,940 | Sole candidate; no best-checkpoint selection |

A and B are exactly **two fresh initializations**, both seed 160 with empty AdamW
state; no old trained/probe/recovery weights are loaded. B1 is a prespecified
observation within B, not a third run or an adaptive stopping/selection point.
Run A first; B is conditional on valid control mechanics, not control quality.
A's expected quality failure is not a reason to tune it. B completes its planned
exposure before B1/B4 fixed development results are examined; diagnostic feedback
cannot alter training. No extra seed or fresh retry is allowed.

Hold model, tokenizer, accepted corpus, context 512, batch 2 × accumulation 8,
AdamW coefficients, clipping, BF16/float32 policy, deterministic execution and
all input identities fixed. Use the same seed-160 first permutation in both.
For later passes use successive permutations from the persisted CPU sampler RNG;
each corpus window appears exactly once per pass. Flush accumulation at each
pass boundary, retaining the two-microbatch tail, then start the next pass. Thus
B has exactly 29,940 updates, not a rounded or cross-boundary accumulation count.
No document stitching, repacking, data omission or schedule restart is introduced.
The longer horizon is an explicit treatment, not a silent change to the old plan.

B4 has only the original 37,811,418 distinct target positions at most; its other
113,434,254 target executions are repetitions. Four passes are an intentionally
bounded fourfold contrast, broadly comparable to the historical number of passes,
not a predicted optimum or a claim of sufficient data. This choice does not borrow
a scaling rule from another model or promise that gates can be reached.

Compare A–B1 to measure the schedule difference at equal exposure; compare B1–B4
to describe the trajectory under that fixed long schedule; compare A–B4 for the
practical one-pass versus four-pass recipe outcome. B1 is less annealed than A,
and B1–B4 includes later learning-rate evolution. Therefore these contrasts do
not perfectly separate a schedule-free exposure effect. They prevent attributing
all changes to token count, at less cost than a schedule-by-exposure grid. All
results are single-seed, correlated development observations.

Use only the existing Data 2.0 development monitor during prospective training,
at initialization, every 1,000 updates, each pass boundary and final update.
Retain immutable checkpoints at those same boundaries: nine for A, 34 for B.
Log full state/input signatures, counters, clipping and timing. At completion,
run the unchanged fixed **development** protocol on A, B1, B4 and the Phase 5
reference on the same pinned host/runtime/batches; retain all raw rows and paired
uncertainty. Phase 17.1 is an additional historical failed control only.
Do not score other checkpoints. Do not change prompts, decoding or normalization.

Only B4 can be considered for later acceptance. If it passes every development
and resource gate, stop for separate Mac selection/contamination and owner review
before any one permitted final comparison. This proposed study itself has **no
final access**. If B4 fails any gate, preserve the negative result and stop the
unchanged-corpus exposure direction; no substitution of A, B1 or another checkpoint.
Improved monitoring alone, partial gate progress or spare budget does not trigger
another run. A negative result rules out this bounded recipe, not all possible
exposure budgets, domains or architectures. Domain changes then require a new
master-planning decision, not an automatic next experiment.

## RTX 4070 SUPER cost and hard boundaries

Phase 17.1 measured 1,478.9826 seconds inside updates (25,565.8 targets/s), but
4,559.0812 supervised seconds for training/preflight (8,293.65 effective targets/s).
The 3,080.10-second scope difference includes supervision, full-state evidence,
setup, preflight, monitoring and storage. It is not an isolated deterministic
kernel penalty. The Phase 16 23.90-minute central projection is too optimistic
for this full evidence-preserving execution. Cost from the latter measured scope.

| Prospective item | Planning estimate / ceiling |
| --- | --- |
| Serious work | 37,425 updates; 189,057,090 target executions across A+B |
| Conformance | At most the existing 52 tiny CUDA preflight updates once for the study; 37,477 total physical updates maximum; no serious replay |
| Central supervised training/preflight | 5 × 4,559.0812 s = 6.33 hours; includes conservatively repeated fixed overhead |
| Development/inference planning allowance | 15 minutes; existing paired measurement took 162.8 s; four model observations require more work |
| Central execution total | About **6.6 hours**, not measured future runtime |
| Hard time limits | A training/preflight <=2 hours; B <=8 hours; development/inference <=1 hour; cumulative active <=11 hours |
| Retained artifacts | About 25 GiB planned; 32 GiB hard new-artifact limit; >=40 GiB free at start in addition to preserving old artifacts |
| Memory limits | Same <=8 GiB worker host RSS and <=85% actual reported VRAM; measured Phase 17.1 was 2.38 GiB host / 1.15 GiB reserved GPU |
| New paid services | **Zero**; existing local hardware only |
| Preparation/review work | Planning allowance 1–2 working days for implementation, zero-update checks, packaging and evidence review; not executed or measured here |

The storage estimate includes 43 checkpoints at approximately 409 MB each
(~16.4 GiB), logs/state traces, candidate exports, evaluation and administrative
reserve. Scaling the full prior charged 4.52 GiB by five gives 22.60 GiB, below
the 25 GiB planning allowance. It is a projection, not a disk availability check.
No throughput, larger batch, reduced tracing, quantization or cheaper checkpoint
policy is assumed. Longer duration does not imply proportionally higher VRAM,
but sustained Windows execution, storage peaks and multi-pass behavior are untested.
Electricity and equipment opportunity cost are not priced by the saved evidence;
zero new paid services is not a claim that hardware operation has no cost.

Fail closed on nonfinite state, identity/target/split mismatch, OOM, memory/disk/time
limit or numerical-control failure. **No serious-run restart/replay is proposed**:
an interruption preserves the failed attempt and ends the study for review. No
budget reset, extension, fallback backend or automatic replacement run. The existing
52-update synthetic preflight cannot establish real-data multi-pass recovery.

## Conditions before any implementation or execution

Stop now for explicit owner/master-planning review of this document. No runner,
multi-pass adapter, new dataset, training configuration or handoff is implemented.
If this design is approved later, implementation must preserve the historical
contracts and freeze a separate exact source/input/runtime manifest before training.

Required implementation checks include target conservation and two-microbatch tail
handling at all four pass boundaries; deterministic permutation/RNG persistence;
long-horizon schedule binding; unchanged short-control behavior; resource accounting;
immutable checkpoints; and a development-only scoring allowlist. Use zero-update
scripted preparation checks and only the existing capped 52-update CUDA conformance
set. If additional optimizer fixtures prove necessary, stop for a revised bounded
proposal; they are not covered by this proposal. Do not expand the 52-update CUDA
cap or serious exposure to accommodate unspecified tests.

Verify the accepted corpus/tokenizer and Phase 5 reference hashes, native runtime
and deterministic policy. Review existing contamination records against reuse of
identical training bytes and newly protected outputs through controlled fingerprint
checks only; no final examples may be exposed or scored. Missing clearance stops
preparation. A must reproduce the recorded one-pass input/counter/schedule evidence
and checkpoint tensor fingerprints/recorded losses under the unchanged same-host
exact-state and `1e-5` loss policies; unexplained deviations stop B, without relaxing
tolerances. Control comparison excludes run IDs and new administrative metadata,
not model/optimizer correctness. This is a same-pinned-runtime control requirement,
not a promise of cross-runtime bitwise identity.

The Phase 5 reference weights remain
`f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa`.
The fixed protocol remains SHA-256
`a83477dba5fb9ff26661384621e0bf07748dd36cb83069aa9fe9093865121b38`.
All base gates, external raw/byte regression checks, chance-bound check, generation
rules, per-book guards and final-access policy remain unchanged. Development is
already observed and this proposal is motivated by it; report that selection
history, and never describe a later development result as untouched validation.

Alternative actions considered and rejected for this next study: immediate larger
model (feasibility without quality evidence); simultaneous data/tokenizer/size change
(uninterpretable); new data acquisition (outside this review); an old-book or
instruction mixture now (coverage and contamination confounds); or silent continuation
of Phase 17.1 (violates fresh initialization and failure preservation). Stopping
entirely remains a reasonable owner choice if the bounded 6.6-hour estimate/11-hour
ceiling is not worthwhile. Neither choice completes the capability roadmap.

**Current boundary: zero training, zero new scoring/acquisition/paid resources,
no publication, and no Phase 18. Await owner/master-planning review.**
