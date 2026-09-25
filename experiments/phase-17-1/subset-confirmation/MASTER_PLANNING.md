# Owner/master-planning decision after the exposure studies

**Close the fixed-subset exposure direction. Do not authorize another Base Model
2.0 training attempt yet.** First require a materially revised data/training design
brief. More passes, another seed, a different schedule sweep on this same subset,
or selecting E2 instead of E4 would evade the prospective stopping rule.
No new experiment, data acquisition or implementation is proposed for execution here.

Evidence: [final confirmation](CLOSURE.md), [first subset diagnostic](../small-subset/RESULTS.md),
[full Phase 17.1](../CLOSURE.md), [Data 2.0](../../phase-15/REPORT.md), and
[Phase 16 candidate selection](../../phase-16/CLOSURE.md).

## What the accumulated evidence supports

| Area | Supported finding | Limit / implication |
| --- | --- | --- |
| Data/domain composition | Accepted Data 2.0 pretraining is entirely English encyclopedia prose. Both book evaluations regress versus the historical book-trained base. Full Phase 17.1 learned its own distribution but failed quality; both subset studies also fail. | Broader volume and a passing construction audit did not establish adequate general-English coverage. Domain mismatch is a priority hypothesis, not a demonstrated sole cause: no domain mixture was experimentally varied. |
| Training/optimization | At equal exposure, longer horizons improve observed book BPB/repetition in both diagnostic seeds. In the final candidate, passes three/four lower encyclopedia monitoring loss but worsen both book BPB scores relative to E2. | Schedule design matters. More repeated optimization is not a reliable proxy for transfer. This does not identify an optimal rate, prove an AdamW/clipping defect, or establish a universal saturation point. |
| Model scale | The 21.24M model had nearly the same observed early quality as 34.09M at matched exposure, with 39.2% lower complete training-job time. The seed-161 two-pass result is again close in BPB to seed 160. | Use the smaller model as an economical experimental control, not as a proven optimal architecture or accepted base. One matched-depth comparison and two small-model seeds do not establish general superiority/equivalence. |
| Compute feasibility | The final two-arm study, scoring and packaging finish in 16m51.561s under its 30-minute cap. | Bounded local diagnostics are practical. This does not show that a competent base can be produced under 45 minutes; the older full Phase 17.1 training/preflight took about 76 minutes and cannot simply be rerun under today's cap. |

The Phase 5 reference remains 1.872944 BPB; full Phase 17.1 is 2.051643, and E4 is
2.417791. Those runs differ in corpus size/composition, tokenizer, architecture,
schedule and exposure. Cross-tokenizer BPB is comparable, but their histories are
not a controlled causal comparison. Do not conclude that unique data, model depth,
tokenizer size or any single setting explains the historical gap.

The final E4 has 7,772,060 target executions but only 1,943,015 distinct positions.
Its encyclopedia monitoring loss falls 5.42888→5.23005 after E2, while both books get
worse and ARC-Easy drops seven answers with an interval including zero. Mean repetition
improves slightly, but fourteen of 24 prompts worsen. This pattern is compatible
with increasing specialization to the training objective; it is not proof of a
particular overfitting mechanism. Falling monitoring loss cannot justify more subset
exposure or a post-hoc checkpoint choice. All instruction checks remain zero.

## What should change before another attempt

**1. Define a different data proposition, independently of test examples.** Specify
which forms of general-English language and knowledge the base is meant to learn,
and what is absent from the current encyclopedia-only distribution. Review retained
text quality, source diversity, lengths, paragraph/document continuity and repeated
content—not merely raw megabytes. A revised corpus should have an independently
justified coverage/quality rationale, rather than added literary material selected
to make these two known book scores easier. Preserve whole-source-family separation,
provenance/rights, exact/near deduplication and contamination screening, including
already observed evaluation outputs. No protected evaluation source or near-duplicate
may become training data. No revised corpus or source acquisition was made here.

**2. Make the objective and optimization design explicit for those inputs.** Account
for unique positions, repeated executions, useful targets, padding, document boundaries
and EOS exposure. Define the learning-rate horizon relative to that budget before
execution, and monitor the intended training domains without treating their loss as
acceptance. Existing instruction conversations are a separate resource, not pretraining
exposure already consumed; mixing or repurposing them requires its own objective,
provenance and evaluation analysis. SFT, decoding changes or later post-training must
not be used to disguise a failed base gate or begin Phase 18 prematurely.

The optimizer, clipping threshold and precision were held fixed; these studies do
not justify a broad hyperparameter search. Clipping is frequent, but changing it is
not an established remedy. Any later controlled test must target a justified change
in data/training design with an appropriate unchanged-factor control, not reopen
exposure-only tuning on the closed subset. Hold other axes stable where possible.

**3. Keep scale a measured tradeoff.** The smaller dense configuration is a reasonable
low-cost control for a genuinely new data/training hypothesis. Available VRAM alone
does not justify enlargement; Phase 16's 53M/73M probes established feasibility, not
long-run quality benefit. Parameter totals also hide allocation: the existing
16,384×512 embedding alone is 8.39M parameters, about 39.5% of the 21.24M model.
The tokenizer was selected for compression, not proven learned quality. Review
compression, coverage and compute on any proposed revised inputs before changing
it; do not copy an external vocabulary, model layout or recipe. No architecture,
tokenizer, model size or optimizer change is selected by this closure.

**4. Require a credible, fully costed decision before execution.** Any future
experiment needs new explicit owner authorization and a distinct, meaningful
hypothesis; frozen inputs/initialization, control, endpoints and stopping criteria;
unchanged Phase 5 reference and all acceptance gates; and an inclusive projection
using measured supervision/evaluation/checkpoint/packaging costs. The standing
maximum is **45 minutes total Windows GPU/computer time per experiment**, with a
substantially lower target and practical margin where feasible, zero paid resources,
no hidden preparation or deferred packaging, and no extensions or chained runs to
circumvent the cap. Current spare time is not available for further subset work.
If no credible revised design fits, stop or return for a separate roadmap decision;
do not substitute an execution demonstration or lowered threshold for quality.

## Present decision boundary

The owner/master-planning decision is whether to develop such a data/training
redesign brief, or stop/rethink the wider Base Model 2.0 direction. This document
does not authorize that work, name a new dataset, prescribe another executable
experiment or reopen this subset. Preserve all negative results and unaccepted
weights. The final acceptance set stays unscored. Phase 17 remains permanently
failed/interrupted, Phase 17.1 development-quality-failed, and Phase 18 blocked.
**No implementation, new data, training or publication follows automatically.**
