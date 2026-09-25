# Phase 17.1 closure — execution complete, development quality failed

**The authorized fresh attempt completed its execution contract but failed the
prespecified development learned-quality gates. No Base Model 2.0 is accepted.
Final acceptance is prohibited for this candidate, and Phase 18 remains blocked.**

This closes one bounded experiment, not the Roadmap 2.0 Phase 17 quality milestone.
No retry, additional epoch, alternative checkpoint, final scoring or publication
follows from unused budget. The owner/master-planning process must separately
authorize any prospective new work. Publication of this closure also needs its
own explicit approval and would not authorize another experiment.

## Independently verified evidence

Executed controller source: `7cb5afd54d33fda4d18b84c5f0645d661b76e002`.
Return ZIP SHA-256: `66c2ddee8e85a289998461deb0e86945022b9e368a142f86ca9a68ba3a96e358`.
The supplied receipt and 189,072,025-byte archive match. All **1,366 included files**
rehash and match their extracted bytes; **81 omitted tensor/cache records** remain
inventory-bound on Windows. Those absent bytes were not rehashed or replayed on Mac.
The compact return is not a complete backup of the Windows experiment.

The returned controller bundle matches the original Mac handoff and Git source.
Twelve executed native-source snapshots (ten workers and two evaluators), each with
62 Python files, match the unchanged native implementation. Production job inventories,
requests, one session journal, completion receipts, runtime/policy bindings and
checkpoint metadata agree. CPU test fixtures contain deliberately failed/scripted
controls; those are not misclassified as production failures or optimizer work.
No returned source was executed during this review.

[Reconstruction](closure-verification.json), [local validation](closure-validation.json)
and [separate review](CLOSURE_REVIEW.md) record the evidence and its limits.

## Execution and training mechanics

| Measure | Verified outcome |
| --- | --- |
| Windows CPU checks | 35 passed / zero failed / zero skipped in each runtime; zero optimizer updates |
| CUDA adapter preflight | All four prespecified reference/replay pairs passed; 52 tiny optimizer updates total |
| Serious training | Fresh seed 160, empty optimizer, selected 34,087,424-parameter model; 7,485 updates |
| Exposure | 119,748 windows / 37,811,418 targets; one permutation, epoch zero |
| Final accumulation | Two microbatches; actual target denominator retained |
| Serious replay/resumption | Zero; no production interruption or failure recorded |
| Checkpoints | All nine at 0, 1,000…7,000 and 7,485, with exact full-state policy/readback evidence |
| Training/preflight time | 4,559.081206 seconds (75m59s) |
| Development/inference time | 162.799226 seconds (2m43s) |
| Total supervised active time | 4,721.880432 seconds (78m42s), within the three-hour ceiling |
| Peak recorded host / reserved GPU | 2,557,345,792 bytes (2.38 GiB) / 1,233,125,376 bytes (1.15 GiB) |
| Final charged artifacts | 4,852,667,987 bytes (4.52 GiB), including administrative reserve |

All 7,485 full-data input signatures, variable lengths, targets, cumulative counters,
sampler progress and exact same-Windows schedule values reconstruct against the
accepted Mac cache and seed-160 permutation. Every logged full-state digest validates;
checkpoint policy states match their corresponding trace boundaries. Optimizer
settings, model/data identities and the deterministic execution policy stay fixed.
The source enforces the initial free-disk prerequisite; no independent numerical
initial free-space measurement is claimed beyond its successful check.

Recorded CUDA-worker/training resource peaks pass the fixed limits; they are not
a claim about every Windows process. CPU tests use bounded tiny/no-update fixtures.
The full run was uninterrupted: same-process checkpoint roundtrips, the four tiny
CUDA restart pairs and the earlier selected-scale synthetic diagnostic are separate
evidence. **A serious real-data interrupted continuation was not exercised.**

## Failed fixed development acceptance

Both models used the same pinned CUDA runtime, deterministic policy, batch eight,
four threads, fixed prompts, scoring and matched-text protocol. Reference is the
preserved Phase 5 base, not the failed Phase 17 recovered model. The original
comparison, including all raw-row aggregates and paired uncertainty, reconstructs
exactly from the saved development outputs. No new forward pass, generation or
final scoring was performed on Mac.

| Required gate | Observed | Required | Result |
| --- | ---: | ---: | --- |
| Matched-text BPB ratio | 1.0954109564 | <=0.90 | fail |
| Alice book BPB ratio | 1.1923261860 | <=1.02 | fail |
| War of the Worlds BPB ratio | 1.0615539175 | <=1.02 | fail |
| ARC-Easy raw accuracy | 165/570 = 28.9474% | >=35% | fail |
| ARC-Easy gain | +1.4035 percentage points | >=5 points | fail |
| Mean repeated-trigram fraction | 0.4988729779 | <=0.25 | fail |
| Repetition increase | 0.1106861299 | <=0.10 | fail |

Matched BPB is **2.0516431115**, versus reference **1.8729437564**: about **9.54%
worse**, rather than the required improvement. Both books regress; this is not
just one book missing its guardrail. These are comparable matched-byte metrics,
not different-tokenizer perplexities.

ARC-Easy improves by eight cases from 157/570. Its Wilson lower bound exceeds
chance, so that particular gate passes. The paired gain interval is approximately
**−2.46 to +5.61 percentage points**, including zero; it does not establish the
required material gain. ARC-Challenge raw accuracy is 59/299 versus 66/299, a
2.34-point drop within the allowed 3-point regression limit. Byte-normalized
external regression gates and empty-continuation limits pass. Those passes cannot
offset the seven failed mandatory gates. Both models remain 0/96 on the recorded
instruction checks; these are descriptive at the base stage, not a new Phase 18 run.

Inference mechanics pass: median first token **6.1934 ms**, median decode
**148.5148 tokens/s**. These are the fixed native float32 KV-cache measurements on
a synthetic 128-ID prompt and 64 forced output tokens, not evidence of useful answers.

## Learning signal and limitations

Monitoring on the new Data 2.0 development split decreases from **9.8115374518**
at initialization to **3.9639621581** at the final checkpoint. This is evidence of
learning under that distribution/objective; it does not establish the required
generalization or override the fixed quality gates. Final monitoring perplexity
52.665582 is only meaningful under that same tokenizer and protocol.

One pass exposes about **1.109 targets per parameter**. The accepted pretraining
corpus is encyclopedia-dominated, whereas the fixed matched-language evaluation
uses literary books. Limited exposure and domain coverage are plausible future
hypotheses, not established causes. This one configuration/run cannot separate
data, exposure, tokenizer, optimization and model-capacity effects. Neither more
epochs nor a larger model is proven to solve the failure, and repeated tokens would
not become new unique data. No threshold, decoding rule or metric is revised here.

## Preserved candidate and permanent boundaries

New candidate SHA-256:
`969034537d6b9740e217b436d5bfffbab51db3bbb3d9be79801a71f2ae336331`.
Its 136,355,352-byte tensor file is retained on Mac and Windows. Read-only CPU
inspection verifies finite float32 tensors, 34,087,424 elements and every final
model tensor fingerprint against the last training state. No model forward pass
or optimizer was used. Checkpoint/optimizer/intermediate tensor identities remain
in the inventories; absent Windows payloads were not numerically reconstructed.

This candidate is **complete but unaccepted**, retained as negative experimental
evidence. No final job, final comparison or acceptance declaration exists in the
production return; both evaluator manifests specify development/validation and
no final access. The authorized integrity stage only opaque-hashed reserved bytes.
No final score is available or inferred. Final feedback did not select this result.

The original Phase 17 stays permanently interrupted/failed with its separately
unaccepted recovered model. The successful synthetic diagnostic stays a mechanics
result. Phase 17.1 is a distinct fresh attempt with successful execution and failed
development quality. No artifact is relabeled as accepted Base Model 2.0, promoted
to Assistant 2.0 or published. [Next decision](NEXT_DECISION.md) is prospective only.
