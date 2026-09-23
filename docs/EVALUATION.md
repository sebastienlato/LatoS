# Evaluation foundation

LatoS evaluation version 1 measures separate properties: language likelihood,
observable generation behavior, objective instruction success, external question
answering and resource use. It produces no composite “LatoS score.” Implementation
correctness does not establish learned quality. See the [fixed plan](../experiments/phase-14/PLAN.md),
[protocol and gates](../configs/evaluation/protocol-v1.json),
[retrospective report](../experiments/phase-14/REPORT.md) and
[Roadmap 2.0](../ROADMAP.md). Original Phase 5–12 evidence remains unchanged.

## Reproduce

The optional evaluation extra adds Apache Arrow's Parquet reader; no training or
model API dependency is introduced. The existing locked training environment is
otherwise retained. From the repository root:

```sh
uv sync --locked --extra evaluation
uv run --locked --extra evaluation python -m latos.evaluation acquire
uv run --locked --extra evaluation python experiments/phase-14/run.py --output-dir outputs/phase-14-repeat
```

The last command requires all five preserved full models and their matching
English tokenizer at the paths in [models.json](../experiments/phase-14/models.json),
and the original prepared English corpus. It runs on MPS by default, in separate
processes, with a 30-minute timeout per model. It includes a base repeat. Acquisition
downloads only the two pinned validation files (141,823 bytes total), verifies size
and SHA-256, checks existing files, and refuses corruption. No training/test split,
retrieval corpus, remote code or pretrained model is downloaded. Evaluation itself
is offline. A clone does not contain the historical learned artifacts; missing
artifacts are an explicit failure, never a random-model substitution.

For one native model, supply its exact hash explicitly:

```sh
uv run --locked --extra evaluation python -m latos.evaluation run \
  --model-dir outputs/phase-5-english-pilot/step-00003000/model \
  --weights-sha256 f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa \
  --tokenizer-dir artifacts/tokenizers/english-bpe-v1 \
  --device mps --output outputs/base-evaluation-new
```

Use a new output directory every time. CPU is the generic runner's default; explicit
unavailable devices fail. Future native checkpoints use the same interface with
new model/hash/tokenizer arguments. There is no Transformers/export adapter.
Model-only loading verifies finite float32 weights, shapes, config, vocabulary and
tokenizer hash. Old optimizer-state recovery is neither needed nor bypassed.

Compare a candidate to the historical base for Phase 17:

```sh
uv run --locked --extra evaluation python -m latos.evaluation.compare \
  outputs/base-evaluation-new outputs/candidate-evaluation-new \
  --stage base --output outputs/base-comparison-new.json
```

For Phase 18, reference the accepted Phase 17 base, select `--stage assistant`, and
provide `--historical` pointing to the strongest historical instruction result
from the fixed five-artifact panel, declared before candidate scoring. Ties may use
any tied artifact, recorded explicitly. Comparisons require identical protocol,
text/suite identities, runtime, backend and batching; rerun paired references when
those conditions change. Native tokenizer identities may differ for matched BPB
and task metrics; raw token perplexity is marked incomparable in that case.

## Held-out language modeling

Development uses all 1,030 paragraphs from the original author/document-disjoint
validation books: *Alice's Adventures in Wonderland* and *The War of the Worlds*.
Their exact prepared split hash is pinned. These books have been observed during
historical development; they are not a fresh test. Phase 1 normalization and dedup
are unchanged. [Data provenance](DATA.md), the [data card](DATA_CARD.md), manifests,
and original local raw notices retain the applicable terms.

Two explicit objectives run on the same text:

- **Legacy continuity:** encode each whole paragraph with BOS/EOS; windows have
  at most 256 IDs and one overlapping token. Every next-token transition scores
  once, including EOS, with context reset at seams. Sum NLL across valid targets,
  then divide by their total. This recreates the historical objective; independent
  float64 reduction can differ slightly from the old float32 training evaluator.
- **Matched-text BPB:** split each paragraph into consecutive spans of at most
  192 UTF-8 bytes, never splitting a Unicode scalar. Boundaries depend only on
  text, not tokenization. Encode each span losslessly with BOS, without EOS.
  Score each content token once from its preceding logit. Sum NLL, divide by
  `log(2) * exact UTF-8 byte count`. Every original byte belongs to exactly one
  span. Report per-paragraph and per-book totals as well as corpus totals.

A byte-level codec requires at most 192 content tokens for these spans, so the
protocol fits the historical capacity. New codecs must pass lossless round trips
and the same context bound. This compares canonical token-sequence code lengths
on the same segmented text, not a marginal probability over all possible token
segmentations or a tokenizer-neutral universal text probability. It is meaningful
as a fixed compression-style comparison with declared segmentation and BOS
semantics. It does not prove performance with long contexts or other domains.

NLL is in natural logarithm units. Perplexity is `exp(total NLL / targets)` and is
comparable only for the same tokenizer, text, boundary policy and objective. Never
compare legacy perplexity with matched-span perplexity or across tokenizers.
Uniform token probability supplies a transparent analytical control, not a trained
model. Existing historical train-only unigram evidence remains in Phase 5.

The evaluator directly reads logits, excludes position zero as a target, and
matches target position t with logit t−1. Explicit masks exclude question/prompt
positions. Right padding occurs after valid causal positions and never scores.
Float32 logits transfer to CPU before float64 logsumexp and summation. Token-weighted
aggregation avoids overweighting short examples. Nonfinite logits, malformed masks,
empty inputs, identity mismatches or invalid context fail rather than produce a
plausible score. No training objective code or labels are reused to implement this
arithmetic.

## Generation and instruction behavior

[Development suite](../configs/evaluation/suite-v1.json): 24 original prompts in
12 categories, including narration, explanation, modern situations, uncertainty,
Unicode, factual continuation and consistency. All raw text, generated token IDs,
stop reasons and prompt lengths are retained. Greedy cached decoding: temperature
0, top-k 0, seed 0, maximum 64 new tokens, prompt plus budget <=256. The existing
sampler suppresses PAD/BOS/UNK; that is part of the versioned decoding contract.
No resampling, prompt search, cherry-picked outputs or subjective judge is used.

Empty output, EOS, replacement characters, word count and repeated word-trigram
fraction are descriptive observations, not a complete quality rubric. Repetition
is `1 - unique trigrams / all trigrams`, zero for fewer than three words. Short
or meaningless responses can evade that diagnostic; objective instruction scores
and retained text are essential. Human review can describe coherence/factual errors
separately, but cannot replace or silently modify numeric results.

The 96 instruction cases have 16 cases in each category: structured output,
transformations, extraction, constrained responses, basic QA, and multi-step work.
Each category includes multiple task families. Shared chat contract 1 formats
user input and the assistant prefix, with no added system prompt. Exact scoring
strips outer whitespace only; JSON scoring permits whitespace/key-order differences
but rejects duplicate keys, trailing prose, nonfinite values and extra fields.
Types matter: Boolean true does not satisfy integer 1. Correct text can score even
when decoding reaches its token budget; stop behavior is reported independently.
No substring/keyword heuristic, answer repair, paid API or model judge is involved.

The inputs/answers are original AI-assisted project fixtures under MIT, not human
validation. They are small and partly templated, not representative of open-ended
assistant use or safety. Historical 32-case results are retained separately and
never silently redefined. The 48 additional [final cases](../configs/evaluation/final-suite-v1.json)
use different task families/wording and remain unscored in Phase 14. They still share
broad skills, and public source cannot guarantee no exposure. Empty, constant-yes
and prompt-echo controls are scored separately from real model results. Gold
fixtures test scoring mechanics only.

## External selection and scoring

The suite contains **ARC-Easy** (570 validation questions) and **ARC-Challenge**
(299). [Clark et al., 2018](https://arxiv.org/abs/1803.05457) describes the original
benchmark; [AI2's dataset documentation](https://huggingface.co/datasets/allenai/ai2_arc)
provides schema, counts and CC BY-SA 4.0 metadata. The
[pinned manifest](../configs/evaluation/external-v1.json) records revision
`210d026faf9955653af8916fad021475a3f00453`, download URLs, hashes, bytes and provenance.
Questions remain in ignored storage under their own terms, not the repository's
MIT license. No question text is redistributed in tracked Phase 14 results.

Easy tests elementary science/factual QA with a feasible short context and an
interpretable chance floor. Challenge adds a harder difficulty partition selected
by the original authors, useful for diagnosing weak reasoning/generalization.
They share a source and domain, so this is not two independent domains or broad
commonsense coverage. Challenge's regression is monitored; it has no separate
above-chance promotion requirement. A tiny historical model may remain near chance.

For each original-order option, score its entire text after
`Question: {question}\nAnswer:` plus one leading space. Joint encoding must preserve
the prefix token boundary; otherwise fail instead of guessing which BPE token owns
the answer. Sum **only** answer-token log probabilities, with no EOS. Primary choice
is the largest sum. A secondary score divides by UTF-8 answer bytes including the
leading space, exposing length sensitivity. Ties choose the first option. Neither
free-generation letters nor likelihood of the label A/B/C/D approximates this rule.
Every case retains option log probabilities, target counts, predictions, gold index,
content hash and failures. Choice order is not shuffled based on model behavior.

Uniform chance is the mean of `1 / number of choices`, not blindly 25%; a first-choice
control exposes answer-order imbalance. Accuracy uses all original cases. Overflow
is retained as incorrect and blocks a complete acceptance result. Wilson intervals
are descriptive; matched paired case bootstrap uses seed 140 and 2,000 samples.
Instructions resample families as clusters. These are uncertainty summaries under
limited sampling assumptions, not corrections for contamination, question
correlation, repeated selection or multiple comparisons. With two LM books, no
misleading paragraph-IID confidence interval is reported.

Alternatives were considered independently. OpenBookQA has similar science scope
and its [author repository](https://github.com/allenai/OpenBookQA) licenses code,
while [dataset metadata](https://huggingface.co/datasets/allenai/openbookqa) leaves
data licensing unknown; an open data-license question makes it unnecessary here.
WinoGrande has attractive short commonsense contexts, but its
[author repository](https://github.com/allenai/winogrande) separates test labels into
a leaderboard, and inspected data documentation does not resolve a clear standalone
data license. Neither was acquired. Large multitask collections and long-form
reasoning/generation judges add cost and scoring ambiguity beyond this phase.
Selection was not based on another model project's benchmark choices.

These are **LatoS zero-shot validation protocols**, not claims of exact
leaderboard comparability. Different prompts, token boundaries, BOS handling,
length normalization, splits or selection policies require explicit distinctions.

## Contamination and final acceptance

The initial [audit](../experiments/phase-14/contamination.json) checks historical
book/instruction training against development prompts/questions/choices using
casefolded full-text equality and contiguous 13-word overlap. Short strings and
semantic paraphrases remain limitations. The audit does not open reserved files.
Generic short answers can legitimately appear in training; no lexical audit proves
absence of memorization. Development fixtures are already exposed to this project.

Phase 15 must exclude all evaluation sources, prompts, answers, generated outputs
and close variants from training, including benchmarks' train splits. Preserve
source-level provenance and record exact/near-duplicate checks against evaluation
texts and historical reserved source IDs before accepting a corpus. Store any
controlled reserved comparison in the data audit without exposing examples to
model development. Do not use Phase 14 scores to tune historical models. New data
and checkpoints need their own contamination status; the old audit is not inherited.

The runner ordinarily opens only `validation.jsonl`. Final access requires
`--acceptance-plan PATH` and a complete selection declaration. The declaration
records schema_version 1, phase (17 or 18), candidate_weights, reference_weights
(list), development_comparison and its SHA-256, contamination_audit and its SHA-256,
and the literal declaration `locked candidate; no training or selection from final feedback`.
Paths are relative to the plan's directory. The comparison must say passed=true,
match candidate identity and stage; the reviewed audit must bind the same candidate
and explicitly say accepted_for_final=true. The plan and all accesses are retained
with the run. This is auditable process control, not an access-security boundary.
The owner/master-planning transition gates still apply.

After development success, lock one candidate and the paired references, then make
one final acceptance comparison per phase. Phase 17 reads the original English test
under the same legacy/matched protocols and development behavioral suite. Phase 18
also requires `--suite configs/evaluation/final-suite-v1.json`; evaluate the fixed
historical panel on that suite only at this final checkpoint to establish its
strongest retrospective instruction reference. The old 32-case instruction test
remains reserved. A final result must satisfy the same numeric gates, plus reviewed
training exclusion and resource/selection evidence. Negative final outcomes block
dependent phases. Do not use them for another tuning cycle under the same holdout;
a fresh design requires an explicit roadmap amendment. Preserve every failed run.

## Prespecified improvement and regression policy

The [machine-readable gates](../configs/evaluation/protocol-v1.json) were fixed
before retrospective scoring, not fitted to historical successes. Phase 17 requires
at least a 10% matched-BPB improvement, no book over 2% worse, ARC-Easy raw >=35%
and a >=5-point gain, with its descriptive Wilson lower bound above chance.
Phase 18 requires >=60% objective instruction success, >=40% per category and a
>=30-point gain over the fixed strongest historical reference. Its BPB regression
may be at most 10% overall and 15% per book versus the accepted Phase 17 base.
Both allow at most 3-point external raw/byte-normalized accuracy regressions,
5% empty continuations, 25% average repeated-trigram fraction, and a 10-point
repetition increase. No metric tradeoff can hide a failed mandatory bound.

Ten percent language improvement is a material effect above numerical noise;
5 points on Easy is roughly 29 extra correct cases, while the absolute floor avoids
promoting a marginal change around chance. Sixty percent instruction success and
category floors require multiple capabilities rather than one easy template.
The assistant LM allowance acknowledges a possible specialization tradeoff while
bounding historical-style regressions. Generation checks catch gross collapse,
not subjective quality. These are conservative project policy minimums, not
scientifically universal definitions of intelligence or release readiness.

Phase 16 separately fixes measured practical latency/memory and training budgets
before choosing a model. Phase 17/18 must pass those too. A threshold amendment
requires owner approval before new improvement attempts, with old results preserved.
The implementation or a lower training loss cannot waive a quality gate.

## Evidence and reproducibility limits

Each run retains a manifest with source commit/dirty state, the actual source
snapshot and hashes, protocol/suite/input identities, model/config/tokenizer hashes,
chat contract, device, versions, seed, decoding and batching. Raw LM rows, MC choices,
instruction outputs and generation samples are separate from summary JSON. The
inventory hashes all completed files; a verifier rejects incomplete or modified
records and recomputes aggregates. Hashes detect changes, not malicious replacement
of both records and hashes. Handled exceptions leave failure.json. Hard termination
can leave an incomplete run without that file; absence of a complete inventory
never counts as success. Existing output directories are refused.

Same-host fixed-backend repeatability uses per-row NLL atol 1e-5 / rtol 1e-6,
exact integer/prediction/greedy-ID equality and fixed inputs. It is not a guarantee
of bitwise agreement across devices or library versions. CPU numerical tests and
Mac MPS historical runs are separate evidence. CUDA is available as an explicit
path but Phase 14 CUDA execution is untested; no Linux run is claimed.

Runtime includes loading, source capture and evaluation, but not imports or final
serialization/inventory. Per-token inference timings exclude loading. RSS is
process-lifetime high-water, not incremental evaluation memory. MPS allocated/driver
values are end-boundary observations, not peaks or additive physical memory pools.
These single-host observations are not production performance claims. Large/generated
files and acquired data stay ignored; tracked summaries identify them without
pretending Git publication backs up local weights or raw results.
