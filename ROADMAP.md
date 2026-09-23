# Roadmap 2.0

## Destination

**LatoS 2.0 will be an independently implemented small language model trained
through the LatoS pipeline that demonstrates measurable learned language and
instruction-following capability on fixed held-out and external evaluations,
runs practically on the available local hardware, provides useful mainstream
interoperability, and ships with reproducible documentation and explicit limits.**

This is a finite plan, not a current capability claim. Phases 0–12 completed the
original engineering roadmap and bounded experiments. Their negative outcomes
motivate a change in emphasis: evaluation → better data → measured CUDA scaling
and training → stronger base quality → useful instruction following → interoperability
→ justified advanced post-training → release. The former open-ended Phase 12+
research extension is retired. Phase 13 is documentation and planning only.

## Acceptance and execution rules

Each phase produces bounded deliverables, actual validation, a separate review
and fixes, an updated [project state](PROJECT_STATE.md), and a reviewed local
commit before explicit phase publication approval. No quality gate can be passed
solely by successful execution, a lower training loss or a feature's existence.

Phase 14 must freeze evaluation versions, scoring, baseline identities and numeric
quality/regression gates **before** model-improvement work. Later phases may refine
resource budgets and experiment details before their runs, but may not lower a
quality threshold after seeing results. A failed gate preserves the run and blocks
the dependent phase. Record a bounded corrective proposal; do not create an endless
retry sequence or claim completion. Changes to the destination, required quality
gates or explicit non-goals require an owner-approved roadmap amendment.

**Phase 14 may start only after Phase 13 is approved, published and remotely
verified, and the owner/master-planning process explicitly authorizes that start.**
Push approval alone does not satisfy this transition gate. Subsequent phases follow
the standing phase workflow unless the owner records another transition gate.

## Phase sequence and gates

| Phase | Deliverable | Exit gate |
| --- | --- | --- |
| 13 — Repository Repositioning & Roadmap 2.0 | Clear public identity, navigable documentation and this finite plan | Honest landing page, evidence preserved, checks/review pass, no learned behavior change |
| 14 — Evaluation Foundation | Fixed evaluation framework and historical baseline | Reproducible scores, contamination policy and prespecified quality/regression gates |
| 15 — Data 2.0 | Materially larger, higher-quality pretraining and instruction corpora | Provenance, terms review, split/dedup audits, composition/token accounting and tokenizer analysis |
| 16 — Model & CUDA Training 2.0 | Measured candidate comparison and selected dense configuration | Correct mixed precision, feasible memory/throughput and bounded training plan on the available GPU |
| 17 — LatoS Base Model 2.0 | Serious pretraining from random initialization | Material learned-quality improvement over the historical base on the fixed evaluation framework |
| 18 — Assistant 2.0 | Improved instruction-tuned model | Measurable held-out instruction following with acceptable language-model regressions |
| 19 — Ecosystem Interoperability | Native-compatible mainstream loading/export | Tested Transformers path and evidence-based local deployment/export decision |
| 20 — Alignment & Tool Learning | Bounded, justified post-training assessment | Controlled comparisons against a competent assistant; honest positive or negative conclusions |
| 21 — LatoS 2.0 Release | Reproducible final model and software package | Quality gates met, supported paths reproduced, cards/results/artifacts/checksums and limits published after approval |

### Phase 13 — Repository Repositioning & Roadmap 2.0

Make the README a project landing page. Remove obsolete identity language from
active branding while retaining materially historical records. Place chronology,
platform scope, experiment evidence and governance in their appropriate documents.
Preserve old reports, metrics, artifacts and releases; explain historical pending
publication snapshots without rewriting their original evidence.

Acceptance: relevant checks and the existing test suite pass; links and package
contents are reviewed; existing tags/releases and historical evidence are unchanged.
No architecture, training behavior, model weights, data or learned-model experiment
changes. No Data 2.0, mixed precision, interoperability, MoE, RL, serving or export
implementation occurs in this phase.

### Phase 14 — Evaluation Foundation

Build evaluation before attempting quality improvement. Freeze a versioned suite
covering held-out language modeling, fixed-prompt generation, instruction following,
regressions and selected external English benchmarks appropriate for very small
models. Select benchmarks independently for task relevance, licensing, difficulty,
scoring reliability and feasible cost; document primary sources and exclusions.
Include simple controls (such as random/chance or trivial task baselines) so scores
are interpretable. Separate deterministic scoring from any qualitative review.

Record exact dataset revisions/splits/hashes, prompt and chat formats, decoding,
seeds, context limits, scoring code, aggregation, uncertainty and resource use.
Retain every fixed sample and failure. Identify contamination risks and keep model
selection validation separate from final held-out acceptance. The two historical
reserved tests stay reserved; any eventual final use needs an explicit protocol,
not routine tuning. Historical development cases remain labeled as already observed.

Exit: reproduce the preserved ~17.3M Phase 5 baseline and relevant SFT/adaptation
baselines on the new suite where inputs are available; report missing artifacts as
blockers. Demonstrate score repeatability within declared tolerances and regression
detection with controlled failures. Freeze numeric minimum scores, meaningful
improvement margins and maximum regressions for Phases 17/18, with rationale and a
final-test access policy, before Phase 15. No invented benchmark scores or quality
thresholds are asserted by this planning phase. Across different tokenizers, use
matched text and a comparable normalized metric (for example bits per byte);
raw token perplexity is comparable only under compatible tokenization/objectives.

### Phase 15 — Data 2.0

Independently select a materially larger and higher-quality corpus for pretraining
and instruction tuning. Retain the Phase 1 corpus, manifests, split assignments,
tokenizer and all evidence unchanged. Use separately versioned new inputs.

Exit: each source has provenance, revision, licensing/terms review, intended use,
acquisition and integrity records; acquired payloads stay outside normal Git history.
Audit exact/near duplicates within and across splits, split documents/source groups
before chunking, and screen for evaluation contamination. Report exclusions and
residual contamination uncertainty, domains/languages/composition, raw and retained
counts, unique and total token counts, and measured quality samples. Fit tokenizer
candidates only on training data; compare compression, coverage and compute/context
tradeoffs. Select or retain the tokenizer with explicit evidence. Dataset acceptance
is a construction/quality audit, not a claim that future training will succeed.

### Phase 16 — Model & CUDA Training 2.0

Measure the available **NVIDIA RTX 4070 SUPER, approximately 12 GB VRAM**, before
choosing a substantially more capable dense model. Investigate a rough 50–80M
parameter region only if measurements support it; this is not a predetermined
architecture. Account for vocabulary/context, optimizer state, activations, batch
size, accumulation, checkpointing and evaluation overhead.

Implement and validate justified mixed-precision CUDA training against float32
references, including finite losses/gradients, stable updates, validation and
save/load/recovery behavior under documented tolerances. Compare a bounded set of
candidate scales on a common short-run protocol; these are feasibility probes,
not the accepted Phase 17 base. Measure synchronized throughput, peak allocated
and reserved VRAM, host memory and checkpoint/storage cost on the actual GPU.

Exit: select a native dense configuration using measured memory headroom, throughput
and early quality evidence; document token exposure, runtime/storage estimates,
precision, seeds, selection rule, maximum attempts and stop/recovery conditions
for Phase 17. Define practical local inference latency/memory acceptance targets
before selecting the release candidate. If available hardware cannot support a
credible quality run, record the resource blocker and propose a bounded revision;
do not substitute a tiny execution demo for the planned training.

### Phase 17 — LatoS Base Model 2.0

Run the planned serious pretraining from random initialization through LatoS.
Use the accepted Phase 15 data/tokenizer and Phase 16 configuration; retain immutable
checkpoints, source/environment identities, logs and failed/interrupted attempts.

Exit: meet Phase 14's prespecified held-out language-quality improvement and
regression gates against the preserved 17,308,032-parameter Phase 5 base. Report
fixed generation samples, external evaluation, comparable tokenization-aware
metrics, total/unique token exposure, resource use and selection rule. Run final
held-out acceptance only under its frozen access policy. Successful execution or
training loss alone cannot pass. If quality fails, Phase 18 stays blocked.

### Phase 18 — Assistant 2.0

Begin only after the new base passes its quality gate. Use substantially improved,
independently constructed/selected instruction data with the established chat
contract 1 or a deliberately versioned successor shared by training and inference.
Document masking, split/template separation, multi-turn behavior and decoding.

Exit: exceed Phase 14's fixed held-out instruction-following thresholds and baseline
margins, including external evaluation and tasks beyond familiar training templates;
meet its language-model regression limits and Phase 16's practical local inference
targets. Retain all samples, failures and checkpoint-selection evidence. Lower SFT
loss or improved stopping alone does not establish a useful assistant.

### Phase 19 — Ecosystem Interoperability

Preserve the independent native implementation as the source of truth. Prioritize
a clean Hugging Face/Transformers-compatible export/load path, standardized
model/tokenizer configuration and documented chat/special-token behavior.

Exit: round-trip parameter/configuration identity where lossless, verified token
IDs and serialization, logits/generation agreement within declared tolerances,
clean-environment loading and regression scores on fixed evaluations. Document
version support and any custom-code requirements. Evaluate GGUF/llama.cpp and/or
Ollama against actual architectural/tokenizer feasibility, maintenance cost and local
value. Implement a justified path if viable; otherwise record a bounded feasibility
rejection and supported alternative. Quantized paths need explicit size, memory,
latency and quality tradeoffs; format creation alone is insufficient. No third-party
model implementation replaces native LatoS.

### Phase 20 — Alignment & Tool Learning

Begin only with a competent Assistant 2.0 baseline. Reuse the historical LoRA,
full-tuning, DPO and tool results as negative baselines, distinguishing their older
model/data scale. Select a bounded method or justified no-additional-method outcome
from observed deficits, data/reward validity and available compute.

Improved LoRA/DPO, GRPO/RLAIF or another method are possibilities, not requirements.
PPO, GRPO, MoE and agentic RL are never additions for feature parity. Any selected
experiment fixes its hypothesis, controls, data, budgets, scoring, regression
limits and stopping rule before execution. Revisit learned tools only after ordinary
instruction following is established: measure parsing, correct arguments/calls,
end-to-end tasks and failure handling independently of scripted controls.

Exit: a reproducible decision backed by controlled comparisons, retaining negative
results and explicit unsupported claims. Promote a post-trained candidate only if
it improves the targeted held-out measure without violating prior quality gates;
otherwise retain Assistant 2.0. A negative or no-go research result can close this
phase but cannot waive the base/assistant release requirements.

### Phase 21 — LatoS 2.0 Release

Package the accepted base/assistant and justified interoperability artifacts with
model/data cards, evaluation versions/results, reproducibility instructions,
supported platforms, checksums, provenance/terms and explicit limits. Verify an
independent clean installation and the supported native/export paths on available
hardware. Establish a permitted, durable artifact destination before publication;
source Git history is not a weights/log backup.

Exit: Phases 17/18 quality gates and practical local execution targets remain met
by the exact release artifacts. Separate engineering correctness, platform validation,
learned capability and unsupported claims. Preserve negative historical and new
results. Publish only the exact reviewed release after its explicit approval;
version/tag/assets belong in that concrete proposal. If core quality gates remain
unmet, report an incomplete Roadmap 2.0, not a successful LatoS 2.0 capability release.

## Hardware and evidence model

Mac Work remains authoritative for development and repository publication.
Windows/RTX 4070 SUPER can provide independent CUDA validation and substantive
training where explicitly planned above; this does not retroactively upgrade old
synthetic validations into full learned-run reproductions. Hosted Linux CPU CI
is separate evidence. Physical Linux stays deferred unless the owner changes that
constraint. Record unavailable execution paths as untested. Use existing resources;
the default new paid-service budget is zero. Do not rent compute or change access.

## Explicit non-goals

MoE, multimodal/vision/audio, diffusion language modeling, linear-attention variants,
elaborate web UI, distributed/multi-node training and feature-for-feature replication
of another project are not LatoS 2.0 requirements. Additions need compelling evidence
and an explicit owner-approved roadmap amendment. PPO/GRPO/RLAIF/agentic RL remain
possible Phase 20 research choices only, subject to its quality and budget gates.

## Historical roadmap and evidence

Phases 0–8 built the foundation, data pipeline, tokenizer, dense decoder, training,
pretraining, SFT, inference and v1.0.0 release. Phases 9–12 added LoRA/merge, DPO,
bounded tools and the fixed context-budget experiment. All are complete within
their recorded scopes; completion did not establish useful instruction following,
preference improvement or learned tools. Phase 12 closure was published at
`c74290ee1208dc6b0e4077ce3c7b2aad262f3c42`.

The [historical roadmap](docs/history/ROADMAP_1.md) is retained as a superseded
planning record. [Evidence index](docs/INDEX.md), [changelog](CHANGELOG.md),
[model card](docs/MODEL_CARD.md) and [data card](docs/DATA_CARD.md) preserve the
results and provenance. Existing reports' pending-publication statements describe
their original snapshots; use [PROJECT_STATE.md](PROJECT_STATE.md) for active state.
No existing tag or release is reassigned by Roadmap 2.0.
