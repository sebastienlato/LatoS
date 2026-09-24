# Documentation and evidence

Start with the [README](../README.md) for installation and a tiny offline workflow.
[PROJECT_STATE.md](../PROJECT_STATE.md) records the active checkpoint and publication
gate. [Roadmap 2.0](../ROADMAP.md) describes planned work, not existing capability.

## Usage and engineering

| Topic | Guide |
| --- | --- |
| Installation and platforms | [Setup](SETUP.md); [historical environment measurements](ENVIRONMENT.md) |
| Data and provenance | [Data pipeline](DATA.md); [data card](DATA_CARD.md) |
| Tokenization | [Tokenizer](TOKENIZER.md) |
| Native architecture | [Dense model](MODEL.md); [model card](MODEL_CARD.md) |
| Training | [Training and recovery](TRAINING.md); [pretraining](PRETRAINING.md); [selected CUDA configuration and budget](TRAINING_2.md) |
| Instruction tuning | [Chat contract and SFT](INSTRUCTION_TUNING.md) |
| Local inference | [Streaming, cache and terminal chat](INFERENCE.md) |
| Adaptation and preferences | [LoRA/merge](ADAPTATION.md); [DPO](PREFERENCES.md) |
| Bounded tools | [Protocol and separate evaluation](TOOLS.md) |
| Stable release | [v1.0.0 reproduction](RELEASE.md); [historical release notes](../experiments/phase-8/RELEASE_NOTES.md) |
| Development | [Working instructions](../AGENTS.md); [decisions](../DECISIONS.md); [attribution](../THIRD_PARTY.md) |

## Retained experiments

Reports and adjacent machine-readable results, samples, inventories and reviews
retain the original evidence. Some reports say publication was pending when
written. Those are historical snapshots; Phase 12 closure is now published and
Phases 0–12 are complete within their documented scopes. That status does not
change an experiment's negative outcome or platform limits.

| Phase | Evidence | Interpretation |
| --- | --- | --- |
| 0 | [Environment](ENVIRONMENT.md) | Package/backend foundation, no learned model |
| 1 | [Corpus report](../experiments/phase-1/REPORT.md) | Bounded English books, documented splits and lexical deduplication |
| 2 | [Tokenizer report](../experiments/phase-2/REPORT.md) | Independently fitted codec; compression is not model quality |
| 3 | [Model report](../experiments/phase-3/REPORT.md), [closure](../experiments/phase-3/CLOSURE.md) | Numerical correctness on random models |
| 4 | [Training report](../experiments/phase-4/REPORT.md), [closure](../experiments/phase-4/CLOSURE.md) | Tiny fixture overfit/recovery, not broad learning |
| 5 | [Base report](../experiments/phase-5/REPORT.md), [closure](../experiments/phase-5/CLOSURE.md) | 17,308,032 parameters; validation loss 9.089003 → 4.731898; generated text remains limited |
| 6 | [SFT report](../experiments/phase-6/REPORT.md), [closure](../experiments/phase-6/CLOSURE.md) | Exact replies 0/32; English loss 4.731898 → 5.653422; no useful instruction following |
| 7 | [Inference report](../experiments/phase-7/REPORT.md), [closure](../experiments/phase-7/CLOSURE.md) | Streaming/cache/CLI mechanics; no improvement in learned quality |
| 8 | [Release report](../experiments/phase-8/REPORT.md), [review](../experiments/phase-8/REVIEW.md) | Reproduction and separately distributed tiny fixture |
| 9 | [Adaptation report](../experiments/phase-9/REPORT.md), [closure](../experiments/phase-9/CLOSURE.md) | LoRA and full tuning both 0/32 exact replies; no useful assistant established |
| 10 | [DPO report](../experiments/phase-10/REPORT.md), [closure](../experiments/phase-10/CLOSURE.md) | Full Mac ranking 16/32 → 15/32, exact 0/32; English loss better than SFT but worse than original base |
| 11 | [Tool report](../experiments/phase-11/REPORT.md), [closure](../experiments/phase-11/CLOSURE.md) | Original Mac base/SFT/DPO each 0/16 valid JSON; no learned tool use |
| 12 | [Context report](../experiments/phase-12/REPORT.md), [portability correction](../experiments/phase-12/PORTABILITY_CORRECTION.md), [closure](../experiments/phase-12/CLOSURE.md) | All 44 Mac blocked retries gain replies at 512; still no valid JSON, calls, tasks or learned failure handling |

Historical SFT/LoRA/DPO task sets are narrow and share templates; these results are
not broad benchmark measurements. Scripted tool successes bypass model generation.
The original CPU merge tolerance failure remains retained; the later explicit
merge tolerance amendment is documented in Phase 9. Both reserved real test sets
remain excluded from routine tuning/evaluation.

## Platform scope

Mac CPU/MPS evidence includes the retained full learned experiments. Windows/RTX
4070 SUPER evidence is owner-reported and bounded by each closure report; it did
not reproduce those full Mac experiments. In Phase 12, Windows used a fixed random
synthetic fixture: **16 initial context blocks removed**, zero replies at 256 to
one at 512, 1,024 CUDA tokens, 0/16 JSON, no calls, 0/12 tasks, 0/4 learned failure
handling. This is distinct from the Mac's **44 learned retry transitions**.

Hosted Linux CPU CI was inspected separately within each recorded scope. Physical
Linux remains deferred. Platform passes and scripted controls do not establish
learned quality. No new Windows/CUDA or hosted Linux run is implied by Phase 13.
Roadmap 2.0 permits substantive Windows/CUDA training in its planned training phases;
that is future work, not a reinterpretation of historical evidence.

## Planning and chronology

- [Changelog](../CHANGELOG.md): release/change chronology.
- [Historical roadmap](history/ROADMAP_1.md): superseded Phase 0–12 plan.
- [Phase 12 checkpoint snapshot](history/PHASE12_CHECKPOINT.md): prior detailed state,
  including platform boundaries and historical publication-pending wording.
- [Retired Phase 13 research handoff](PHASE13_HANDOFF.md): retained historical planning;
  its kickoff is superseded by the owner's repository-repositioning direction.
- [Phase 13 validation and review](PHASE13_VALIDATION.md): documentation-only scope,
  checks and preservation evidence for this checkpoint.

## Roadmap 2.0 evaluation

[Evaluation methodology and reproduction](EVALUATION.md) ·
[Phase 14 retrospective results](../experiments/phase-14/REPORT.md) ·
[Fixed evaluation and quality gates](../configs/evaluation/protocol-v1.json)

## Data 2.0

[Method and source rights](DATA_2.md) · [Phase 15 results](../experiments/phase-15/REPORT.md) ·
[Phase 16 local inputs](PHASE16_INPUTS.md). These are data/codec artifacts; historical
learned models and fixed evaluations remain unchanged.

## Phase 16 — measured CUDA configuration

[Completed analysis](../experiments/phase-16/CLOSURE.md), [independently verified Windows evidence](../experiments/phase-16/windows-verification.json), [selection](../experiments/phase-16/selection.json) and [full-corpus layout](../experiments/phase-16/full-corpus-layout.json). Feasibility and a bounded training plan are established; learned-quality gates and Phase 17 remain pending. The original [Mac handoff](PHASE16_WINDOWS_HANDOFF.md) is retained as its historical protocol.
