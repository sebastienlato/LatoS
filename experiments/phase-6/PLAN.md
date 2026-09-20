# Phase 6 predeclared experiment

Recorded before optimization, 2026-09-20. Hypothesis: a bounded SFT run can lower
assistant-target validation loss and improve exact completion of short English
requests. A negative instruction result still measures the experiment. No useful
assistant claim follows from successful execution.

- Initialize only model weights from selected Phase 5 update 3,000, SHA-256
  `f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa`.
  New AdamW optimizer, new seed-61 shuffle, no pretraining resume.
- Retain tokenizer SHA-256
  `7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
- Original synthetic data: 48/8/8 word source groups, four conversations per group,
  192 train / 32 validation / 32 reserved test. Group allocation precedes formatting.
  Template families: copying, label extraction, first letter, two-turn recall.
  Vocabulary groups are disjoint; templates and generic answers overlap by design.
  This is within-template lexical generalization, not an unseen-task benchmark.
- `data.py` is original Codex-authored project material under MIT, with no external
  dataset, paid service, or model API. Deterministic expansions and manifest go to
  ignored `data/processed/english-instructions-v1`. Test bodies are never loaded
  by tuning/evaluation. The generator necessarily defines test material, but test
  predictions/loss are reserved from this phase.
- Shared segmented chat serialization, assistant content plus EOS targets only.
  Sequence length 256, within model capacity 512. No packing or truncation.
- Resource calibration: 10 updates on the selected base in a separate output;
  discard the calibrated weights. The main run is fixed at 200 updates, batch 4,
  accumulation 2, float32, learning rate 0.0003, warmup 10, cosine minimum ratio 0.1,
  weight decay 0.01, gradient clipping 1. Final update 200 is selected in advance.
  No hyperparameter sweep or validation-driven early stopping.
- Evaluate every validation conversation with teacher-forced assistant-only loss
  before/after. Generate each final response from identical gold conversation
  prefixes with shared formatter; greedy decoding, 32-token cap, EOS stop, seed 71.
  Score exact case-sensitive decoded text after outer whitespace stripping; report
  EOS separately and per-family counts. Recall gets gold prior assistant history,
  so it is not an autonomous multi-turn dialogue assessment. Report every sample.
- Regression: all-target English LM loss on the **full existing validation split**,
  with the same 256-token windows before/after; never read English test text.
  Report degradation even when instruction loss improves. Objectives differ and
  their raw losses must not be equated.
- Preserve initial and update 100/200 recovery checkpoints, full metrics, runtime,
  identities, sampling and memory evidence. Record any failed run separately.
  Require tiny CPU recovery equality and actual MPS recovery replay within stated
  tolerance; no general CUDA or cross-device determinism claim.
- No Phase 7 interface/cache work. No new paid services. Stop after separate review,
  fixes, checks, and a local commit for explicit Phase 6 publication approval.
