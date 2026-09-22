# Separate Phase 10 review

A separate pass in the same Work task followed implementation and the first learned
run. Reviewed DPO equation/gradient, shifted targets, final-response boundaries,
reference ownership, split handling, exposure accounting, experiment identities,
CLI failures, snapshot round trips and interpretation of held-out results.
No external reviewer or unperformed platform run is implied.

Findings fixed:

- The wrapper originally appended protocol and runner copies after the artifact
  inventory. Capture both before evaluation, put their hashes in the plan, and
  include both in the inventory. The verifier now requires all core artifacts.
  Preserve the first run at `outputs/phase-10-dpo`; repeat the unchanged training
  protocol once at `outputs/phase-10-dpo-reviewed` after the metadata fix. This is
  not a hyperparameter search or quality-based checkpoint choice.
- The first independent verifier tried to convert MPS logits to float64 before
  transferring them to CPU, which MPS rejects. Transfer to CPU first; use Python
  scalar log-sum-exp only after float32 model inference. Retain the original
  TypeError log. This is a scoring diagnostic, not mixed-precision training.
- Strengthen verification with scalar loss/gradient checks, per-token probability
  sums, absence of prior-turn/padding gradients, invalid/nonfinite inputs, CPU
  repeatability, missing test payloads, immutable reference, injected failures,
  inventory corruption rejection and independently replayed training exposures.
- Keep reference-relative wins separate from raw chosen-likelihood wins; the
  initial relative score is entirely ties. Preserve worsening held-out preference
  loss/ranking and zero exact replies. Report the slight English recovery relative
  to SFT alongside its remaining degradation relative to the Phase 5 base.
- Document training-prompt reuse from SFT, shared validation prompts, synthetic
  labels, familiar templates, length sensitivity and lack of alignment evidence.

Final validation and preservation records are in [validation.json](validation.json)
and [verification.json](verification.json). No actionable finding remains within
the bounded experiment scope. No adapter resume, Phase 9 merge contract, shared
chat serialization, reserved-test evaluation policy or earlier evidence is changed.
