# Separate Phase 17.1 proposal review

A separate Mac review pass followed diagnostic reconstruction and the first
proposal draft. This is specification review within the authoritative task, not
an independent external reviewer or validation of an unimplemented controller.

- The positive diagnostic meets the unchanged combined-policy rule. It does not
  isolate the historical cause; the proposal uses the complete tested policy and
  makes no claim that one switch alone is sufficient. Phase 17 remains failed.
- The proposal starts randomly at seed 160 with an empty optimizer. Neither failed
  nor synthetic/probe weights are eligible. Model, accepted data/tokenizer, full
  one-pass exposure, optimizer, partial tail and fixed quality gates are hash-bound
  to the prior contracts. No quality threshold is changed or justified by this result.
- The old monitored-ledger replacement must not be reused. Specify a separate
  append-only supervisor and immutable receipts with persistent launch exclusion,
  hard limits and failure preservation. This addresses the observed administrative
  failure without claiming the original PowerShell process caused it.
- Determinism settings absent from the native checkpoint format need an additional
  mandatory immutable policy binding. The native core/loader remain unchanged;
  controller checks must reject absent sidecars and old checkpoints. A checkpoint
  is complete only when tensors, policy and readback all succeed. This avoids
  counting an incomplete policy write as a resumable state.
- The maximum lost interval includes interruption during checkpoint writing: use
  1,000 rather than 999 updates. Two conditional external resumptions yield at most
  9,485 physical serious updates, with repeated/unknown work separately charged.
  Budget availability does not waive numerical or correctness failures.
- Record same-Windows learning rates before training and compare exactly throughout.
  The disclosed Mac schedule arithmetic issue is not a reason to relax replay
  rates, full-state equality or the unchanged `1e-5` loss gate.
- Replace reliance on the old short training-time estimate with the measured
  instrumented diagnostic estimate (~61 minutes for updates alone). Keep the
  two-hour training/preflight/recovery cap and one-hour evaluation reserve. The
  proposal cannot promise that all allowed resumptions fit; overruns stop.
- The initial preflight description left tiny fixtures for later selection. The
  final JSON now fixes architecture, counts, seeds, token formula, optimizer,
  eight-update reference and five-update replay. At most 52 tiny CUDA updates
  across checkout/existing wheel; no additional selected-scale diagnosis or tuning.
- Keep paired fixed development and final distinct. The existing evaluator's
  numeric phase-17 schema must be accompanied by the new attempt identity. Final
  access still requires passing development/resource gates, independent Mac review
  and the exact locked-candidate/contamination declaration. No final score has been
  used by Phase 17, and no new final payload was opened during this preparation.
- Avoid implying an executable transfer exists: the checked deliverable is a
  prespecified proposal and preparation handoff. Controller implementation, focused
  tests, source freeze and Windows preflight are explicit prerequisites after
  separate owner authorization. No runnable full-training bundle was rebuilt or
  training begun. Review of the future code remains mandatory before dispatch.

The final proposal has no identified unresolved scientific-choice issue. Its
material remaining limitation is implementation/validation of the specified
controller on the existing Windows runtime before any serious execution. If that
reveals a need to alter the recipe, controls, budgets or acceptance gates, return
for renewed approval; never quietly substitute a changed attempt.

Local artifact consistency checks verify the contract's input/config hashes,
unchanged inherited training values and gates, and the 52-update preflight and
9,485-update serious-work bounds. Actual validation is recorded in
[return-validation.json](../recovery-diagnostic/return-validation.json). No full
runtime suite, CUDA training, inference/quality evaluation, hosted CI or physical
Linux execution was performed in this proposal task.
