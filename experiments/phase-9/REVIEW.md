# Separate Phase 9 review

Performed after implementation and initial validation, by a distinct review pass
in the same Work task. No external reviewer or independent agent is claimed.

Reviewed: adapter equation/initialization, gradient and freezing boundaries,
optimizer membership, base/tokenizer binding, tensor validation, non-overwriting
save and non-mutating merge, shared chat masking, comparison fairness, artifact
preservation, publication scope and claims. Prior platform/quality limits remain.

## Findings and disposition

1. **Comparison completeness:** matching two traces alone could accept equally
   incomplete traces. Added explicit configured-final-step, contiguous-update,
   count and exposure-summary checks; integration test rejects an altered summary.
2. **Independent verifier mode:** constructing loaded parameters under a function-wide
   inference-mode decorator creates tensors without version counters. The cache
   correctly rejected them. Move inference mode to numerical forwards, preserving
   cache mutation detection. Retain the original failed verification log.
3. **CPU merge acceptance:** full learned adapter failed the planned 1e-5 CPU bound
   near zero despite tiny correctness tests passing. Preserve failure; explicitly
   amend full float32 merge to 1e-4 and add an independent float64 equation check.
   Propagated the amended bound to the training CLI and asserted it in integration
   tests. Historical CPU cache tolerance remains 1e-5. Scope/error measurements are in
   the report rather than implying the first threshold passed.
4. **Performance isolation:** preliminary timing overlapped the suite. Retain it,
   use a separate repeat with validation idle and unchanged training controls.
   Report all samples and avoid method-optimality/production-speed claims.
5. **Validation launch environment:** direct pytest lacked CLI on PATH. Correct the
   invocation with the locked runner; preserve the seven launch failures and the
   subsequent complete suite. No test was removed or weakened.

Follow-up checks cover CPU/MPS full saved-artifact merge/cache through capacity,
float64 diagnostic, complete suite and installed-wheel use. No remaining actionable
finding is identified within the bounded scope. The post-observation CPU merge
amendment, single-run performance limits and lack of adapter optimizer resume are
explicit limitations, not hidden completion criteria.
