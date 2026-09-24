# Diagnostic implementation review — actual CUDA evidence pending

A separate review pass followed the initial data/state/control implementation and
CPU checks. No independent external reviewer or CUDA diagnostic execution is claimed.
All selected-scale optimization is confined to the forthcoming bounded Windows run.

Findings and decisions:

- An initial draft used separately trained prefixes. That can confound restart
  fidelity with pre-checkpoint run-to-run divergence. Before any CUDA optimization,
  change to a shared reference checkpoint: 997 uninterrupted updates versus its
  exact 512-step split plus a new-process 485-update continuation. This reduces the
  frozen study to 2,964 physical updates across two pairs. No result-driven change.
- Do not clear CUDA allocator caches at the reference split. The uninterrupted
  reference keeps its warm process/cache history; the matching replay starts in a
  new process. Discard temporary readback objects without manually equalizing the
  allocator state, which could mask the restart sensitivity under investigation.
- Loss agreement alone can hide optimizer/sampler differences that amplify later.
  Capture per-tensor model and AdamW-state hashes, parameter-index mapping, group
  settings, sampler/counters, global RNG and input signatures after every update.
  Require exact write/readback/new-process restoration, then compare state after
  each continued update. A positive correction also requires reproducing the legacy
  loss failure; a passing control is inconclusive, not proof of the old cause.
- The checkpoint format predates some numerical controls. Preserve the imported
  training package exactly and bind new process controls in immutable diagnostic
  sidecars. Set workspace before importing Torch; never load a historical model.
  A successful prospective policy would need explicit future checkpoint/runtime
  binding in Phase 17.1, after separate authorization; no historical metadata migration.
- Use only training-length metadata and generated token IDs. Repeated full-size
  synthetic cache construction on Mac is byte-identical, and the executed shuffle
  reproduces all 15,940 lengths. Freeze its binary hashes before CUDA runs. The dummy
  validation identity required by the checkpoint API is never evaluated. No real
  corpus, tokenizer, held-out/final payload or old model is in the transfer.
- Enforce the one-launch receipt, fixed five-job schedule (preparation plus four
  optimizer workers), append-only supervisor journal, active deadline, artifact
  cap/reserve and no retries. The watchdog tests kill a real sleeping child and
  reject a failed child without performing any optimizer update.
- Check the remaining budget before dispatch as well as during each worker. Normalize
  recorded numerical controls through JSON so tuple/list-valued backend properties
  compare consistently with their persisted sidecars. No numerical setting is changed
  by this representation normalization.
- A checkpoint directory can be atomically renamed while the supervisor counts
  bytes. Tolerate vanished paths in that scan, count the final path on the next
  scan, and retain pre-write checkpoint space reservation. Other I/O errors still
  fail closed. A real rename fixture covers this race without optimizer work.
- Initial watchdog tests exposed a missing test import; both failed attempts and
  logs were retained. Fix the import, then repeat the focused tests in development
  and the existing non-editable wheel environment. Formatting/import-order findings
  were fixed. A documentation-inspection command initially used system Python with
  no Torch; rerunning with the existing project interpreter resolved that read-only
  lookup. No model, dependency or experiment was changed by it.
- All preflight tests forbid `Trainer.update`. They use only tiny CPU state fixtures,
  data construction, pure comparisons and process controls: zero optimizer updates.
  Do not run unrelated historical evaluation tests under this diagnostic authorization.
- Explicitly preserve causal limits. Matching tensor/RNG state after loading rules
  out a lossy snapshot in the measured pair; it does not identify a unique historical
  kernel. The two process policies test a combined prospective correction. Neither
  synthetic learning nor lower loss establishes language quality or Phase 17.1 acceptance.

The actual results are in `validation.json` and `data-validation.json`. Original
Phase 17 reports, failed artifacts and frozen contracts remain unchanged. This
checkpoint is an implementation/handoff, not a diagnosis already established on
RTX hardware. Mac review of returned evidence is required before preparing any
fresh Phase 17.1 proposal. Full Phase 17.1, Phase 18 and remote publication remain
unauthorized.
