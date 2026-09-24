# Separate Phase 17 failed-experiment closure review

A distinct review pass followed independent evidence reconstruction and the first
artifact inspection. This was performed within the Mac task, without an external
reviewer. No returned administrative source was executed. The review itself did
not run a serious optimizer update or any fixed acceptance scoring.

- Bind the recovery return to the owner's exact SHA-256, verify all included file
  bytes, match the training snapshots to Git commit `0fa94ad580110fd2dc7aaa2aa3560950abe73a02`,
  and match the administrative bundle to both its original ZIP and Git commit
  `1436ec27a9ef42c6a72c7625ba9b4414cdf82daf`. An initial check trusted the previously
  verified local bundle; the strengthened check now independently binds each file
  back to Git as well. The extracted inventory is also explicitly bound to the
  verified ZIP manifest and exact member set. Repeating verification produces
  identical numerical evidence.
- Confirm that all original return records, old ledgers and source identities are
  preserved. Inventory references to omitted Windows tensors are not described as
  rehashing those absent bytes on Mac. The new final model bytes are present and
  verified; safe CPU loading checks their dtype, shapes, names and finite values
  without a forward pass or scoring.
- Reconstruct all replay loss differences independently. Preserve the strict `>`
  comparison to the unchanged `1e-5` bound. The focused tests cover the exact boundary,
  failing differences, correct repeated-work accounting and malformed counter,
  schedule, missing-row and nonfinite records. No threshold is inferred from results.
- The worker's completion receipt could be mistaken for overall acceptance. Read
  the corrective failure and terminal journal together: worker complete, corrective
  validation failed, no corrective completion receipt, no paired/final scoring.
  The final model's storage validity and monitoring loss cannot override that gate.
- Distinguish failed recovery fidelity from unmeasured learned quality. No fixed
  ARC/BPB/instruction/generation score is fabricated. The original missing final
  state stays missing; recovered bytes are not relabeled as original evidence.
- Treat the second interruption allowance as conditional, not an available retry
  slot after numerical-equivalence failure. No further execution or scoring is
  prepared under the current contract. The prospective next-decision proposal is
  non-executable and requires explicit owner authorization. It targets the observed
  full-scale replay horizon rather than relying only on tiny controls that passed.
- Keep causal uncertainty: matching identities and counts do not establish why the
  replay drifted; the old PowerShell involvement is still unproven. No claim that
  the failure is harmless follows from its small absolute magnitude.
- Review public status/card/changelog wording so closure means a faithfully recorded
  interrupted/failed experiment, not successful Phase 17 or Roadmap 2.0 completion.
  Phase 18 remains blocked, including after any approved publication of this result.

Formatting findings were fixed before final checks. Actual validation and
preservation results are in `closure-validation.json`. Training/evaluation runtime,
frozen source/configuration/gates, prior evidence, tags/releases and dependency
versions were not changed. No CUDA replay, CI inspection, physical Linux validation,
new paid resource, GitHub write or Phase 18 work occurred during closure.
