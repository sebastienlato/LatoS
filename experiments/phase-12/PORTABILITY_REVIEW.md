# Separate review — Phase 12 portability correction

Performed after implementation within the authoritative Mac task; no delegated
review or native Windows execution is implied.

- Confirmed the reported mixed key format against the published runner/verifier.
  The initial emulation first exposed backslash resolution on the Mac; moving the
  final inventory check before condition-file resolution reproduced the precise
  Windows membership error. Both diagnostic logs are retained, not model attempts.
- Reviewed all newly written evidence path maps, not just the two literal entries:
  inventory, captured source and input keys now use forward slashes. Actual I/O,
  input identity checks and current-source validation are unchanged.
- Consolidated final assembly through the tested inventory function. Only its own
  root inventory is excluded; both nested condition inventories remain covered.
  The regression invokes the actual final writer without model evaluation, checks
  exact membership and hashes, repeat writes and rejection of changed file contents.
- Confirmed the Windows-path simulation fails with published logic and passes with
  corrected logic. Native Mac success or simulated Windows strings cannot establish
  Windows/CUDA PASS; the owner must perform the fresh external retest after publication.
- Checked both original runs read-only with the corrected inventory checker and
  original captured-source verifier. Preserve their source fingerprints, plans,
  hypotheses, results, all raw outputs, model identities, context limits and tolerances.
  No research rerun is warranted for a string-representation correction.
- Keep the source-identity guard strict. Old-run full verification from the changed
  checkout must still reject changed source; document use of exact original source
  for historical replay, rather than updating or bypassing historical fingerprints.
- Keep owner-reported Windows 299 passes/13 skips, 229 unchanged tracked files and
  3,045 unchanged evidence files distinct from local checks. No raw handoff files,
  Phase 12 CUDA execution, fresh Windows wheel result or new Linux result is invented.

Complete Mac regression, fresh locked non-editable wheel validation, package/privacy,
prior-artifact hashes and the full staged diff are recorded in portability-correction.json
and the local publication record before commit. No unresolved finding in this scope.
Stop for explicit main-only correction publication approval; all eleven tags remain
unchanged. After approved publication wait for the owner's fresh Windows/CUDA result.
No Phase 12 closure or new research extension.
