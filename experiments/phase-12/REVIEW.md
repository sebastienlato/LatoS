# Separate Phase 12 review

Review performed after the first implementation/evaluation, within this task.
No delegated review or external platform execution is implied.

Findings and fixes:

- **Verifier exceptions could be swallowed.** The runtime conversation handler
  catches ValueError/RuntimeError. Initial synthetic tamper testing rejected the
  evidence but surfaced a generic session mismatch rather than its precise token
  error (one failing test / five passes). A crafted generation-error session could
  mask the underlying validation exception. Use a dedicated EvidenceError outside
  that handler's caught types; add a forged-generation-error regression test.
- **Per-case scores were not independently cross-checked by the older verifier.**
  Add explicit score reconstruction alongside saved-response replay; retain the
  unchanged independent aggregate recount from Phase 11.
- **Evidence checks needed stronger provenance and resource coverage.** Check current
  source against captured hashes, full inventory coverage, matching condition plans
  and top-level summaries, final time/RSS budgets, and retain partial budget-failure
  traces. Preserve input identities and compare actual tensors before/after evaluation.
- **Scope could overstate context extension.** Existing models already support 512
  positions; this changes only the experimental session budget. Do not alter default
  inference limits or claim learned long-context quality. Preserve original prompt,
  fixed cases and both reserved tests. More turns and scripted successes are not
  learned tool use. Windows compact-prompt evidence remains a different experiment.
- **Resource timing is not a benchmark.** Disclose concurrent full-suite checks;
  distinguish RSS high-water from MPS boundary readings. Attempt-boundary limits
  cannot stop an already executing kernel. No paid resource or upgrade is used.

The retained first evaluation and reviewed second evaluation have identical raw
sessions, attempt budgets, scores, metrics and token totals. No protocol/hypothesis
change or result selection occurred. Seven focused tests pass; the reviewed full
suite and fresh non-editable wheel each pass 312 tests. The saved-evidence verifier
passes; rehashed metric corruption is rejected by the independent recount, and
inventories/forged replay regressions pass. Final packaging, prior-artifact hashes,
source/default identity, private-content/history, links and complete staged diff
are checked before commit. Actual results appear in validation.json.

No actionable finding remains within this bounded experiment. This does not close
unperformed Phase 12 external validation or establish a learned capability. Stop at
the concrete main-only publication checkpoint; all eleven tags remain unchanged.
