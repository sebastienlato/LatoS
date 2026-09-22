# Separate Phase 10 closure review

Documentation/evidence review performed after the closure draft, within this Work
task. No independent Windows execution or physical Linux validation is implied.
Compared the owner's summary with the attributed closure records; inspected GitHub
run/job metadata, logs, exact-commit workflow and preference test/fixture bodies.

Review resolutions:

- Keep Windows **243 passed / 11 skips** and Linux **245 passed / 9 MPS-only skips**
  separately attributed. Windows individual skips/raw logs/hashes were not supplied;
  do not fill those gaps from Linux logs or earlier platform results.
- Distinguish tiny Windows **4 updates / 16 pairs / 169 response targets**, unchanged
  **16/32 → 16/32** ranking, from full Mac **100 updates / 400 pairs / 2,498 targets**,
  worsening **16/32 → 15/32** ranking. Both exact-reply results stay **0/32 → 0/32**.
  Slight English loss recovery versus SFT does not surpass the original base or
  establish preference-learning/instruction-following improvement.
- Limit Linux evidence to 17 CPU preference checks / one MPS preference skip inside
  the 254-case suite, tiny two-update/eight-pair integration runs, tiny oracle/replay,
  existing tests and offline workflows/builds. Its separate dense fixture recovery
  is not DPO or adapter optimizer resume. No fresh installed-wheel suite or full
  learned Mac experiment/oracle appears in the workflow. Physical Linux deferred.
- Keep original Phase 10 report, results, samples, script/inventory files and failures
  intact. Update current navigation rather than rewriting historical measurements.
  Preserve earlier negative SFT/LoRA quality, amended merge 1e-4/original 1e-5 failure,
  separate cache bounds, unsupported adapter/DPO resume and capability limits.
- Separate owner-reported Windows preservation (2,017 evidence/report files and 194
  tracked files) from 54,356 independently rehashed local files. Reserved payloads
  receive only opaque integrity hashing; no parsing or evaluation.
- Make the next gate explicit: prepare the fresh-chat handoff only; no Phase 11
  selection, implementation or training. Approval permits the closure publication,
  verification and stop, not Phase 11 work in this chat.

Validation: structured evidence counts, CI commit/job/step identity and log hashes,
original runtime/evidence identity, all preservation hashes, Markdown links, JSON,
full staged diff/whitespace, private-content and history checks. No new runtime
suite, training, benchmark or build was needed or run for documentation-only closure.
No actionable finding remains within this scope. Publication requires a new explicit
approval for the exact reviewed closure commit; no tag, release or assets proposed.
