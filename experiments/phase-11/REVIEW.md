# Separate Phase 11 review

A separate pass after implementation, within this task, reviewed runtime boundaries,
CLI behavior, metric denominators, actual evidence, packaging and preservation. This
was not independent external platform validation or a second model's review.

Findings and fixes:

- A trusted callback could return oversized text which was rejected for execution
  but retained unbounded in the session trace. Cap retained text at 1,024 characters
  and mark truncation. Added a 100,000-character rejection/trace-bound regression.
- The first runner captured runtime/plan/lock but omitted its independent verifier,
  tests and package metadata. Capture these in the reviewed run before evaluation;
  preserve the first run, with both inventories and all responses intact.
- JSON validity must be measured separately from EOS completion. Parse and validate
  the returned text before the completion gate, while refusing execution/finalization
  for every non-EOS response. Reject exponent overflow as well as literal NaN/Infinity.
- Keep absent-call argument/execution denominators null, required-call omissions as
  failures, and direct lucky final answers out of tool-task success. Tests include
  same-answer/wrong-argument calls and a missing-key final without a lookup.
- Report context exhaustion as an explicit limitation: the unchanged learned models
  produced no valid first response and no second learned turn. Scripted chains do
  not establish learned multi-turn success. Do not silently increase context or
  improve prompts after viewing the fixed result.
- Synchronize every fresh-wheel dependency from the lock: the initial offline install
  selected a newer cached build-helper dependency. The equality check caught it
  before tests; the corrected isolated environment matches the locked installation.
- Retain the initial test harness PATH failure and original run. Use the locked
  environment for the corrected suite. No tests or negative results were removed.

Checked exact source capture, all saved response metrics via an independent recount,
corruption rejection, no changed model tensors/gradients, unchanged shared chat code,
finite bounded tools, no external tool side effects, all old artifact/test hashes,
version-only lock delta, all prior capability/platform distinctions, source/wheel
contents, isolated installed-wheel tests, public links and private-content exclusion.
Final validation results are in [validation.json](validation.json). No actionable
finding remains within this bounded scope. Full learned tool quality, external
platform validation and broader tool coverage are not claimed.

Publication remains main only, no tag/release/assets, pending explicit approval for
the reviewed local commit. Phase 12 has not begun.
