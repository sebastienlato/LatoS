# Small-subset packaging-only corrective checkpoint

Status: Mac implementation/review complete; **stop before Windows execution**.
The owner reports a packaging exception after 21m13s, without a verified return ZIP,
receipt or completed record. The actual Windows artifacts have not been supplied
or independently reviewed on Mac. No A/B/C quality result is inferred here.

## Cause established in prepared source

The frozen native `compare(reference, candidate, "base")` correctly requires the
reference weights to be the historical Phase 5 base. The original small-subset
packager first makes four legitimate Phase 5-to-endpoint comparisons. It then
mistakenly calls the same base-gate function for **A1→B1** (and later other cross-arm
contrasts). A1 is not Phase 5, so this call raises the reported exception even when
the real historical reference is present and valid. The `diagnostic` variable name
and output label do not change the comparator's acceptance contract.

A regression fixture using previously saved Phase 17.1 outputs reproduces the exact
exception with a valid Phase 5 reference. That fixture is not this Windows study's
A/B/C evidence. The original 23-check suite did not exercise this native cross-arm
comparison end to end; the corrective tests do. The implementation error is in our
packager, not a reason to weaken the acceptance guard or replace the historical base.

The correction keeps all four original base-gate comparisons unchanged. For the
four already-prespecified descriptive contrasts, it uses the unchanged paired-row
bootstrap helper directly, after all saved outputs pass primary identity/runtime/
row validation. It never applies a base gate to A1/B1/B2/C2 as a reference, substitutes
an assistant gate, edits scores, selects another checkpoint or computes new logits.

## Conditional scientific/procedural validity

One packaging-only correction is valid **if the preserved records confirm this
specific failure, complete original jobs/development outputs and sufficient budget**.
The corrective launcher checks those prerequisites on Windows. It stops if the
traceback instead indicates a missing/invalid Phase 5 reference, incomplete original
execution, a different failure, changed source/cache, or insufficient time. It cannot
train, recover a model, acquire missing inputs, rerun evaluation or repair evidence.

It verifies the original tool bundle/source hashes, recomputes saved execution/input
accounting, and requires agreement with the original `run/verification.json`.
This existing file must not be overwritten: merely retrying the old packaging
function would encounter its exclusive-write guard before reaching the comparison fix.
All new reports, archive and receipts instead go to a new sibling directory,
`small-subset-packaging-fix1`. The complete original study directory is inventoried
before and after processing and must stay byte-for-byte unchanged. Its failure logs
and failed guardian/outcome remain in the corrective return; no original `completed.json`
is invented. Omitted tensor/cache bytes remain inventory-bound on Windows.

## Budget: carry the old debit, never a new 45 minutes

The current explicit owner authorization permits this one stopped-study packaging
continuation only. Charge every second of its new invocation, including Python
startup, validation, aggregation, hashing, packaging, readback and closure, on top
of the original execution. The original total does not reset.

- Original debit `D = max(1273, ceil(original guardian.total_wall_seconds))`.
- Remaining total is `2700 − D`: **at most 23m47s**, not independently verified yet.
- Also retain the original five-minute packaging stage: debit
  `P = ceil(D − original outcome.elapsed_before_package)` for the failed packaging.
- Corrective cap `C = min(2700 − D, 300 − P)`. Refuse if `C <= 10` seconds.
  Thus even the old packaging-stage allowance is not reset.
- The worker stops at new origin + `C−10`; the pinned existing Windows Job Object
  guardian kills its entire process tree at origin + `C−5`. The new guardian receipt
  must show `D + correction_elapsed <=2700` and `P + correction_elapsed <=300`.

All time checks use the kickoff's Windows QPC timestamp captured before Python
starts. A fresh timestamp measures only the additional interval; it never replaces
D. No budget credit is given because original packaging failed. No GPU preflight,
model construction/loading, forward pass, optimizer update, generation, scoring,
paid resource, retry or resumption is permitted. Runtime tripwires prohibit those
paths and CUDA is hidden. This is saved-record arithmetic and file handling only.
If it cannot finish within C, preserve the failed correction and stop.

## Minimal Windows handoff

Only one new executable file is transferred: `small-subset-packaging-fix.py`, plus
its receipt and prepared kickoff. It reuses the hash-pinned original guardian and
verification helpers already in `small-subset-v1/run/tools`. No dataset, checkpoint,
environment, worker or training configuration is shipped or modified.

Use the original root `C:\LatoS-Validation\phase17\execution-20260924-verified`.
Do not manually extract evidence, patch old tools, delete the original verification,
rerun the original launcher, pre-hash large artifacts, install dependencies or launch
A/B/C. Run only the exact block in the Mac-generated corrective kickoff once.
The fixed new correction directory is a permanent attempt claim; never remove it
to retry. No action occurs on Windows during this Mac preparation.

On success return these existing files from `small-subset-packaging-fix1`:
`small-subset-corrected-return.zip`, `return.receipt.json`, `guardian.json`.
If no verified return is produced, stop and report the existing correction error/
log files; do not start a second correction or create a manual archive. Keep all
original and omitted artifacts. The Mac corrective reader independently checks
archive/source bindings, preservation, cumulative budgets, saved execution and
saved comparisons without running returned code or any model.

## Separate review and validation

The separate review checked reference identity versus descriptive pairing, the
secondary exclusive-write failure, original-file preservation, conditional admission,
conservative rounding and both cumulative limits, immutable attempt naming, readback,
zero-model/optimizer tripwires and the absence of any training/evaluation dispatch.
Focused tests reproduce the original defect on real historical saved rows, verify
primary gate results unchanged, reject a genuinely invalid/missing reference, reject
bad time/exit/trace/completion states, detect mutations, and exercise a complete
scripted packaging/readback while preserving the original verification/failure.
Actual counts and source hashes are in [validation.json](validation.json).

No new learned-model result, Windows execution or physical GPU validation is claimed.
The authorization overrides the old prohibition on later packaging only for this
specific bounded correction. It authorizes no study change, new experiment, final
scoring, acceptance, Phase 18 or publication. Phase 17 remains permanently failed/
interrupted, Phase 17.1 development-quality-failed, and no Base Model 2.0 is accepted.
