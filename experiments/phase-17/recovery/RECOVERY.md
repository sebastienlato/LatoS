# Reviewed Windows recovery — original Phase 17 attempt only

The owner requested Mac review and a bounded recovery handoff. The first Windows
return was independently verified. This is the resulting administrative correction,
not a fresh training attempt or a change to the frozen experiment. Keep the original
training checkout at `0fa94ad580110fd2dc7aaa2aa3560950abe73a02` unchanged, even though
Mac has a newer local review commit. Do not pull, install a new LatoS wheel or copy
these files into `source/src/latos`. There are no GitHub writes or Phase 18 work.

## Preserve and verify

Use the original extracted transfer root containing `handoff.json`, `preflight.json`,
`source/`, `inputs/`, `evaluation/` and `validation/`, and its original checkout
`.venv/Scripts/python.exe`. The attempt stays at
`source/outputs/phase17-cuda-attempt1`. Preserve both `ledger.json` and `ledger.json.tmp`:
the former is stale, and the latter records the supervisor failure. Do not change
either status or invoke the original `--resume-external-interruption` flag.

Verify this recovery ZIP's SHA-256 against the Mac message, then extract it into a
new `reviewed-recovery-tools` directory under the original transfer root. The bundle
contains only new administrative tools, their nine regression tests, this guide,
`recovery-plan.json` and `bundle.json` with their hashes. It contains no replacement
model, data, tokenizer, optimizer, original package source or environment. Its
manifest is also verified automatically before execution. Preserve original raw
artifacts and the first return ZIP/receipt outside Git; do not discard any tensor.

Confirm no old supervisor/worker remains. The controller also requires exclusive
original active/worker locks. Do not monitor by reading an actively replaced worker
status file. The new administrative journal is safe to inspect as complete JSONL
lines; after termination read the terminal receipt. A partial last line means
incomplete evidence, never completion.

## Administrative checks

From the original transfer root, Windows Work operates:

```powershell
source/.venv/Scripts/python.exe reviewed-recovery-tools/run_checks.py
```

Retain stdout/stderr in a new log outside `source` and `validation` (for example in
`reviewed-recovery-tools`). All nine tests must pass with no skip. The Windows-only
test holds a read handle on a disposable ledger, demonstrates that replacement is
blocked, and verifies separate journal writes. This reproduces a possible mechanism,
not proof that the PowerShell monitor caused the actual incident. The generated
`windows-checks.json` binds the exact scripts to passing results. Mac ran eight
checks; that Windows API case was skipped, not simulated. The original 411/15
checkout/wheel results and actual CUDA controls remain pinned; no new training
package or dependency is introduced by this controller.

## One corrective replay

```powershell
source/.venv/Scripts/python.exe reviewed-recovery-tools/recover.py --transfer . --stage recover
```

The first invocation creates `phase17-reviewed-recovery` beside `source` and refuses
any second corrective launch. It rehashes original source, inputs, preflight, logs,
ledgers, cached data and all checkpoint tensors before spawning the original
`latos.training.base2` worker for segment 1. Only this reviewed launcher may dispatch
that internal worker. It holds the original supervisor lock and keeps progress in
a new append-only journal; it does not change original trainer or supervisor code.

The original worker must strictly load step 7,000, including model, optimizer,
permutation/RNG, counters and exact runtime/source identity. It replays exactly
updates 7,001–7,485 and then performs the planned final full-development monitoring,
save/readback and completion receipt. Original artifacts are rehashed afterward.
New outputs appear only in segment 1, the final checkpoint and new completion files.

This consumes the first same-attempt resumption, not a new attempt. Lost/replayed
work is 485 updates / 2,426,653 targets. The final logical pass remains 37,811,418
targets; physical executions including replay total 40,238,071. The launcher checks
that distinction and the existing same-backend loss tolerance. It does not claim
bitwise identity with lost final weights that were never saved.

The original active time is charged at 1,562 seconds. At most 5,638 active seconds
remain for recovery, including its integrity and final checks. The cumulative
training cap stays 7,200 seconds, evaluation 3,600, total 10,800. Host/GPU/storage
bounds remain 8 GiB / 85% reserved / 20 GiB. Any failure, identity mismatch, replay
mismatch or exhausted bound stops; retain all results and return to Mac review.
Do not edit receipts, replay again, allocate the second resumption automatically,
change settings or attempt a fresh run. No file-sharing failure retry is hidden.

## Development evaluation only after recovery passes

Require BOTH a complete `recover-result.json` and its matching terminal journal
record, with no recovery failure file. Then:

```powershell
source/.venv/Scripts/python.exe reviewed-recovery-tools/recover.py --transfer . --stage development
```

This runs the original fixed historical-base/candidate evaluator and comparison,
with the same CUDA backend/runtime/batching and original Phase 14 gates. It has
one launch and the unchanged 3,600-second reserve, including verification and
resource checks. Record learned negatives faithfully. Do not run final acceptance
from this handoff, even on passing development; return the evidence for the normal
Mac selection/contamination review. Do not use the stale original ledger to launch
another stock supervisor. A later final comparison, if allowed, must debit this
receipt's evaluation time from the same cumulative reserve.

## Return all evidence and stop

```powershell
source/.venv/Scripts/python.exe reviewed-recovery-tools/package_return.py --transfer . --output phase17-recovery-return.zip
```

This includes original failure evidence, new administrative journals/receipts,
segment 1, full development outputs if reached and final learned model bytes if
produced. Cache/optimizer/intermediate tensors remain on Windows with hashes;
the return is not their complete backup. Keep those originals. Supply the new ZIP
and generated receipt, and summarize actual recovery/test/evaluation outcomes,
active time, repeated versus logical exposure and failures. No source edits, tensor
substitutions, thresholds changes, paid compute, publication or Phase 18.
