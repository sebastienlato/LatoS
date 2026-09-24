# Project state

Updated: 2026-09-24.

## Active checkpoint

**Phase 17 is incomplete after a Windows supervisor I/O failure.** The first
bounded attempt ran all planned optimizer updates, but no final checkpoint or
training completion receipt was saved. No fixed development/final scoring ran.
Mac has independently reviewed the return and prepared a bounded administrative
recovery handoff for the existing Windows artifacts. **No Phase 18 or push.**

Original training source remains exactly
`0fa94ad580110fd2dc7aaa2aa3560950abe73a02`; its imported implementation hash is
`76cf4a198ee6acde566fdcc70c9420c0cae8ab6cc0d6b4d7e4c9c0cba5c8f09c`.
All `src/latos`, frozen training/model/evaluation configuration, acceptance gates
and lockfile bytes remain unchanged. New recovery code lives outside that package.
The current Mac review commit must **not** replace the Windows training checkout.

Phase 16 remains published at `ce9de1b734a8e23fd54e9d137cf269f484631765`.
Phase 17 work remains local and unpublished. Package is 1.3.0; existing tags/release
are not changed. Historical pending-publication snapshots retain their original scope.

## Verified first-attempt evidence

[Independent review](experiments/phase-17/INTERRUPTION_REVIEW.md) and
[numerical record](experiments/phase-17/interruption-review.json):

- Return ZIP matches the owner's `b4d95919…b5338e` digest; 3,761 included files verify.
  Returned source and tested wheel bind to the exact local training commit.
- Windows checkout/wheel each: **411 passed / 15 skipped / zero failures**. Both
  required CUDA controls executed. Mac did not rerun CUDA or execute returned scripts.
- **7,485 updates / 37,811,418 targets / 119,748 windows**, epoch zero; final update
  two microbatches / 1,520 targets. Schedule and exposure independently reconstruct.
- Last complete checkpoint: **step 7,000 / 35,384,765 targets / 112,000 windows**.
  Eight checkpoints through 7,000 have consistent metadata/inventory identities.
  Their tensor bytes remain on Windows; strict rehash/load there is still required.
- **485 updates / 2,426,653 targets** were executed after that checkpoint and lost
  when the supervisor terminated the worker. Step 7,000 is not a final candidate.
- The preserved temporary ledger records **1,561.6473704 seconds**, `supervisor-failure`,
  Windows access denied replacing the main ledger. The main ledger is stale at
  1,560.9963497 seconds, `running`. Both remain immutable evidence.
- Latest logged peaks: **1.10 GiB reserved VRAM / 2.37 GiB host working set**.
  Abrupt termination left no final worker resource receipt. No numerical/resource
  failure was recorded. Concurrent PowerShell reading is a plausible contributor,
  not an established cause; the actual lock owner was not instrumented.

## Bounded recovery decision

The complete checkpoint and existing exact-source recovery mechanism support a
conditional same-attempt replay after the worker's administrative interruption.
A stock resume based on the stale ledger is prohibited. The reviewed launcher uses
append-only status, preserves both failed ledgers, excludes concurrent workers and
calls the exact original worker in its original environment. No source/metadata
migration, monkeypatch, new model, epoch, seed, schedule or threshold change.

[Recovery guide](experiments/phase-17/recovery/RECOVERY.md): first allowed resumption,
**one corrective launch**, restoring step 7,000 and replaying only updates 7,001–7,485.
Logical pass stays 37,811,418 targets; successful total physical execution would be
40,238,071 targets, explicitly including 2,426,653 replayed targets. No claim of
bitwise comparison to absent lost final weights. Original strict loader and existing
same-backend recovery tolerance must pass before evaluation.

Charge prior training time upward to **1,562 seconds**. Recovery verification/replay/
final save/readback has at most **5,638 additional active seconds**. Evaluation still
has its cumulative **3,600-second** reserve. Original 7,200/10,800-second training/
total limits and 8 GiB host / 85% GPU / 20 GiB artifact caps remain. Any new failure
returns to Mac review; no automatic second resumption or fresh attempt.

## Validation and next action

The [recovery validation](experiments/phase-17/recovery-validation.json) records
actual local checks and the separate review. The Windows-only file-sharing fixture
must execute there before replay; Mac results do not substitute for it. The
original CUDA numerical/recovery controls and source-bound preflight stay preserved.

Transfer the small reviewed recovery bundle to the **existing** Windows transfer
root, outside its source package. Windows Work runs administrative checks, the one
bounded replay, then paired fixed development evaluation only after a valid final
checkpoint and complete administrative receipt. Return full evidence and final
model bytes if produced. Passing development still requires Mac contamination/
selection review before the single final comparison. Negative results stay retained.

Phase 17 has no learned acceptance result yet. Historical negatives and fixed
quality gates remain unchanged. Do not begin Phase 18, publish, create remote
artifacts or discard Windows originals. Exact administrative commit/archive IDs
are in the ignored recovery handoff record; the imported training commit remains
`0fa94ad580110fd2dc7aaa2aa3560950abe73a02` regardless of later Mac review commits.
