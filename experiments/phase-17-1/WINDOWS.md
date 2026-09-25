# Phase 17.1 Windows execution checkpoint

The owner explicitly authorized this fresh attempt and its implementation/preflight
after reviewing the frozen proposal. `authorization.json` binds that authorization
to the unchanged `plan.json` and `PROPOSAL.md`; their historical pending-approval
wording remains a preserved snapshot. No Phase 18 or publication is authorized.

## Placement and verification

Open the preserved original transfer root in a **fresh Windows Work session**:
`C:\LatoS-Validation\phase17\execution-20260924-verified`.
It must contain the original `source`, `inputs`, `evaluation` and `handoff.json`,
plus the original checkout `.venv` and installed `.wheel-env` inside `source`.
Do not pull Git, replace historical source, upgrade packages/driver, change policy,
install a new environment, move/delete old tensors or rerun old suites.

Verify the new ZIP against the exact SHA-256 in the Mac kickoff and receipt before
extracting. Extract its flat contents into a **new** `phase17-1-tools` child folder.
Never extract into `source` or an existing tools directory. The bundle contains
only reviewed controller code, zero-update tests and metadata; it carries no corpus,
tokenizer, weights, reserved text or private context. It deliberately reuses the
preserved original transfer and both existing pinned runtimes.

Read `plan.json`, `PROPOSAL.md`, `authorization.json` and this guide. Verify the
manifest and its payloads using the original checkout Python, from the transfer root:

```powershell
source/.venv/Scripts/python.exe -c "import sys; from pathlib import Path; sys.path.insert(0, 'phase17-1-tools'); from common import verify_bundle; verify_bundle(Path('phase17-1-tools')); print('Bundle verified; no optimization')"
```

Work performs these operations and preserves console output. At any verification
failure, stop and report; do not repair source locally, substitute input files or
silently repackage. The reviewed source commit is in `bundle.json`; it is local to
Mac and intentionally absent from remote main until publication is separately approved.

## Exactly one initial supervisor invocation

Ensure no concurrent CUDA workload and at least 20 GiB free on the destination.
Keep the original runtime (Python 3.14.7, Torch 2.14.0+cu130, driver 616.92, RTX 4070
SUPER) and all controls specified in the approved plan. From the original transfer root:

```powershell
source/.venv/Scripts/python.exe phase17-1-tools/run.py --transfer . --stage start > phase17-1-execution.log 2>&1
```

Use only this supervisor. Do not invoke `worker.py` directly or any old Phase 17,
recovery or diagnostic training command. Do not run the full training on Mac/MPS.
The supervisor exclusively claims `phase17-1-launch.json` and creates the fixed
new `phase17-1-results` root. Never delete either or choose another name to obtain
a second launch. Reinvocation after failure is not authorized.

The bounded sequence is:

1. Verify every original transfer record and the installed-wheel source using
   hashes before importing the training package. Reserved LM test handling here
   is opaque hashing only, never parsing/scoring. The accepted source/data remain
   unchanged; all generated files go to the new attempt roots.
2. Run only the new CPU checks in checkout and installed wheel. They forbid
   optimizer updates and all must pass without skips. Their controller sequencing
   tests use explicitly scripted child records, not real training. Retain both
   JUnit files and logs; do not run unrelated historical tests/evaluation suites.
3. Execute the four predeclared tiny adapter pairs, each 8-update reference plus
   a separate-process 5-update replay from its step-3 checkpoint: ordinary and
   partial-tail fixtures in both runtimes, at most **52 CUDA fixture updates**.
   Fixture identities, model dimensions, seeds and token formula are frozen in
   the plan/identity file. Require exact states/inputs/rates and unchanged `1e-5`
   loss equivalence. Any failure stops before serious training; no retest/tuning.
4. Start **one fresh seed-160 model**, never load a previous learned/probe/diagnostic
   model. Run the unchanged 34,087,424-parameter Phase 16 model, accepted Phase 15
   data/tokenizer and one-pass recipe: **7,485 logical updates / 37,811,418 targets**,
   BF16, batch 2 × accumulation 8, final accumulation 2. Record all full-state/input
   fingerprints and the fixed Windows learning-rate table. Complete all immutable
   checkpoints and policy bindings at 0, 1,000…7,000 and 7,485.
5. Only after valid complete training, run the unchanged paired fixed **development**
   evaluation against the preserved Phase 5 reference and the practical inference
   measurements. Retain positive or negative results. The process completing is
   not a learned-quality pass: inspect the comparison and inference gate results.

The tested prospective deterministic settings are established before CUDA/Torch
work and checked repeatedly. Mandatory checkpoint policy records bind the source,
attempt, runtime/controls, data, schedule and exact full-state fingerprints. Missing
or mismatching records fail closed; old checkpoints cannot become new ones.

## Limits, monitoring and failure preservation

The two-hour cumulative training budget includes integrity checks, CPU/CUDA preflight,
data/cache preparation, imports, state hashing, training, monitoring, checkpoints
and any reviewed recovery. Paired development/final evaluation share one active
hour; total active ceiling three hours. The prior diagnostic budget is closed and
supplies no extra time/updates. Host cap 8 GiB, reserved GPU cap 85%, new artifact
cap 20 GiB with a 1 GiB administrative reserve. Reused original payloads/environments
are separate; all new copies count. No paid resources or automatic extensions.

Monitor complete lines of `phase17-1-results/sessions/00/events.jsonl` and finished
immutable receipts. Each later authorized session gets a distinct journal. No
active ledger is replaced. Do not treat a partial/torn file, worker completion alone
or missing receipt as success. Preserve all exceptions, logs, staging directories,
checkpoints, model/optimizer tensors and the original failed Phase 17 evidence.

On interruption or failure, stop and return evidence. At most two genuine external
same-attempt resumptions may later be permitted after **Mac review**, within the
existing cumulative bounds and exact replay gates. Windows must not author its own
review, restart at a different checkpoint, rerun a failed preflight or reset time.
The `resume` CLI requires a separately prepared Mac record binding the complete
prior file inventory, checkpoint-policy hash, conservative prior time/work charges
and verified external-interruption classification. Numerical/resource/identity/
quality/unsupported-operation failures are terminal. The physical serious-work
ceiling including possible replay is 9,485, not permission for extra epochs.

Final acceptance is **not part of this initial invocation**. Even passing development
requires independent Mac review, a candidate-bound contamination/selection declaration
and a separate `phase17.1-final-Mac-review` record. The final CLI rejects absent or
mismatched records and a prior final attempt; it never generates its own permission.
The old instruction reserved test stays reserved. No final-feedback tuning.

## Return to this authoritative Mac task

After the supervised run finishes or fails and no worker remains, package the evidence:

```powershell
source/.venv/Scripts/python.exe phase17-1-tools/package_return.py --transfer . --output phase17-1-windows-return.zip
```

Return `phase17-1-windows-return.zip`, its generated
`phase17-1-windows-return.receipt.json`, and a concise summary of the actual CPU/CUDA
preflight, training/replay/exposure, time/resources, development/inference gates and
any failures. Verify the archive readback/receipt digest. If packaging itself fails
(for example the artifact ceiling has already been violated), preserve everything
and report that failure and paths; do not remove evidence or spend extra budget.

The compact return includes final new candidate model bytes if present, all scalar
and state-digest evidence, policy records, source snapshots, requests, journals,
inventories and failures. Cache/optimizer/intermediate tensor originals remain on
Windows with their hashes. This subset is not their complete backup; do not delete
them without a separately verified backup. No archive/tensor is uploaded to GitHub.

Stop after returning evidence. Published Phase 17 stays failed. No Phase 18, push,
tag, release, model promotion or other publication is authorized.
