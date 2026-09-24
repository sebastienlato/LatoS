# Windows recovery-mechanics diagnostic handoff

This is a newly authorized bounded diagnostic supporting a possible Phase 17.1.
Published Phase 17 remains permanently failed. Do not continue its old attempt,
load its learned models, alter its ledgers or reinterpret its tolerance/results.
Do not begin full Phase 17.1, Phase 18 or any remote publication.

## Placement and preflight

Verify the supplied ZIP digest, then extract it to a **new** directory such as
`recovery-mechanics-tools` alongside the preserved original transfer. Read `PLAN.md`
and `plan.json`. `bundle.json` binds the exact new reviewed source. The bundle has
no corpus, tokenizer, historical weights, held-out inputs or final suite payloads.
Do not copy it into the old `src/latos`, modify old source or install a new LatoS wheel.

Use the existing original checkout `.venv/Scripts/python.exe`, whose installed
LatoS source must have hash `76cf4a198ee6acde566fdcc70c9420c0cae8ab6cc0d6b4d7e4c9c0cba5c8f09c`.
Keep the original locked versions and driver. No package upgrades, new environments,
paid resources or alternate machines. Work runs the commands itself, retaining logs.
For example, from the original transfer root (adjust only the new tools directory):

```powershell
source/.venv/Scripts/python.exe recovery-mechanics-tools/run_checks.py
```

This runs only the new tiny CPU engineering tests. They forbid optimizer updates;
no CUDA comparison or held-out reader is invoked. All must pass with zero skips.
Do not run the old full test/evaluation suites for this handoff. Preserve any error
and return for Mac review instead of editing measured source or rerunning a study.

## One study invocation

```powershell
source/.venv/Scripts/python.exe recovery-mechanics-tools/run.py --output recovery-mechanics-results
```

Use only the supervisor, not internal worker options. Choose this one new output
root; do not use a second name to bypass its launch receipt. Do not run concurrent
CUDA work during the comparison. The supervisor creates synthetic IDs from the
supplied training-length metadata, then runs the legacy reference/replay pair and
the prospectively deterministic reference/replay pair in separate processes.

Each reference starts randomly and runs 997 updates, saving a complete checkpoint
at 512 before continuing. Its matching new-process replay restores that checkpoint
and executes exactly 485 updates. Both use the selected 34.09M model, BF16, batch two
and accumulation eight, ending with accumulation two. Total ceiling: **2,964 physical
updates**, no retries or extra profiles. This is synthetic mechanics, not serious
English pretraining, historical recovery or a model to promote.

The original `latos` package is imported unchanged. Process controls differ only
as declared: deterministic algorithms/error mode, cuDNN determinism and cuBLAS
workspace configured before process startup. Relevant additional controls/runtime
are recorded. Any unsupported operation fails; do not silently choose another
profile, switch precision or enable warn-only behavior.

The parent enforces 60 active minutes including preflight/preparation/hashing and
all workers, 6 GiB new artifacts (including its reserve), and the per-job update
limits. The worker enforces 8 GiB host / 85% reserved GPU bounds. Existing Phase 17
artifacts and reusable environments are preserved outside this new artifact scope.
Monitor only complete lines of the supervisor's append-only `.events.jsonl` or
finished immutable receipts; do not recreate the old ledger-replacement problem.
All failures and partial results remain evidence. A hard termination/incomplete
journal is not a passing result and does not authorize a repeat.

## Outcome and return

A positive decision requires a complete legacy loss-equivalence failure and a
complete corrected continuation with exact per-update tensor/optimizer/sampler/RNG
hashes and no loss excess above `1e-5`, with exact checkpoint roundtrips. If control
drift is not reproduced or corrected state differs, the frozen decision is
inconclusive/failed. Do not reinterpret those conditions or tune from the result.
Mac will independently review the source binding, state/exposure records and scope.

Regardless of positive, negative or incomplete outcome:

```powershell
source/.venv/Scripts/python.exe recovery-mechanics-tools/package_return.py --run recovery-mechanics-results --output recovery-mechanics-return.zip
```

Return the ZIP, generated receipt and concise actual test/job/update/time/resource/
comparison results. The archive includes all scalar/state-digest traces, environment,
source snapshots, inventories, supervisor journals, bundle and failures. Checkpoint
and synthetic-cache tensors remain on Windows with hashes; keep originals, since
the compact return is not their full backup. Do not upload either archive or tensors
to GitHub. Stop after returning evidence. No full Phase 17.1 run or Phase 18; even a
positive diagnostic only enables a Mac-reviewed proposal for a separately authorized
fresh Phase 17.1 Mac Work chat.
