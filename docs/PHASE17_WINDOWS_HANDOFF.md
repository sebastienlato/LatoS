# Phase 17 Windows/CUDA execution handoff

The owner has authorized Phase 17 serious pretraining and fixed evaluation. This
handoff is an execution checkpoint, not completed training or phase acceptance.
Use only the actual RTX 4070 SUPER and the frozen contract in `docs/TRAINING_2.md`,
`configs/training2/phase17-plan.json` and `experiments/phase-17/PLAN.md`. The plan's
historical start flag is superseded by the recorded owner authorization; its
configuration and budgets remain unchanged. No Phase 18 or GitHub writes.

## Transfer and isolation

Use the Mac-provided `phase17-windows-handoff.zip` and its SHA-256. This is an explicit
reviewed source/input allowlist, not a workspace archive and not a public release.
Extract into a new directory, preserving previous source, probe checkpoints and
all failed attempts. Remote main does not yet contain this local implementation.
`handoff.json` binds the exact reviewed commit and every transferred file.
Private planning, credentials and environments are excluded. Dataset terms and
notices travel with the package; the transfer is for the owner's existing machines.

Before installing, use an existing standard Python to run:

```powershell
python source/experiments/phase-17/verify_handoff.py .
```

Verify the outer ZIP digest first. Read `AGENTS.md`, `PROJECT_STATE.md`, `ROADMAP.md`,
`experiments/phase-17/PLAN.md`, this guide and the frozen training/evaluation docs.
Use the exact extracted source; do not pull or substitute another commit.
The package includes pinned Phase 15 inputs, the actual historical base/tokenizer,
ARC validation inputs, and original LM validation/report/test. The test is opaque
until the final-access gate permits scoring. Never inspect reserved examples to
plan training or tune output. The old instruction test is not transferred.

## Preflight — Windows Work performs the commands

Use existing Windows x86-64 Python 3.14.7 and uv 0.12.15, with the unchanged lock.
`uv sync --locked --all-extras` routes Torch to 2.14.0+cu130. Capture installation,
`uv pip check`, doctor CUDA JSON, driver 616.92, hardware/disk reports and all exact
commands. If the required machine/driver/runtime is unavailable, record the blocker
and stop; no paid compute, new driver, dependency upgrade or substitute backend.

Keep logs/build outputs in a new `validation/` directory at the transfer root,
outside source. Run lint and formatting, the entire checkout test suite with
`--junitxml ../validation/checkout.xml`, and a fresh non-editable wheel suite with
`--junitxml ../validation/wheel.xml`. Build the wheel/sdist into `validation/dist`.
Use the same locked dependency versions in the isolated wheel environment; verify
imports resolve to site-packages, with no editable project on the test path. Run
pytest from the source directory with its environment interpreter and put that
environment's executable directory first on PATH for CLI subprocess tests. Do not
use an import mode that hides the test modules required by existing tests.
Keep runtime environments outside `validation/` (for example `source/.venv` and
`source/.wheel-env`); that directory is only retained evidence/build outputs.
Keep all failed commands and logs; do not rewrite test outcomes or force skip counts.
Both `test_cuda_bf16_numerical_control` and `test_full_single_pass_cuda` must pass,
in checkout and wheel, along with cache/gradient/recovery tests. Stop on failure
before serious optimization. Return an actual source defect to Mac with evidence;
do not silently change measured source or scientific settings on Windows.

Then, from `source`, using the tested checkout interpreter:

```powershell
python experiments/phase-17/preflight.py --transfer .. --checkout-junit ../validation/checkout.xml --wheel-junit ../validation/wheel.xml --wheel ../validation/dist/latos-1.3.0-py3-none-any.whl
```

This produces a source-bound `preflight.json`. Work also verifies the wheel import
and complete logs; the receipt is audit evidence, not an authentication mechanism.
Ensure at least 20 GiB free. Account for transfer, validation and new run artifacts
against the 20 GiB hard phase cap; report reusable environment sizes separately.

## One serious attempt

Use one output directory, never a second fresh root to evade the attempt count:

```powershell
python -m latos.training.base2 --transfer .. --output outputs/phase17-cuda-attempt1 --stage train
```

Use the supervisor command, never internal `--worker`. It enforces the two-hour
active training allowance and records an incremental ledger; the worker verifies
source/input bytes, runtime, full-data cache hashes and counts before initialization.
No automatic retries, epochs, checkpoint selection or settings changes. It records
all update exposure, peak memory, boundaries, source snapshots, immutable checkpoints
and failures. The initial model is freshly initialized at seed 160; no probe reuse.
Full new-development monitoring is separate from fixed acceptance evaluation.

A numerical/resource/corruption failure stops the phase attempt. An external
interruption may resume the same output root at most twice, after verifying the
previous supervisor AND worker processes have exited and retaining their logs.
Use `--resume-external-interruption "specific observed cause"` with `--stage train`.
Never use this flag for a numerical/resource failure. The last complete checkpoint
is loaded with strict source/runtime/configuration/permutation identities. Previous
segment records remain; overlapping updates are replayed work, not extra unique
data. If no valid checkpoint or intact cache exists, stop and return the blocker.
A running ledger left by an OS termination requires a documented process-exit check
before resumption. Time already recorded is retained, not reset.

## Paired development evaluation

Only after complete training and no resource failure:

```powershell
python -m latos.training.base2 --transfer .. --output outputs/phase17-cuda-attempt1 --stage development
```

This evaluates the preserved historical base and the final candidate using the
unchanged Phase 14 evaluator on identical CUDA runtime/backend/batch settings,
then reconstructs the gate comparison and measures the frozen inference resource
protocol. Its one-hour active evaluation budget is shared with any later final
comparison. Keep every sample and failure. A missing baseline/input is a blocker,
not permission to substitute random weights. No changing protocols or thresholds.

If development or resource gates fail, retain the negative result and return it;
do not run final acceptance, train again or begin Phase 18. If they pass, return
the evidence for the separate authoritative Mac selection/contamination review.
Do not create a positive contamination declaration merely because training ran.
Mac will supply the reviewed declaration and any bounded continuation needed for
one final comparison; this review is part of the already authorized Phase 17,
not another routine phase-start approval. No reserved feedback may drive training.

## Return and preservation

Keep all original outputs and every tensor checkpoint on Windows. From `source`:

```powershell
python experiments/phase-17/package_return.py --transfer .. --attempt outputs/phase17-cuda-attempt1 --validation ../validation --output ../phase17-windows-return.zip
```

The return contains all attempt/validation evidence plus the final learned model
when produced. It excludes cache binaries and initial/intermediate/optimizer tensors
but records every omitted file's size and SHA-256. It is an evidence subset, not a
complete checkpoint backup. Keep original checkpoint directories until a verified
backup/transfer destination exists; never discard the only copy. Do not upload this
archive to GitHub. Include the archive digest, exact source commit, actual training
status/exposure, gate scores, resources and failures in the return summary.

Stop after the return. Mac verifies file/source bindings, reconstructs exposure and
scores, reviews model outcomes and integrates evidence before any phase-completion
or publication proposal. Positive and negative learned-quality outcomes are both
valid reports; passing engineering controls is not passing the learned-quality gate.
