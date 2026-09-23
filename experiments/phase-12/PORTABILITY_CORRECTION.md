# Phase 12 Windows evidence-path correction

Base: published `5b72872a2032f8999aae9c28ff0ab5c704e3d893`, package 1.3.0.
This is a correction within Phase 12, not closure or a new research experiment.
Publication approval and a fresh owner Windows/CUDA retest remain required.

## External failure and evidence scope

The owner reports **FAILED at the first project failure** on Windows with the
RTX 4070 SUPER. The supplied detailed handoff message is the source; no separate
raw Windows logs or files were available in this Mac workspace.

- Fresh exact published checkout and locked Windows installation passed.
- Complete prescribed checkout suite: **299 passed, 13 expected skips, zero
  failures/errors**. The inventory failure happened after that suite.
- CUDA availability/device detection succeeded. Actual Phase 12 CUDA execution
  was **not reached**; no CUDA model outcome or fresh-wheel suite is inferred.
- All 229 tracked files and 3,045 prior evidence/report files remained unchanged;
  reserved payloads untouched; no Windows fix, commit or push.
- Read-only Windows diagnosis found every referenced file present with matching
  byte count and SHA-256. The failure was path representation, not corruption.

The final inventory combined native relative strings with literal
`256/artifacts.json` and `512/artifacts.json`. On Windows the verifier's native
relative strings used backslashes, so exact key-set comparison raised
`EvidenceError: Inventory is incomplete`. Dependency installation, CUDA, model
quality and missing artifacts are not the reported failure category.

## Correction

Write every serialized relative path with `as_posix()`: file inventory keys,
captured-source keys and input-file keys. The verifier constructs its expected
inventory keys using the same platform-independent representation. Joining those
relative forward-slash keys with a local Path remains native filesystem access.
No model/evaluation behavior, scoring, hypothesis, context limit or tolerance changes.

Inventory construction now excludes only the inventory at the current root,
including nested condition inventories automatically. The final writer uses that
same inventory function rather than inserting separately formatted keys by hand.
The original Mac inventories retain identical path keys, sizes and hashes under
the corrected construction. No existing evidence file was normalized or rewritten.
Broken historical Windows inventories should be retained as failure evidence;
new correction validation outputs belong in a fresh directory.

Two regression variants exercise the actual final writer and verifier with real
files under both `256` and `512`, including nested source files and both inventories.
One uses native paths; the other supplies PureWindowsPath relative values while
keeping real local I/O. Both run on any supported host; this is not Windows execution
on the Mac. They check exact portable keys, complete membership, all byte/hash checks,
condition inventories, repeat writing without self-inclusion and corruption rejection.
The simulated-Windows test reproduced the exact published `Inventory is incomplete`
failure before the fix (native passed), using only mechanical extraction of the
published final writer. Both pass after correction. The earlier flat regression stays.

## Validation and preservation

Mac development and fresh isolated non-editable wheel suites each passed **314 tests**;
nine focused tests passed, including the native and simulated-Windows final inventory
variants. Locked dependency identity, lint/format and source/wheel builds passed.
**78,404 prior files remained unchanged**; only the two authorized working experiment
source files differ in the 78,406-file preservation snapshot. Both reserved payloads
were opaque-hashed only. These are local correction checks, not a Windows/CUDA PASS.

[Structured correction evidence](portability-correction.json) records actual local
checks, log identities and preservation. The original [report](REPORT.md), [plan](PLAN.md),
results, samples, verification and validation remain the accepted historical evidence.
A [separate review](PORTABILITY_REVIEW.md) checks the minimal scope and retest gate.

The corrected inventory checker validates all six inventories from both original
Mac runs. Rebuilding their inventory dictionaries in memory produces exactly the
saved dictionaries, without writing to them. Running each original verifier from
its own captured source reproduces its original saved replay/recount result.
There is **no new research evaluation, training or model generation**. Engineering
regressions still exercise their existing tiny CPU/MPS fixtures.

The full verifier's current-source identity guard remains strict: the corrected
checkout intentionally rejects an old run at `Current source differs`, because the
runner, verifier and tests have changed. Do not rehash old plans or bypass that guard
to make the corrected checkout claim it generated historical outputs. To verify a
historical run end-to-end, use its exact original checkout/captured source and original
tokenizer. New synthetic retest evidence must capture the corrected source identities.
This source-identity rule is separate from the repaired path representation defect.

The experimental result remains exactly: extending the session budget from 256 to
512 let all 44 previously blocked sessions retry, but produced no successful learned
tool use. Capacity relief is not learned capability improvement. All original model,
tokenizer, chat format, scoring, context/tolerance and reserved-data limits remain.
Distinct Phase 11 negative full Mac and compact-prompt tiny Windows evidence,
separately scoped hosted Linux evidence and prior negative SFT/LoRA/DPO results remain.
No new hosted Linux CI result is claimed; physical Linux remains deferred.

## Publication and fresh external retest

Prepare a reviewed local correction commit for existing
`https://github.com/sebastienlato/LatoS.git`, `main` only. No tag, release, asset upload,
visibility change, paid service or dependency upgrade. After explicit approval and
verified publication, wait for the owner's fresh independent Windows/CUDA validation
on the RTX 4070 SUPER. Neither Phase 12 closure nor a subsequent experiment is authorized.

The retest should use an exact corrected checkout and locked install; run the full
prescribed suite (including both final-inventory variants), then the existing bounded
external protocol with new output directories and corrected source identities.
Preserve the failed run, all earlier reports/artifacts and reserved payloads. Report
engineering evidence separately from any actual CUDA execution and from learned
capability. A passing portability regression alone is not completion of that retest.
