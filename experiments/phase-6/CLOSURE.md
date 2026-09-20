# Phase 6 closure

Recorded 2026-09-20. Phase 6 engineering implementation and validation are closed
locally. This documentation-only closure awaits explicit publication approval.
**Do not begin Phase 7 in this chat**, including after closure publication.

## Validated checkpoint

Published annotated `v0.7.0`, commit `f6636af34933b9678824cc8dd50f6ab6559a2de5`,
package 0.7.0. Remote main and the peeled tag were checked again before closure;
both resolve to that exact accepted commit. Tag object:
`bdbaa5499847ae7080e2b32bc96c5400ed14df80`. Preserve all existing tags unchanged.

This closure changes documentation/evidence only. Runtime source, tests, experiment
scripts, configurations, dependency lock, package version and CI workflow are
unchanged. The original measured results, samples, validation snapshot and artifact
inventory retain their recorded scope; this closure adds subsequent evidence.

## Evidence by environment

| Environment | Evidence source and result | Actual scope |
| --- | --- | --- |
| Mac CPU/MPS | Previously recorded: 187 tests passed in development and fresh non-editable wheel installation | Full fixed 200-update / 6,188-assistant-target SFT experiment, preserved base comparison and English regression |
| Independent Windows / RTX 4070 SUPER | **PASS**, owner's external summary at exact v0.7.0 | 184 passed, three MPS-only skips, zero failures; fresh wheel suite; bounded four-update / 175-assistant-target CUDA exercise |
| GitHub-hosted Linux CPU | **PASS**, run/job metadata, exact-commit workflow and logs directly inspected | 184 passed, three MPS-only skips; tiny CPU SFT runner/verifier tests and Phase 4 CPU fixture acceptance |
| Independent physical Linux | **DEFERRED — NOT PERFORMED** | Hosted CI is separate evidence |

### Owner-reported Windows/CUDA PASS

The owner reports the following on the NVIDIA RTX 4070 SUPER at the unchanged
validated commit:

- Fresh checkout and locked installation passed; **184 tests passed, three
  MPS-only skips, zero failures**. The complete suite also passed in a fresh
  non-editable wheel installation.
- Shared chat serialization, assistant-only masking and next-token alignment
  passed. Ignored prediction positions had zero direct loss gradient while
  assistant targets remained correctly trainable.
- Actual Phase 6 SFT completed **four optimizer updates and 175 assistant-target
  exposures** on CUDA. Accumulation, scheduling, active gradient clipping, finite
  gradients/optimizer state and validation-state preservation passed.
- Checkpoint creation/loading, independent verification and recovery passed.
  All **49 artifact files** verified and all **64 base/SFT held-out responses**
  were independently replayed. Assistant evaluation and English-regression
  measurement mechanisms passed.
- Reserved instruction and English test payloads stayed excluded from training
  and evaluation, and unchanged. All **326 prior Phase 5 evidence files** were
  preserved on that Windows installation.
- CLI, lint, formatting, dependency integrity, diagnostics, offline data/tokenizer
  workflows, builds, packaging and privacy checks passed.
- The commit and all **128 tracked files** remained unchanged, the working tree
  stayed clean, and nothing was committed or pushed from Windows.

Attribution: this is the owner's independent validation summary, not CUDA execution
by this Mac session. Raw Windows logs, exact local artifact/configuration identities,
driver/runtime details and numerical recovery tolerances were not supplied here;
none are inferred. The Windows evidence-file count is specific to that installation,
not a replacement for the Mac preservation inventory.

This bounded exercise **did not reproduce the full Mac 200-update / 6,188-target
experiment or its learned artifact**. Both tiny Windows models scored **0/32 exact
held-out replies**. It verifies the reported engineering mechanisms, not useful
instruction following, general CUDA determinism, cross-device numerical equality,
mixed-precision/distributed behavior, sustained performance or broader capability.
No unreported exact CUDA equality guarantee follows from the PASS.

### GitHub Linux CPU CI verified separately

[Run 35522033012](https://github.com/sebastienlato/LatoS/actions/runs/35522033012)
was triggered by the implementation push at
`f6636af34933b9678824cc8dd50f6ab6559a2de5`. The successful CPU job
[106107655953](https://github.com/sebastienlato/LatoS/actions/runs/35522033012/job/106107655953)
completed on 2026-09-20 at **16:14:24 UTC**; run status became completed/success at
16:14:25 UTC. The workflow targets Ubuntu 24.04. Logs identify Python 3.14.7 and
PyTorch 2.14.0+cpu, with **184 passed / three MPS-only skips / zero failures**
from 187 collected tests (22.93 seconds).

Successful steps: locked installation, Ruff lint/format, pytest, CPU doctor and
debug model diagnostics, offline data acquisition/preparation/audit, source/wheel
builds, offline tokenizer fitting/validation, and offline CPU training acceptance.
The exact-commit tests and workflow were inspected to establish scope:

- `tests/test_instruction.py`: 19 CPU checks passed; one MPS case skipped. Includes
  format/mask/alignment, padding, direct loss gradients, overflow rejection,
  split/test isolation, token weighting, read-only validation and CPU recovery.
- All three `tests/test_phase6.py` cases passed. They exercise a **two-update tiny
  randomly initialized CPU base**, with both test files removed, the SFT runner,
  base/final evaluation and samples, inventory/checkpoint checks, verifier function
  replay, overwrite refusal, injected failure retention and wrong-base rejection.
  The verifier is called inside pytest, not as an independent-process CLI job.
- The existing suite also covers the earlier engine's separate-process CPU CLI
  recovery. No new CUDA or MPS run follows from the CPU test result.
- The explicit acceptance step runs `experiments/phase-4/validate.py` at the Phase 6
  commit: 127,808 parameters, **400 updates / 131,600 all-target exposures**. Train
  loss 5.8146375958 → 0.0100304652; fixture validation loss
  5.8069872746 → 11.2451782983. Resume from update 97 matched the remaining 303
  non-timing metrics and final weights, optimizer and sampler exactly on that run.
  Validation preserved weights, optimizer, RNG and gradients and restored mode.
  This is fixture memorization, not English or instruction capability.

CI did **not** reproduce the full Mac SFT experiment or learned checkpoint, run
CUDA/MPS, validate physical Linux, or execute the complete suite from a fresh
non-editable wheel installation. Building a wheel does not establish the latter.
There is no standalone Phase 6 verifier CLI step or separate archive/privacy audit
in this workflow. Fixture test-data preparation/audit is distinct from using the
reserved instruction/English test payloads for training or model selection.

## Closure decision and retained quality limits

The Phase 6 engineering deliverable and requested independent Windows/CUDA
validation passed within the above scopes. No unresolved failure was reported.
The negative quality result remains unchanged: Mac assistant loss
**8.354879 → 5.815233**, exact final replies **0/32 → 0/32**, EOS stops **0/32 →
32/32**, and English validation loss **4.731898 → 5.653422**. Keep the predeclared
final update 200 and every earlier attempt; do not promote the SFT artifact as an
improved base or useful assistant. See [REPORT.md](REPORT.md), [results.json](results.json)
and [samples.json](samples.json). External checks do not expand these claims.

Physical Linux remains deferred. Prior datasets, tokenizer, base, SFT checkpoints,
source snapshots, failed attempts and logs remain preserved and ignored. External
GPU validation does not provide this session access to that machine. No new paid
service, learned-artifact upload or Phase 7 implementation is part of closure.

## Separate closure review and publication

A separate documentation review checked exact refs/counts, attribution, actual CI
scope, unchanged quality limits, hashes, relative links, privacy and the fresh-chat
gate. All **49 SFT inventory files**, the selected Phase 5 base, tokenizer and
random baseline were rehashed unchanged; the package implementation fingerprint
still matches the measured run. JSON, links, whitespace and privacy/history checks
passed. No actionable finding remains. **No new runtime tests, training or builds
were run for this documentation-only closure.**

Publish only the reviewed closure commit to existing private
`https://github.com/sebastienlato/LatoS.git`, branch `main`, after explicit approval.
**No new tag, old-tag movement, release or artifact upload.** Package stays 0.7.0.
After publishing, verify the exact closure commit and all unchanged tags, record
verification locally, and **stop**. Closure approval does not authorize Phase 7.

Phase 7 requires an explicit start in a fresh Work chat after closure publication
has been verified. [PHASE7_HANDOFF.md](../../docs/PHASE7_HANDOFF.md) contains the
preserved inputs, planned acceptance checks and concise reusable prompt. It is
preparation only; Phase 7 has not begun.
