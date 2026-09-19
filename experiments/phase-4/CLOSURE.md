# Phase 4 closure

Recorded 2026-09-19. Phase 4 implementation and validation are closed locally.
This documentation-only closure awaits explicit publication approval. Phase 5 has
not begun; closure publication does not authorize starting it in this chat.

## Validated checkpoint

Tag `v0.5.0`, commit `cb585c321c92f5d774fb59234f76c1d3783a635a`, package 0.5.0.
Remote main and the annotated tag were verified at this commit after the approved
implementation publication and checked again before preparing this closure.
The tag remains fixed. No runtime source, tests, configuration, lockfile, package
version, or CI workflow changes are included in this closure.

## Evidence by environment

| Environment | Result and evidence source | Scope |
| --- | --- | --- |
| Local macOS CPU/MPS | Previously recorded: 161 passed in development and clean wheel installation | CPU fixture overfit/exact resume, validation isolation, short MPS recovery smoke; [implementation report](REPORT.md) |
| Independent Windows / RTX 4070 SUPER | **PASS**, owner-reported external validation of exact v0.5.0 | 159 passed, two MPS-only skips, zero failures; actual CUDA training and separate-process recovery |
| GitHub-hosted Linux CPU | **PASS**, completed run metadata and logs directly inspected | 159 passed, two MPS-only skips; CPU fixture overfit/exact resume and workflow checks below |
| Independent physical Linux | **DEFERRED — NOT PERFORMED** | Not a pass; hosted CPU CI is separate evidence |

### Windows/CUDA owner report

The owner reports successful fresh checkout and locked installation, retaining
exact commit `cb585c321c92f5d774fb59234f76c1d3783a635a` throughout. Validation covered:

- 159 tests passed, two MPS-only tests skipped, zero failures.
- Actual Phase 4 training on the NVIDIA RTX 4070 SUPER, including CUDA
  forward/backward and AdamW optimizer updates.
- Scheduling, gradient accumulation, clipping, finite gradients, and validation
  state preservation.
- CUDA fixture training loss 5.814638 → 0.010030 over 400 updates.
- CPU exact-resume requirements, plus CUDA checkpoint save/load and separate-process resume.
- CLI workflows, CPU/automatic diagnostics, offline data/tokenizer workflows,
  lint, formatting, dependency integrity, builds, and archive inspection.
- Clean working tree and all 91 tracked files unchanged; no commits or pushes from Windows.

This is the owner's external summary, not CUDA execution by this Mac session.
Raw Windows logs and exact per-run runtime/driver versions were not supplied with
this Phase 4 summary; no new versions or numerical measurements are inferred.
Exact CUDA equality was observed in this particular validation. It is **not a
general CUDA determinism guarantee** and does not extend the documented CPU
same-host/runtime exact-resume contract to arbitrary CUDA environments.

### GitHub Linux CPU CI verified separately

[Run 35471367201](https://github.com/sebastienlato/LatoS/actions/runs/35471367201)
completed successfully on 2026-09-19 at the exact validated commit. Run metadata,
workflow definition, and logs were inspected read-only. Ubuntu 24.04 hosted runner,
Python 3.14.7, PyTorch 2.14.0+cpu; 159 tests passed and two MPS-only tests skipped.

Successful steps: locked installation, Ruff lint/format, pytest, CPU doctor, debug
model CPU check, offline data acquisition/preparation/audit, source/wheel builds,
offline tokenizer training/validation, and the Phase 4 offline training acceptance.
The test suite includes separate-process CLI recovery. The acceptance log reports:

- CPU training loss 5.8146375958 → 0.0100304652 over 400 updates.
- Resume from update 97: all 303 remaining non-timing metrics, final weights,
  optimizer tensors, and sampler state exactly equal within that run.
- Validation left weights, optimizer, gradients, and RNG unchanged and restored mode.
- Held-out loss 5.8069872746 → 11.2451782983.

This run does not verify Windows, CUDA, MPS, independent physical Linux, production
English pretraining, or large-model throughput. Archive inspection and dependency
integrity beyond the workflow's actual steps are not separately claimed for CI.

## Closure and remaining limits

The training-engine deliverables and independent Windows/CUDA checkpoint have
passed; no unresolved Phase 4 failure is reported. Held-out loss worsening remains
fixture memorization, not language-quality improvement. The 17,308,032-parameter
English pilot remains randomly initialized. No Phase 5 run was started.

Physical Linux stays deferred under the owner's instruction. The external NVIDIA
machine is validation evidence, not authorization to use its compute. No paid
services, remote artifact uploads, new tags, or changes to existing tags are proposed.

## Review and publication

Closure review checks evidence attribution, exact commits/counts, CI step scope,
CUDA reproducibility limits, the physical Linux deferral, internal links, private
content exclusion, and the Phase 5 transition gate. The existing tokenizer and
pilot snapshot identities are rechecked locally. No redundant training or runtime
test execution is required or claimed for this documentation-only change.
[closure.json](closure.json) is the machine-readable evidence summary.

Proposed publication: only the reviewed closure commit to existing private
`https://github.com/sebastienlato/LatoS.git`, branch `main`, after explicit approval.
No new tag; keep v0.5.0 at the validated implementation. After an approved push,
verify exact remote main and the unchanged tag, record publication locally, and
stop. Phase 5 requires an explicit start in a fresh Work chat after closure
publication is verified. [The handoff](../../docs/PHASE5_HANDOFF.md) supplies the
starting context and a reusable prompt; it is not authorization to execute now.
