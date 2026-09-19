# Phase 3 closure

Recorded 2026-09-19. Phase 3 implementation and validation are closed locally;
publication of this documentation-only closure requires owner approval. Package
version remains 0.4.2. Phase 4 has not begun and must not begin in this chat.

## Validated implementation

- Tag: `v0.4.2`.
- Exact commit: `1102b64714b58c1f2289f80863fa032cc192c47b`.
- Private repository: [sebastienlato/LatoS](https://github.com/sebastienlato/LatoS).
- The closure commit changes documentation/state only; it is not a newly tested
  implementation. Keep v0.4.0, v0.4.1, and v0.4.2 fixed. No new tag is proposed.

## Evidence by environment

| Environment | Result and evidence source | Scope and limits |
| --- | --- | --- |
| Local macOS CPU/MPS | Previously executed and recorded: 132 tests passed in development and clean installation | Numerical/model checks and checkout regression; see the implementation and correction reports |
| Independent Windows / RTX 4070 SUPER CUDA | **PASS**, reported by the owner in this conversation | Exact v0.4.2; 131 tests passed, one Apple-MPS-only skip; real CUDA execution described below |
| GitHub-hosted Linux CPU CI | **PASS**, workflow and logs directly inspected | Exact v0.4.2; Ubuntu 24.04 runner, Python 3.14.7, PyTorch 2.14.0+cpu; 131 passed, one MPS-only skip |
| Independent physical Linux machine | **DEFERRED — NOT PERFORMED** | Owner says the machine is temporarily unavailable; do not record this as a pass or equate it with CI |

### Windows/CUDA owner report

The owner reports that the independent validation machine completed:

- Fresh checkout and locked installation.
- 131 passing tests and one Apple-MPS-only skip.
- Actual RTX 4070 SUPER CUDA execution, including matrix multiplication/backward.
- Debug and pilot transformer forward, loss, gradients, and causality checks on CUDA.
- CLI, CPU/automatic diagnostics, offline data/tokenizer workflows, builds, and archive inspection.
- All 74 tracked files unchanged, with no fixes, commits, or pushes from Windows.

The machine description supplied earlier in the same validation sequence was
Windows 11 Home 25H2, Python 3.14.7, PyTorch 2.14.0+cu130, driver 616.92, and PyTorch
CUDA runtime 13.0. The PASS is an owner-supplied external result, not execution
performed by this Mac session. Raw Windows logs and numerical measurements were
not supplied with the summary; none are invented here.

The earlier failures remain in their historical reports: v0.4.0 could not install
the Windows environment, and v0.4.1 installed/detected CUDA but failed a byte-pinned
fixture after Git checkout conversion. The v0.4.2 PASS supersedes those blockers
for this exact validated checkpoint without rewriting their evidence.

### GitHub Linux CI observed separately

[Run 35463376332](https://github.com/sebastienlato/LatoS/actions/runs/35463376332)
completed successfully on 2026-09-19 at commit
`1102b64714b58c1f2289f80863fa032cc192c47b`. Its logs report 131 passed and one skipped
test. Inspected successful steps cover locked installation, Ruff lint/format,
pytest, CPU doctor, the debug model CPU diagnostic, offline data preparation/audit,
package builds, and offline tokenizer fitting/validation.

This is a hosted Linux CPU job, not an independent physical Linux-machine test,
not a Linux CUDA test, and not a pilot CLI benchmark. The published workflow runs
the debug model CLI check. Do not extend its claims to unexecuted environments.

## Closure scope and remaining limitations

The Phase 3 deliverables are complete: original dense transformer, numerical and
causality checks, next-token loss, tokenizer binding, model-only snapshots, bounded
sampling, and the two checkout/environment corrections. No unresolved Phase 3
failure is reported after the Windows PASS.

The owner explicitly deferred physical Linux validation and requested Phase 3
closure with that limitation. This replaces the earlier requirement to wait for
both physical-machine passes for this closure; it does not turn the deferred test
into a pass, authorize Phase 4 here, or grant ongoing access to the Windows GPU.

Weights remain randomly initialized. There is no model training/quality result,
training engine, optimizer/resume checkpoint, mixed-precision validation, KV cache,
or chat interface. External validation machines remain validation-only. Existing
resources may be used in a later authorized phase; the new paid-service budget is zero.

## Documentation checks and publication

This closure updates the evidence, state, current support statements, workflow
transition rule, and [fresh-chat handoff](../../docs/PHASE4_HANDOFF.md). It changes
no runtime source, tests, model/tokenizer configurations, fixture bytes, lockfile,
package version, or CI workflow. No redundant model/test/training run is claimed.
Review verifies internal links, consistent commit/status/test counts, documentation-only
scope, retained local artifact identities, and absence of private planning material.
The machine-readable summary is [closure.json](closure.json).

Proposed publication: push the reviewed closure commit to existing private main,
without creating or moving tags, releases, or artifact uploads. Stop for approval
first. After any approved publication, verify main and the unchanged validated
tag, record the result, and stop. Begin Phase 4 only in a fresh Work chat with
explicit owner authorization after the closure publication is verified.
