# Phase 0 environment and validation

Recorded on 2026-09-16 against the actual execution host. These are observations
from local commands, not inferred specifications of a different machine.

## Host

| Property | Observed value |
| --- | --- |
| OS | macOS 26.6.2, build 25G83 |
| Architecture | arm64 |
| CPU | Apple M4 Max, 16 logical CPUs |
| Physical memory | 68,719,476,736 bytes (64 GiB) |
| Free workspace disk | Approximately 631 GiB after environment installation; changes over time |
| System Python at inspection | 3.14.4; left unchanged |
| Project Python | CPython 3.14.7, uv-managed local runtime |
| PyTorch | 2.14.0 |
| CPU | Available; float32 matrix product and backward check passed |
| MPS | Built and available; same check passed without fallback |
| CUDA | Unavailable; build version null; execution untested |

Memory is total physical memory, not a training allocation budget. No throughput,
peak model memory, model size, or training duration has been measured.

## Dependency selection

Checked official Python downloads/support status, PyTorch release notes and MPS
documentation, uv documentation, and live PyPI JSON metadata on 2026-09-16.
PyTorch metadata advertised Python >=3.10 and a macOS arm64 CPython 3.14 wheel.
The latest stable selections resolved and installed successfully:

| Component | Version |
| --- | --- |
| CPython | 3.14.7 |
| PyTorch | 2.14.0 (macOS); 2.14.0+cpu locked for Linux |
| NumPy | 2.5.3 |
| uv | 0.12.15 |
| pytest | 9.1.1 |
| Ruff | 0.16.8 |
| Hatchling | 1.32.0 |

The lock contains 23 entries including the project and two platform variants of
PyTorch; 22 distributions are installed locally. Exact transitive versions and
sources are in `uv.lock` and [THIRD_PARTY.md](../THIRD_PARTY.md).
The installation selector contained an older displayed stable label, so version
selection was cross-checked against the official 2.14.0 release and wheel metadata.

The uv Python download emitted an install-name patch warning for native extension
builds. The installed binary, prebuilt PyTorch/NumPy wheels, package builds, and
all listed checks worked. Compiling new native extensions is untested. No global
Python or package environment was replaced. The bootstrap-created external Python
symlink was removed; the downloaded runtime stays within the ignored project area.

## Executed checks

| Check | Result |
| --- | --- |
| Initial environment install and subsequent locked sync | Passed |
| Installed package import and version | 0.1.0 |
| CLI help, subcommand help, module entry point | Passed |
| `pytest` in development environment | 22 passed |
| Clean environment with locked dependencies and built wheel, outside source directory | 22 passed; import verified from site-packages |
| CPU doctor | Passed float32 forward and backward values |
| MPS doctor and automatic device selection | Passed; auto selected MPS |
| Explicit unavailable CUDA request | Correct JSON error, exit code 1, no fallback |
| Missing tensor runtime, invalid command, simulated accelerator execution failure | Passed failure-path tests |
| NumPy/PyTorch interchange | Passed |
| Ruff lint and format checks | Passed after format correction |
| `uv pip check` | All 22 installed packages compatible |
| Source distribution and wheel build | Passed; wheel built from source distribution |
| Build member inspection | Only intended source/docs/tests/configuration and package metadata; no private material or environments |
| Workflow YAML | Parsed locally; 8 steps; pinned upstream action commits verified read-only |

The separate review covered device selection, false success paths, numerical
expectations, CLI exit codes, package contents, dependency sources, CI configuration,
and documentation claims. It found an implicit default dtype in the smoke check;
the implementation now explicitly uses float32 and a regression test changes the
global default to verify it. Ruff's requested formatting change was also applied.

## Limits

Linux CPU CI is defined but has not run on GitHub. CUDA, Windows, Intel macOS,
Linux ARM, mixed precision, native extension compilation, and full model/training
operations are untested. A simulated backend test proves selection/error handling,
not hardware execution. No datasets, tokenizer, weights, training, paid compute,
or remote publication were involved in Phase 0.

## Publication follow-up — 2026-09-16

After the owner's approval, Phase 0 commit
`a33d84d0200a3e46879bc506a3f8c9a82db2ff7c` was published to private
`sebastienlato/LatoS`, with `main` and annotated tag `v0.1.0` verified remotely.
[Linux CPU CI](https://github.com/sebastienlato/LatoS/actions/runs/35145730007)
passed: 22 tests, lint/format, CPU doctor, and package build. Its runtime reported
Python 3.14.7 and PyTorch 2.14.0+cpu. The statements above about unexecuted Linux
CI describe the original pre-publication checkpoint. Phase 1 validation is
recorded separately in [its experiment report](../experiments/phase-1/REPORT.md).
