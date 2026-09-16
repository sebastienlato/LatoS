# Decisions

## 2026-09-16 — Foundation scope

Implement an installable `src/latos` package with standard-library argument parsing
and a PyTorch doctor. Defer model and data modules until their acceptance criteria
can be tested. Original code uses MIT; data and future weights require independent
terms. Package-index name availability has not been checked; no PyPI publication
is proposed.

## 2026-09-16 — Stable environment

Use CPython 3.14.7, PyTorch 2.14.0, NumPy 2.5.3, uv 0.12.15, pytest 9.1.1,
Ruff 0.16.8, and Hatchling 1.32.0, selected from current official releases and
package metadata. Bound Python to the 3.14 series and pin the local patch version.
NumPy is included to verify PyTorch array interchange. Tokenizer and interface
dependencies are deferred. The lock records all transitive versions and hashes.
Upgrade deliberately between experiments and rerun checks.

Resolve PyTorch from PyPI on macOS and the explicit official CPU index on Linux.
Limit the initial lock to Apple Silicon macOS and x86-64 Linux. This avoids a large
CUDA dependency download in CPU CI. CUDA runtime detection is implemented, but
CUDA installation and execution support are deferred until hardware is available.
Pin the build backend as well as runtime/development tools; build using the synced
environment. Source archives use an explicit inclusion list and are inspected.

## 2026-09-16 — Diagnostics and execution claims

Report advertised availability separately from a completed arithmetic/gradient
check. Explicit device requests must fail when unavailable. Auto selection uses
CUDA, then MPS, then CPU, with no retry on another device after an execution error.
This order is a policy, not a performance result. Float32 smoke checks do not
establish training correctness, numerical equivalence, or throughput.

## 2026-09-16 — Publication and CI

Keep this checkpoint local until approved. The initial proposal is a private
GitHub repository, `main`, and tag `v0.1.0`; no release or artifact upload is part
of it. CPU CI uses read-only permissions, pinned action commits, a 15-minute job
limit, and no full training. It has not run on GitHub before publication. Do not
buy Actions capacity or enable paid resources; an unavailable included quota is
a publication-time limitation to report.
