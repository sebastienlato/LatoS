# Project state

Updated: 2026-09-16.

## Active phase

Phase 0 — foundation complete locally; publication approval pending.

Implemented an installable `latos` package, help/version/doctor CLI, explicit backend
selection with float32 execution checks, pinned environment and lockfile, tests,
CPU CI definition, setup/evidence/provenance docs, MIT license, and phased roadmap.
The starter instructions and ignore rules were preserved.

## Evidence and limits

- 22 tests passed in both the development environment and a clean environment
  using the built wheel outside the source directory.
- CPU and MPS arithmetic/gradient checks passed on macOS arm64, Apple M4 Max,
  64 GiB RAM; Python 3.14.7 and PyTorch 2.14.0.
- Lint, format, dependency compatibility, build, and archive checks passed.
- Separate review completed; float32 default-dependence and formatting findings
  fixed. Publication content inspected for private material.
- Linux CI has not executed remotely; CUDA and other platforms are untested.
  No model, data pipeline, training, or language-quality result exists yet.
- No implementation blocker. Detailed evidence: [environment report](docs/ENVIRONMENT.md).

## Pending publication

- Local branch: `main`; checkpoint commit message: `Complete Phase 0 foundation`.
  The checkpoint is the commit containing this Phase 0 state; its exact object ID
  is given in the approval request (inspect `git log -1 --format=%H` at this checkpoint).
- No remote configured and no GitHub writes performed.
- Read-only GitHub checks authenticated `sebastienlato`; repository lookup returned
  404 and the account's repository listing had no matching name on 2026-09-16.
- Proposal: create **private** `sebastienlato/LatoS`, add `origin` at
  `https://github.com/sebastienlato/LatoS.git`, push the reviewed checkpoint to
  `main`, and create/push annotated tag `v0.1.0` at that same commit.
- No release, package publication, archive, or model upload proposed.
- Approval: **not yet granted**. Do not interpret development authorization as
  permission to write to GitHub. Recheck destination before creating it.

## Next action

Await the owner's explicit response to “Push Phase 0 to GitHub?” After approval,
publish the exact checkpoint and verify remote branch and tag identities, inspect
the first CI result without enabling paid resources, then begin Phase 1 English
data development. Phase 1 needs its own publication approval.

The project-local bootstrap uv executable is `.private/tools/bin/uv`; the Python
runtime is under `.private/python`. `uv run --locked` uses the existing `.venv`.
Use the setup guide if these ignored environments are absent in another checkout.
