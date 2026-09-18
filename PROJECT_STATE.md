# Project state

Updated: 2026-09-18.

## Active phase

Phase 3 — Windows environment correction complete locally; push approval pending.
Phase 4 remains paused and is not authorized.

The published Phase 3 commit is `deed6e9b0303967dd82f7a3205e03b9c4f05ffe7`,
fixed at v0.4.0. Owner validation on Windows 11 Home 25H2 / NVIDIA RTX 4070 SUPER
reported an unsupported lock environment during `uv sync --locked`. PyTorch was
not installed and no CUDA/model tests ran. This is a dependency configuration
blocker, not evidence of a Transformer or CUDA failure.

## Correction and evidence

- Package 0.4.1 adds Windows x86-64 with explicit official PyTorch 2.14.0+cu130.
  Both native AMD64 and cross-target x86_64 markers are covered; other Windows
  architectures remain excluded. uv 0.12.15 and Python 3.14.7 remain pinned.
- macOS retains PyPI PyTorch 2.14.0; Linux retains official 2.14.0+cpu. All 34
  external packages on each existing platform keep their versions and sources,
  and every original wheel URL/hash is preserved. Windows-only Colorama is locked.
- No `src/` code, model settings, numerical thresholds, or CI jobs changed.
  Associated tests now handle an absent Unix memory API and read UTF-8 explicitly.
- 128 tests passed in the Mac environment and a fresh installed-wheel environment.
  Lint/format, dependency checks, builds, CPU/MPS doctor and pilot checks passed.
- All three targeted locked install dry runs pass; the complete dependency graphs
  have hashed CPython 3.14 wheels. These checks ran on Mac, not Windows/Linux.
- Windows installation and real CUDA remain unexecuted here; the owner must retest
  the new exact commit on the RTX 4070 SUPER. External native Linux results are pending.
- Evidence and retest procedure: [correction report](experiments/phase-3/WINDOWS_CORRECTION.md),
  [setup](docs/SETUP.md). Original model evidence remains in the Phase 3 report.

## Publication history and proposed correction

Private origin: `https://github.com/sebastienlato/LatoS.git`.
Phases 0–2 and Phase 3/v0.4.0 were explicitly approved and published. The original
[Phase 3 Linux CPU CI](https://github.com/sebastienlato/LatoS/actions/runs/35368714640)
passed. It does not replace external native Windows/CUDA or Linux validation.

- Local branch: main; checkpoint message: `Fix Phase 3 Windows CUDA dependency lock`.
  The correction is the commit containing this state; its full ID is given in the
  approval request and by `git log -1 --format=%H` at that checkpoint.
- Proposal: push this reviewed correction to private sebastienlato/LatoS main and
  create/push annotated tag v0.4.1. Leave v0.4.0 fixed. No releases or artifact uploads.
- Correction push approval: **pending**. No correction has been pushed.
- Preserve ignored `checkpoints/phase-3-pilot-initial/`,
  `artifacts/tokenizers/english-bpe-v1/`, and original raw/prepared corpora.

## Next action and external gate

Stop for owner approval before pushing. If approved, publish only the described
correction, verify the exact remote commit/tag, then PAUSE for the owner's Windows
RTX 4070 SUPER and native Linux retests. Those machines are validation-only and
will not modify or push the repository. Address any supplied Phase 3 failure
before progression. Start Phase 4 only when both environments pass AND the owner
explicitly authorizes it; no earlier automatic-transition instruction overrides
this gate. Do not infer authorization from a push approval or CI result.

Local uv: `.private/tools/bin/uv`; `.venv` uses the runtime under `.private/python`.
