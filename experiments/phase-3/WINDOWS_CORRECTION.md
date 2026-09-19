# Phase 3 Windows environment correction

Historical correction evidence. The final v0.4.2 Windows/CUDA PASS and the owner's
physical Linux deferral are recorded in [CLOSURE.md](CLOSURE.md), which supersedes
the pending validation and transition instructions below.

Recorded 2026-09-18. Base checkpoint:
`deed6e9b0303967dd82f7a3205e03b9c4f05ffe7`, tag `v0.4.0`.
Local correction package version: 0.4.1. This remains Phase 3; Phase 4 is paused.

## Reported blocker

The owner tested the published checkpoint on Windows 11 Home 25H2 with an NVIDIA
RTX 4070 SUPER, driver 616.92, driver-reported CUDA 13.4, uv 0.12.15, and Python
3.14.7. `uv sync --locked` rejected Windows because the allowed lock environments
were only macOS ARM64 and Linux x86-64. PyTorch was not installed, so neither CUDA
nor model tests ran. This is packaging evidence, not a Transformer/CUDA failure.
The actual RTX 4070 SUPER supersedes the previously planned RTX 4080 description.

The same platform rejection was reproduced on Mac using uv's Windows-targeted
dry run against the original configuration. The original Phase 3
[Linux CPU CI](https://github.com/sebastienlato/LatoS/actions/runs/35368714640)
passed; it did not validate Windows or the owner's external Linux machine.

## Correction and preservation

- Add Windows x86-64, covering native Python's `AMD64` and uv 0.12.15's observed
  cross-target `x86_64` marker. An AMD64-only constraint still rejected the
  cross-target dry run; both spellings now resolve the same Windows wheels.
- Route Windows PyTorch explicitly to the official CUDA 13.0 index. Keep the
  base requirement `torch==2.14.0` and lock the exact Windows variant
  `2.14.0+cu130`. Simply admitting Windows while falling back to PyPI would not
  establish the intended CUDA build. No index override or unlocked install is needed.
- Preserve the macOS PyPI build and Linux CPU index. All 34 external packages
  reachable on each existing platform retain their versions/sources, and every
  original wheel URL/hash is retained. Windows adds its compatible wheels and
  Colorama 0.4.6, a Windows-only pytest/tqdm dependency. No existing dependency
  was upgraded. uv 0.12.15 and Python 3.14.7 remain pinned.
- Add offline regression checks for environment admission, backend selection,
  complete CPython 3.14 binary-wheel coverage and hashes across dependency graphs,
  explicit-index isolation, and rejection of other architectures.
- Fix two test-only portability assumptions: permit monkeypatching the absent
  Unix `os.sysconf` API, and explicitly decode tokenizer JSON as UTF-8. Its byte
  alphabet cannot be decoded under Windows cp1252. Add a missing-attribute test.

No file under `src/` changed. Model equations, numeric tolerances, sampling,
snapshot formats, existing artifacts, and CI configuration are unchanged. The
original tag must remain fixed; proposed correction tag is `v0.4.1` at the new
reviewed commit. No remote write is authorized yet.

## Validation evidence

- Local locked sync, dependency compatibility, lint/format, and package builds pass.
- 128 tests pass on the actual Mac, including the existing model/MPS regressions.
- The same 128 tests pass in a fresh locked Mac environment using the built wheel
  outside the source directory. Package inspection excludes private data and weights.
- Locked install dry runs pass for Windows x86-64, Linux x86-64, and macOS ARM64.
  The synthetic Mac target explicitly uses `MACOSX_DEPLOYMENT_TARGET=14.0` because
  uv otherwise defaults to macOS 13; PyTorch already required a macOS 14 wheel.
- Native Windows and cross-target marker variants select the same CUDA wheel;
  all runtime/development dependencies have matching hashed Windows wheels.
- CPU and MPS doctor and pilot-model checks pass. Pilot count remains 17,308,032;
  both report finite gradients, zero causal-prefix difference, and zero optimizer steps.
- The previous Mac/Linux dependency graphs were compared to the exact published
  lockfile, not just to the intended configuration.

Structured evidence, reported machine details, and the pinned Windows wheel
identity are in [windows-correction.json](windows-correction.json). The standard
Windows CPython wheel is `torch-2.14.0+cu130-cp314-cp314-win_amd64.whl`, SHA-256
`78ab64d12e478c8baedc4d90e662f6ffca2b6cb8a872a02b734a8aa3e00277eb`.
This hash is pinned from the official index; the Windows binary was not executed
on this Mac. Resolver/wheel checks are not native installation or CUDA evidence.

The separate review checked the generated lock diff for unexpected upgrades,
platform leakage, source changes, missing wheels, and unpinned replacements.
It also checked the Windows test assumptions and kept all existing assertions
and numerical thresholds. The Windows retest uses an explicit CUDA device, so
CPU success cannot silently satisfy the external GPU gate.

## External retest and pause

[SETUP.md](../../docs/SETUP.md#windows-validation-powershell) gives the PowerShell
retest sequence using only the committed lock, followed by explicit CUDA doctor
and debug/pilot model checks. The driver-reported CUDA 13.4 does not require a
13.4 wheel; CUDA 13.0 is the pinned runtime. NVIDIA's documented backward/minor
compatibility supports the driver selection, but real execution is still pending.

Stop for approval before publishing this correction. After any approved push,
verify the new commit and pause for the owner's Windows RTX 4070 SUPER and native
Linux results. Do not begin Phase 4 without both passing and explicit authorization.
