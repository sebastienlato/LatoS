# Setup and validation

## Current scope

The pinned environment below remains in use on current main (package 1.3.0).
For the published stable release, follow [v1.0.0 reproduction](RELEASE.md).
Platform counts and the PowerShell retest below describe their named historical
checkpoints; they are not a current-main certification. See the
[evidence index](INDEX.md) for later results and [project state](../PROJECT_STATE.md)
for the active phase. Mac Work remains authoritative for development. Roadmap 2.0
explicitly permits substantive training on Windows/RTX 4070 SUPER (~12 GB VRAM)
in its CUDA training phases. Historical validation-only restrictions below apply
to those old retests. Physical Linux remains deferred.

## Supported checkpoint environment

Use CPython 3.14.7 and uv 0.12.15. The lock has these platform-specific PyTorch builds:

| Environment | Locked PyTorch | Source |
| --- | --- | --- |
| macOS ARM64, macOS 14+ wheel target | 2.14.0, CPU/MPS | PyPI |
| Linux x86-64 | 2.14.0+cpu | Official PyTorch CPU index |
| Windows x86-64 | 2.14.0+cu130 | Official PyTorch CUDA 13.0 index |

The Windows route is new in the Phase 3 correction, version 0.4.1. It selects the
CUDA build even when a GPU is not detected at install time. Both native Python's
`AMD64` spelling and uv's cross-target `x86_64` spelling are admitted; Windows ARM
and 32-bit Python are not. The two custom indexes are explicit and used only for
PyTorch. Other packages continue to resolve from PyPI. macOS/Linux versions,
sources, and existing wheel hashes are unchanged.

Phase 3 is closed locally at validated implementation v0.4.2,
`1102b64714b58c1f2289f80863fa032cc192c47b`. Local Mac CPU/MPS checks passed.
The owner reports a complete independent Windows/RTX 4070 SUPER CUDA PASS:
131 tests passed, one MPS-only skip, and real CUDA tensor and debug/pilot model
execution. GitHub-hosted Linux CPU CI passed separately at the same commit with
131 passes and one MPS-only skip.

Independent physical Linux testing is explicitly deferred by the owner because
the machine is temporarily unavailable. It was not performed; CI does not replace
that evidence. Other architectures, Python series, and mixed precision remain
untested. See [closure evidence](../experiments/phase-3/CLOSURE.md).

Earlier Windows blockers are preserved in the
[environment correction](../experiments/phase-3/WINDOWS_CORRECTION.md) and
[checkout correction](../experiments/phase-3/CHECKOUT_CORRECTION.md) reports.
The v0.4.2 PASS supersedes those failures for this exact implementation. Closure
documentation is published at `294d2e0af2712266b398151eb7e335fa7ccdc31e`. The owner
authorized Phase 4 in a fresh chat; see [training and recovery](TRAINING.md).

Phase 4 v0.5.0 has since passed independent Windows/RTX 4070 SUPER validation,
reported by the owner: 159 tests passed, two MPS-only skips, actual CUDA training
and separate-process recovery. Completed GitHub Linux CPU CI was verified
separately with 159 passes/two skips and offline training acceptance. Exact CUDA
equality in that validation is not a general determinism guarantee. Physical
Linux remains deferred. See [Phase 4 closure](../experiments/phase-4/CLOSURE.md).

Phase 7 v0.8.0 (`04031e5ea98da8db495242165a78c216ab1d4cf4`) has also passed
owner-reported Windows/RTX 4070 SUPER validation: 202 passed, eight expected skips,
fresh non-editable wheel suite and actual CUDA cache/inference/CLI checks. These
used tiny 256-position base/SFT artifacts plus a separate synthetic 512-position
model; the full Mac learned artifacts were unavailable and not validated on Windows.
Windows OS-level Ctrl-C/console-event delivery remains unvalidated; callback
cancellation passed. The POSIX process-SIGINT test is intentionally skipped there.

GitHub Linux CPU CI passed separately at v0.8.0: 203 passed, seven MPS-only skips,
tiny CPU inference/CLI/POSIX SIGINT tests and existing offline workflows/builds.
Building a wheel in CI is not evidence of a fresh-wheel-installed test suite.
Physical Linux remains deferred. See [Phase 7 closure](../experiments/phase-7/CLOSURE.md)
and [inference usage](INFERENCE.md) for exact scope and limitations.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) version
0.12.15. To keep the bootstrap entirely within the checkout when uv is absent:

```sh
python3 -m venv .private/tools
.private/tools/bin/python -m pip install uv==0.12.15
export PATH="$PWD/.private/tools/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$PWD/.private/python"
uv python install --no-bin 3.14.7
uv sync --locked
```

This uses ignored directories for bootstrap tools, the Python runtime, and `.venv`.
It does not replace system Python. uv may also use its normal download cache.
With uv already on PATH, `uv sync --locked` reads `.python-version`, obtains the
runtime if needed, and creates `.venv`. Always work from the repository root.

## Checks

```sh
uv run --locked python -c 'import latos; print(latos.__version__)'
uv run --locked latos --help
uv run --locked python -m latos --version
uv run --locked latos doctor --device cpu --json
uv run --locked latos doctor --device auto
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv pip check
uv build --no-build-isolation
```

The default doctor tests one selected backend. Use `--device mps` to explicitly
check an available Apple GPU. Windows installs the pinned CUDA-enabled wheel;
Linux intentionally retains the CPU-only contract. Use `--device cuda` for an
explicit CUDA check on the Windows validation machine.
The command never installs packages or enables a fallback. JSON reports include
availability, selected test device, float32 precision, and `status`; failures return
exit code 1. Invalid command-line arguments return 2. A successful auto check does
not mean every backend was tested.

The report includes physical memory when the OS API exposes it and free disk
space on the current working directory's filesystem. These are host observations,
not promises about memory available for training. No usernames, absolute project
paths, or hostnames are intentionally collected. Review errors before sharing
reports because upstream error text may contain local details.

If PyTorch cannot load, help/version still work and doctor returns an error.
Use the recorded platform and a fresh `uv sync --locked`; do not substitute a
nightly build to conceal a compatibility problem. MPS fallback configuration is
reported; the recorded smoke checks use no fallback. Hardware smoke checks cover
only a tiny matrix multiplication and backward pass, not all future model kernels.

## Reproducibility and artifacts

Use `--locked` to reject stale project metadata rather than silently changing the
lock. Keep `uv.lock` committed. `.python-version` selects the patch version, and
`pyproject.toml` pins the required uv version. Accepted experiment environments
must remain recorded when later upgrades are introduced. Even documentation-only
package source edits change implementation fingerprints: use the original checkout
and runtime for historical optimizer recovery, and never rewrite old inventories
to make a current checkout appear to have produced an earlier run.

Build products remain in ignored `dist/`. Inspect source and wheel member lists
before sharing; never archive the entire working directory. Package builds use
explicit source inclusion rules to exclude local planning, environments, acquired
corpora, and checkpoints. Source archives retain project docs, tests, the lockfile,
data manifests, the original tiny fixture, configurations, and compact experiment
reports. Learned tokenizer files remain in ignored `artifacts/`; model tensor
snapshots remain in ignored `checkpoints/`.

## Windows validation (PowerShell)

For reproduction, use the validated v0.4.2 implementation above and record
`git rev-parse HEAD` before testing. No source or lock edits, unlocked upgrades,
manual pip replacement of PyTorch, index overrides, or `--no-sources` are needed.
The existing v0.4.0 tag continues to identify the original blocked checkpoint.

Use a **fresh checkout of the exact validated commit**. Existing working-tree files
are not necessarily rewritten when a new `.gitattributes` is pulled. There is no need to change system
or global Git configuration or edit fixture files on the validation machine.
Before syncing, these read-only commands should show `i/lf`, `w/lf`, and `eol: lf`
for the three fixture payloads and their manifest:

```powershell
git rev-parse HEAD
git ls-files --eol -- data/fixtures/tiny/*.txt data/fixtures/tiny/manifest.json
git check-attr text eol -- data/fixtures/tiny/train.txt data/fixtures/tiny/validation.txt data/fixtures/tiny/test.txt data/fixtures/tiny/manifest.json
```

The repository rules apply only to fixture text/JSON files. They preserve the
existing committed bytes under `core.autocrlf=true`, `input`, or `false`, without
altering checksum verification. The regression suite requires Git on PATH and
uses isolated temporary repositories; it does not modify user Git settings.

```powershell
uv --version
uv sync --locked
uv run --locked python -c "import platform, torch; print(platform.python_version(), platform.machine()); print(torch.__version__, torch.version.cuda); assert torch.__version__ == '2.14.0+cu130'; assert torch.version.cuda == '13.0'; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv pip check
uv run --locked latos doctor --device cuda --json
uv run --locked latos model check --config configs/model/debug.json --device cuda
uv run --locked latos model check --config configs/model/pilot.json --device cuda
```

The CUDA doctor/model commands must explicitly pass on the NVIDIA GPU; passing
CPU tests alone does not validate CUDA. The MPS-specific test should skip on
Windows. Record command results and failures as well as the package, GPU, and driver
versions. Model checks use synthetic token IDs and require no corpus, tokenizer,
or checkpoint download. No training or Phase 4 work is part of this validation.

The owner's reported RTX **4070 SUPER**, driver **616.92**, and driver-reported
CUDA **13.4** replace the earlier planned RTX 4080 description. The driver's CUDA
number describes driver capability, not the runtime inside the PyTorch wheel.
The selected wheel uses CUDA 13.0; newer compatible drivers can run it under
[NVIDIA's compatibility policy](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
The reported driver is above that policy's CUDA 13.x minimum of 580. This supports
the selection; it is not a substitute for real execution. A separately installed
CUDA 13.4 toolkit is not required to use this prebuilt wheel.

For the existing offline tokenizer example in PowerShell, set
`$env:RAYON_NUM_THREADS = "1"` before its `uv run` command. Inline environment
assignments shown in historical guides use POSIX shell syntax. Stop and report any failure rather
than changing the validated environment on a validation-only machine. Windows
validation is complete; independent physical Linux testing is deferred at the
owner's direction. These commands describe the completed Phase 3 retest. Phase 4
was subsequently authorized in a fresh Work chat after closure publication was
verified; its distinct local evidence is in [the Phase 4 report](../experiments/phase-4/REPORT.md).
