# Setup and validation

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

Local CPU/MPS regression checks pass. The original Phase 3 Linux CPU CI passed;
native external Linux validation remains pending. Windows resolution was checked
on Mac; the owner subsequently confirmed native installation and GPU detection.
Real CUDA tensor/model execution still awaits the retest. These are different
evidence levels. Other architectures, Python series, and mixed precision remain untested.
See [the correction evidence](../experiments/phase-3/WINDOWS_CORRECTION.md).

The subsequent v0.4.1 Windows retest successfully installed PyTorch 2.14.0+cu130
and detected the RTX 4070 SUPER, but stopped at a fixture setup error after nine
test passes. Git's `core.autocrlf=true` had converted the byte-pinned LF fixtures
to CRLF. Version 0.4.2 supplies fixture-specific checkout attributes; see
[the checkout correction evidence](../experiments/phase-3/CHECKOUT_CORRECTION.md).
CUDA tensor/model execution remains unvalidated because the external run stopped
at that first test error.

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
must remain recorded when later upgrades are introduced.

Build products remain in ignored `dist/`. Inspect source and wheel member lists
before sharing; never archive the entire working directory. Package builds use
explicit source inclusion rules to exclude local planning, environments, acquired
corpora, and checkpoints. Source archives retain project docs, tests, the lockfile,
data manifests, the original tiny fixture, configurations, and compact experiment
reports. Learned tokenizer files remain in ignored `artifacts/`; model tensor
snapshots remain in ignored `checkpoints/`.

## Windows validation (PowerShell)

Use the exact approved correction commit once it is published; record
`git rev-parse HEAD` before testing. No source or lock edits, unlocked upgrades,
manual pip replacement of PyTorch, index overrides, or `--no-sources` are needed.
The existing v0.4.0 tag continues to identify the original blocked checkpoint.

For the 0.4.2 checkout correction, use a **fresh checkout of the exact approved
commit** after publication. Existing working-tree files are not necessarily
rewritten when a new `.gitattributes` is pulled. There is no need to change system
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
`$env:RAYON_NUM_THREADS = "1"` before its `uv run` command; the README's inline
environment assignment is POSIX shell syntax. Stop and report any failure rather
than changing the validated environment on a validation-only machine. Both
external platform results and explicit owner authorization are required before
Phase 4 can begin.
