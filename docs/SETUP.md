# Setup and validation

## Supported checkpoint environment

The recorded local environment is CPython 3.14.7 on Apple Silicon macOS. Linux
x86-64 CPU dependencies are locked and Phase 0 passed its published CI job.
Phase 1 data CI awaits publication. Other operating systems, architectures, Python series, CUDA,
and mixed precision are untested. See [ENVIRONMENT.md](ENVIRONMENT.md).

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
check an available Apple GPU. `--device cuda` requires a separately validated
CUDA installation, which the Phase 0 Linux CPU environment does not provide.
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
data manifests, the original tiny fixture, and compact experiment reports.
