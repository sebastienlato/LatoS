# Third-party provenance

Original LatoS implementation is MIT licensed. Dependencies are installed from
their upstream distributions; their notices and licenses remain applicable. No
external model, tokenizer, dataset, or implementation source is included.

## Installed Python dependencies

Snapshot: 2026-09-16. Versions and license descriptions below come from installed
distribution metadata. The lockfile records sources, versions, and artifact hashes.
Composite expressions include bundled components; inspect each distribution's
license files before redistributing dependencies. This repository does not bundle
their source or wheels.

| Distribution | Version | Declared license |
| --- | --- | --- |
| [filelock](https://pypi.org/project/filelock/3.32.7/) | 3.32.7 | MIT |
| [fsspec](https://pypi.org/project/fsspec/2026.7.0/) | 2026.7.0 | BSD-3-Clause |
| [hatchling](https://pypi.org/project/hatchling/1.32.0/) | 1.32.0 | MIT |
| [iniconfig](https://pypi.org/project/iniconfig/2.3.0/) | 2.3.0 | MIT |
| [Jinja2](https://pypi.org/project/Jinja2/3.1.6/) | 3.1.6 | BSD License |
| [MarkupSafe](https://pypi.org/project/MarkupSafe/3.0.3/) | 3.0.3 | BSD-3-Clause |
| [mpmath](https://pypi.org/project/mpmath/1.3.0/) | 1.3.0 | BSD |
| [networkx](https://pypi.org/project/networkx/3.6.1/) | 3.6.1 | BSD-3-Clause |
| [numpy](https://pypi.org/project/numpy/2.5.3/) | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| [packaging](https://pypi.org/project/packaging/26.3/) | 26.3 | Apache-2.0 OR BSD-2-Clause |
| [pathspec](https://pypi.org/project/pathspec/1.1.1/) | 1.1.1 | Mozilla Public License 2.0 (MPL 2.0) |
| [pluggy](https://pypi.org/project/pluggy/1.6.0/) | 1.6.0 | MIT |
| [Pygments](https://pypi.org/project/Pygments/2.21.0/) | 2.21.0 | BSD-2-Clause |
| [pytest](https://pypi.org/project/pytest/9.1.1/) | 9.1.1 | MIT |
| [ruff](https://pypi.org/project/ruff/0.16.8/) | 0.16.8 | MIT |
| [setuptools](https://pypi.org/project/setuptools/84.0.0/) | 84.0.0 | MIT |
| [sympy](https://pypi.org/project/sympy/1.14.0/) | 1.14.0 | BSD |
| [tomlkit](https://pypi.org/project/tomlkit/0.15.1/) | 0.15.1 | MIT |
| [torch](https://pypi.org/project/torch/2.14.0/) | 2.14.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| [trove-classifiers](https://pypi.org/project/trove-classifiers/2026.6.1.19/) | 2026.6.1.19 | Apache Software License |
| [typing_extensions](https://pypi.org/project/typing_extensions/4.16.0/) | 4.16.0 | PSF-2.0 |

Linux resolves the separate official `torch==2.14.0+cpu` wheel. It has not
executed on this host. Its upstream licensing remains part of the PyTorch
distribution; platform wheels may bundle different components.

## Runtime and automation tools

- [CPython](https://www.python.org/downloads/release/python-3147/), 3.14.7: PSF license and bundled component notices. The local uv-managed runtime is supplied through [python-build-standalone](https://github.com/astral-sh/python-build-standalone).
- [uv](https://github.com/astral-sh/uv/releases/tag/0.12.15), 0.12.15: MIT or Apache-2.0; project environment and locking.
- [actions/checkout](https://github.com/actions/checkout/releases/tag/v7.0.1), v7.0.1: MIT; CI checkout.
- [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv/releases/tag/v10.1.0), v10.1.0: MIT; CI tool installation.

The workflow pins action commit identities. Release tags and metadata were checked
with read-only GitHub API requests on 2026-09-16. No action implementation is vendored.

## Primary documentation consulted for Phase 0

- [Python supported versions](https://devguide.python.org/versions/) and [downloads](https://www.python.org/downloads/): stable runtime selection.
- [PyTorch 2.14.0 release](https://github.com/pytorch/pytorch/releases/tag/v2.14.0) and [package metadata](https://pypi.org/pypi/torch/2.14.0/json): stable release and platform wheel availability.
- [PyTorch installation](https://pytorch.org/get-started/locally/) and [MPS backend](https://docs.pytorch.org/docs/2.14/notes/mps.html): runtime installation and backend detection.
- [uv projects](https://docs.astral.sh/uv/concepts/projects/), [PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/), and [GitHub Actions integration](https://docs.astral.sh/uv/guides/integration/github/): locking, indexes, and CI.

Consulted on 2026-09-16. Phase 0 did not implement a model or research algorithm.

## Phase 1 data and duplicate detection

- [Project Gutenberg](https://www.gutenberg.org/): twelve independently selected
  English texts; catalog attribution, original credits, revision, terms, source
  URLs, and SHA-256 identities are recorded in
  [the source manifest](data/manifests/english-books-v1.json).
- [Gutenberg license](https://www.gutenberg.org/policy/license.html),
  [permission guidance](https://www.gutenberg.org/policy/permission.html),
  [robot access guidance](https://www.gutenberg.org/policy/robot_access.html), and
  [official mirror list](https://www.gutenberg.org/MIRRORS.ALL): acquisition and
  original-file terms. The acquired books are not bundled in Git or packages.
- [Canadian copyright guidance](https://ised-isde.canada.ca/site/canadian-intellectual-property-office/en/guide-copyright):
  jurisdiction-specific selection context; see [DATA.md](docs/DATA.md).
- Manning, Raghavan, and Schütze, *Introduction to Information Retrieval* (2008),
  [near-duplicates and shingling](https://nlp.stanford.edu/IR-book/html/htmledition/near-duplicates-and-shingling-1.html):
  established shingle-set/Jaccard method. LatoS's implementation is original and
  compares exact sets using an inverted index, not the book's randomized sketch.

Checked on 2026-09-16. The tiny fixture is original AI-assisted writing under MIT;
its provenance is in [the fixture README](data/fixtures/tiny/README.md). No new
third-party Python dependency was added in Phase 1.
