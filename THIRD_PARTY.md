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

## Phase 2 tokenizer dependency and references

[Hugging Face Tokenizers 0.23.2](https://github.com/huggingface/tokenizers/releases/tag/v0.23.2)
was checked against upstream release documentation and live PyPI metadata on
2026-09-16. The stable macOS arm64 ABI3 wheel installed and executed on Python
3.14.7. This library implements BPE; LatoS does not claim to have invented BPE or
reimplemented its Rust trainer. No pretrained vocabulary or merges were imported.

New installed distributions (including transitive dependencies) are listed below;
all versions/hashes are in `uv.lock`. Their own license notices apply. Hub/HTTP
packages are dependencies of Tokenizers; LatoS's tokenizer path makes no Hub or
hosted-service requests.

| Distribution | Version | Declared license |
| --- | --- | --- |
| [anyio](https://pypi.org/project/anyio/4.15.1/) | 4.15.1 | MIT |
| [certifi](https://pypi.org/project/certifi/2026.7.22/) | 2026.7.22 | MPL-2.0 |
| [click](https://pypi.org/project/click/8.5.0/) | 8.5.0 | BSD-3-Clause |
| [h11](https://pypi.org/project/h11/0.16.0/) | 0.16.0 | MIT |
| [hf-xet](https://pypi.org/project/hf-xet/1.6.0/) | 1.6.0 | Apache-2.0 |
| [httpcore](https://pypi.org/project/httpcore/1.0.9/) | 1.0.9 | BSD-3-Clause |
| [httpx](https://pypi.org/project/httpx/0.28.1/) | 0.28.1 | BSD-3-Clause |
| [huggingface_hub](https://pypi.org/project/huggingface_hub/1.31.0/) | 1.31.0 | Apache-2.0 |
| [idna](https://pypi.org/project/idna/3.19/) | 3.19 | BSD-3-Clause |
| [PyYAML](https://pypi.org/project/PyYAML/6.0.3/) | 6.0.3 | MIT |
| [tokenizers](https://pypi.org/project/tokenizers/0.23.2/) | 0.23.2 | Apache-2.0 (upstream classifier/license) |
| [tqdm](https://pypi.org/project/tqdm/4.70.1/) | 4.70.1 | MPL-2.0 AND MIT |

Primary API documentation consulted:

- [Tokenization pipeline](https://huggingface.co/docs/tokenizers/en/pipeline).
- [ByteLevel pre-tokenizers](https://huggingface.co/docs/tokenizers/en/api/pre-tokenizers).
- [BPE trainers](https://huggingface.co/docs/tokenizers/en/api/trainers).
- [Models](https://huggingface.co/docs/tokenizers/en/api/models).
- [Added tokens](https://huggingface.co/docs/tokenizers/en/api/added-tokens) and
  [Tokenizer API](https://huggingface.co/docs/tokenizers/en/api/tokenizer).

LatoS's own integration enforces train-only fitting and its codec contract;
examples in upstream documentation are not treated as data-splitting policy.

## Phase 3 model methods and storage

LatoS's transformer is original PyTorch code implementing established methods.
The following primary papers supplied the mathematical definitions; their source
implementations, pretrained weights, and other assets were not imported:

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
  2017: scaled attention, causal masking, multi-head composition, and embedding sharing.
- Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467),
  2019: RMS normalization.
- Su et al., [RoFormer](https://arxiv.org/abs/2104.09864), 2021: rotary position equations.
- Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202),
  2020: SwiGLU feed-forward transformation.
- [PyTorch 2.14 SDPA documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html):
  library call semantics, causal flag, and explicit dropout handling.

[Safetensors 0.8.0](https://github.com/safetensors/safetensors/releases/tag/v0.8.0),
Apache-2.0, was selected from official release/PyPI metadata and installed/tested
on the actual Python 3.14.7 environment. Its
[PyTorch API](https://huggingface.co/docs/safetensors/en/api/torch) provides
tensor-only storage. It adds no new transitive dependencies to the existing runtime.
The resolved wheel identities are in `uv.lock`; upstream license notices apply.
Sources checked on 2026-09-16.

## Phase 3 Windows correction — 2026-09-18

The Windows route uses the official
[PyTorch CUDA 13.0 index](https://download.pytorch.org/whl/cu130/torch/), selecting
2.14.0+cu130 for CPython 3.14 Windows x86-64. The same upstream PyTorch licensing
and its bundled-component notices apply; no CUDA wheel is redistributed here.
Windows resolution adds [Colorama 0.4.6](https://pypi.org/project/colorama/0.4.6/),
BSD-3-Clause, as a Windows-only transitive dependency of pytest/tqdm. Existing
non-Windows dependency versions and wheel hashes are retained.

Primary guidance: [uv PyTorch sources](https://docs.astral.sh/uv/guides/integration/pytorch/),
[uv environment constraints](https://docs.astral.sh/uv/reference/settings/#environments),
and [NVIDIA driver/runtime compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
Index resolution is verified; real Windows/CUDA execution is still pending.

## Phase 4 optimization and reproducibility

No new dependency or external code was added. LatoS's batching, schedule, training
loop, validation, and checkpoint orchestration are original implementations using
these established algorithms and library primitives:

- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101),
  ICLR 2019: AdamW's decoupled weight decay.
- [PyTorch 2.14 AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html):
  optimizer semantics, state, betas/epsilon, and explicit foreach/fused choices.
- [PyTorch gradient norm clipping](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html):
  global norm clipping and nonfinite-gradient rejection.
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.14/notes/randomness.html):
  version/platform limits and explicit generator control.

Consulted on 2026-09-19. Tensor storage continues to use the existing Safetensors
dependency. The offline fixture and its learned tokenizer remain LatoS-generated
artifacts with the fixture's existing provenance; no pretrained model is used.
