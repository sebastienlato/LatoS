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

## 2026-09-16 — Bounded English data

Begin with twelve hash-pinned historical English books and an independent tiny
fixture. Reserve eight books for training, two for validation, and two for test;
all author groups are disjoint. Fixed document assignments precede paragraph
extraction. No source aliases may cross split boundaries. A small, transparent
corpus lets us prove acquisition and integrity before considering larger data.
It does not support a claim of broad modern English competence.

Use Python's standard library for the pipeline, with a 20 MB raw manifest budget
and at most 50,000 candidate paragraphs. Keep original bytes/notices outside Git.
Pin exact bytes, fail on changed sources, and publish only compact metadata and
reports. Source wrappers/front matter are excluded using checked boundaries.

Compare complete word five-shingle sets at Jaccard >=0.80; index every shingle.
This trades memory for a testable lexical guarantee at the current scale. Preserve
held-out examples before training duplicates. Do not silently alter thresholds or
splits based on future model performance. Record normalization/filter policy and
the pipeline implementation hash with every output. Other language or larger
corpus support needs its own measured implementation.

## 2026-09-16 — Tokenizer contract

Use stable Tokenizers 0.23.2 to train a new BPE vocabulary with all 256 byte
symbols and four reserved IDs: pad 0, BOS 1, EOS 2, unknown 3. Use an 8,192-entry
baseline, minimum frequency 2, and a 32-symbol token length limit; the offline
fixture targets 512 entries. These are bounded starting choices, not optimized
model-quality settings. Never fit on held-out splits.

Choose identity normalization with no inserted leading space. Preserve arbitrary
valid Unicode scalar text, including exact whitespace and composed/decomposed
forms. Phase 1's prior cleaning is a separate transformation. Add BOS/EOS only
through explicit API flags, and encode literal reserved spellings as ordinary
text. Restore the upstream runtime flag after every load because JSON does not
save it. Bare upstream JSON loading is not the complete LatoS codec contract.

Record tokenizer, vocabulary, merge, corpus, and training-input hashes. Keep learned
files outside Git and publish compact evidence. Verify the full training corpus
through encode/decode and compare every encoding after save/load. Report frozen
validation compression without fitting on it; reserve test text. Do not infer
language-model quality from codec coverage or compression.
