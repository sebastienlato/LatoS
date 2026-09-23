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

## 2026-09-16 — Dense transformer baseline

Implement the dense decoder directly in PyTorch using pre-RMSNorm blocks,
interleaved rotary Q/K positions, causal multi-head SDPA, SwiGLU, and tied
input/output embeddings. Omit biases and dropout for this initial numerical
baseline. Use normal initialization with smaller residual-output variance.
Independent equations and gradients are the acceptance reference, not another
implementation's output or a pretrained checkpoint.

Use a 631,104-parameter debug model and a 17,308,032-parameter pilot, both with the
accepted 8,192-entry tokenizer hash. Contexts are 128 and 512. These dimensions
are validated construction choices; training feasibility and quality still require
later measurements. Keep float32 as the supported compute baseline, with CPU
float64 solely for reference checks. Model-only snapshots use Safetensors 0.8.0,
explicit configuration, hashes, strict shapes, and no optimizer/resume state.

Define unshifted labels with internal next-token shifting and explicit -100 target
masking. Reject empty-target loss rather than returning NaN. Start with unpadded
causal sequences; do not imply a padding mask or cross-document attention policy.
Use a single-sequence sampler with explicit limits and no KV cache. Transfer logits
to CPU before float64 probability calculations: the combined MPS transfer/cast
failed on the actual host and is covered by a regression test.

## 2026-09-19 — Training and recovery baseline

Use isolated paragraph windows with BOS/EOS, one-token overlap, and right padding.
The model already shifts targets; only padding and the initial label are masked.
Train with token-weighted accumulation so short windows and partial epoch batches
have their correct contribution. Retain every epoch-tail window, and allow an
update's accumulation to cross epochs. Validate every held-out target once.

Use PyTorch AdamW with matrix-only weight decay, one global norm clip per update,
and a pure update-indexed warmup/cosine schedule. Keep float32 and deterministic
CPU shuffle state; the model has no stochastic layers. No new dependencies.

Store complete model/optimizer/sampler state as Safetensors plus checked JSON.
Require matching implementation/runtime/data/config identities to resume. Do not
silently change the schedule to shorten a test; a separate stop boundary preserves
its original horizon. Retain immutable checkpoints and write segments to new
output directories. Establish bitwise recovery on the same local CPU runtime,
with a weaker numerical smoke check on MPS; do not infer CUDA support from Phase 3.

## Phase 5: one measured MPS pilot before scaling

Retain the accepted 17.3M architecture, float32 engine, tokenizer, and source split.
The measured MPS probe supports this size on the available 64 GiB Mac. Train one
fixed 3,000-update run with 256-token paragraph windows; use the predetermined
final checkpoint and reserve test text. Score full validation against both the
seeded random model and an add-one train-only unigram model. Preserve every fixed
sample and all periodic checkpoints. No hyperparameter sweep or larger-model run
is justified by this small, narrow corpus in Phase 5. Accelerator memory readings
are boundary snapshots, not continuous peak measurements. See the
[protocol](experiments/phase-5/PLAN.md) and [report](experiments/phase-5/REPORT.md).

## Phase 6: bounded original SFT with a retained base

Use 192 deterministic synthetic training conversations and 32 held-out validation
conversations, with an independent test reservation. Split related word groups
before rendering; shared templates deliberately test within-template new-word
generalization. Keep the existing vocabulary, separately encode role headers and
content, and reject overlong conversations. Supervise assistant content/EOS only,
using the existing model's single next-token shift. Train a fresh optimizer from
the fixed Phase 5 update 3,000 weights, preserving every prior artifact.

Select final update 200 in advance. Report exact completion and language-modeling
regression alongside assistant loss. The observed failure to generalize (0/32) and
English regression mean the tuned checkpoint is an experiment artifact, not a
replacement base or useful assistant. Do not expand the run or pick a better
validation checkpoint after seeing this outcome. Record the reviewed repetition
and retain the original run, including its different evidence metadata.

## Phase 7 — bounded local terminal inference (2026-09-20)

Use a streaming terminal as the local interface, with explicit artifact selection
and no server/dependency addition. Keep the exact Phase 6 formatter and reject a
whole new turn if history plus reply budget will not fit. Save only nonempty
EOS-completed responses; reset or cancellation cannot silently alter prior turns.
Prefill a new session-owned cache for every reply, keeping cache state out of model
snapshots and training. Validate explicit offset masks and rotary positions against
the existing full-prefix forward. Measure same-backend differences and latency;
do not infer model quality or CUDA behavior from infrastructure checks.

## Phase 8 — release scope, 2026-09-21

Package 1.0.0 marks the reproducible educational workflow, not useful assistant
quality or universal platform support. Retain dependency versions and runtime
implementation. Provide source, wheel and a separate tiny fixture artifact under
MIT with pinned file identities. Keep acquired books and full Mac learned artifacts
local; document their provenance and reproduction requirements without declaring
worldwide redistribution clearance. Do not replace the fixed negative SFT result
with the fixture model. Old optimizer recovery keeps its original version/source
requirements. Publication is a separate owner-approved GitHub checkpoint.

## Phase 9 — independent attention LoRA and controlled full-tuning comparison

Use fixed fused QKV/output targets, Gaussian A/zero B and alpha/rank scaling. Freeze
all existing weights and optimize only A/B using the same token-weighted trainer.
Bind adapters to the full canonical base tensor/config identity, including tokenizer;
merge into a new dense CPU snapshot, leaving source artifacts unchanged. Reuse the
shared chat format and existing train/validation inputs, never the reserved tests.

The first bounded experiment fixes rank 8, alpha 16 and 200 updates before training.
Both methods receive the same base, target exposures, AdamW schedule and seed in
separate processes. This controls exposure, not hyperparameter optimality or wall
time. No search is justified by this mechanics milestone. Adapter optimizer resume
is deferred explicitly; snapshots are for fresh-optimizer loading or inference.
No new dependency or service is needed. Phase 9 proposes a source-only main update,
with no tag, release or learned artifact upload, and stops at owner approval.

## Phase 11 — bound execution and distinguish mechanics from learned behavior

Use two pure local tools and a strict JSON protocol with four assistant turns and
two executions maximum. Keep all weights/tokenizer/chat serialization unchanged;
use ordinary user messages for host-supplied tool results under chat contract 1.
No arbitrary code, external calls, model training, dependency upgrades or new paid
services. Reject overflow and incomplete generation without executing a pending call.

Measure parsing, envelope and argument schemas, correct call sequences, successful
execution, final answers and expected failure handling separately. Gold scripts
are mechanics controls, never model performance. Fixed base/SFT/DPO development
validation remains negative; all responses and context-limit failures are retained.
A future training experiment would require its own fixed data/compute hypothesis.
Phase 11 proposes main only, no tag/release/assets, and stops for explicit approval.

## 2026-09-23 — Repository positioning and bounded Roadmap 2.0

Present LatoS as an original small language model project built from first principles.
English remains the primary language of current data/evaluations, not a public
identity qualifier. Preserve historical release wording and every experimental
result. The README serves new visitors; detailed evidence, platform scope and
publication governance belong in linked guides, reports and project state.

Retire open-ended research extensions after the completed Phase 12. Phase 13 is
repositioning/planning only. Phases 14–21 progress through evaluation, Data 2.0,
measured dense CUDA scaling, serious base pretraining, instruction quality,
interoperability, justified advanced post-training and a finite LatoS 2.0 release.
Freeze numeric quality gates before improvement runs; failed quality gates block
dependent phases. No architecture, training behavior, weights or learned experiment
changes in Phase 13. CLI/package description edits affect wording only; historical
source fingerprints remain tied to their original checkouts.

Mac remains authoritative for development; the existing Windows RTX 4070 SUPER
(~12 GB VRAM) may perform planned substantive training as well as validation.
Candidate scale follows measurements, not a borrowed architecture or fixed parameter
count. No new paid services. Keep old synthetic CUDA checks distinct from full
learned-model experiments. Phase 14 requires approved, verified Phase 13 publication
and explicit owner/master-planning start authorization. See [Roadmap 2.0](ROADMAP.md).

## 2026-09-23 — Fixed Roadmap 2.0 evaluation before improvement

Keep language modeling, generation, instructions, external tasks and resources
separate. Preserve the legacy validation objective and add lossless 192-byte text
spans with fixed boundaries for cross-tokenizer BPB. Freeze 96 development instruction
cases, 24 generation prompts and 48 reserved acceptance cases. Use ARC-Easy and
ARC-Challenge validation with pinned inputs and answer-only likelihood, plus a
byte-normalized length-bias diagnostic; no free-generation MC approximation.

Freeze numeric improvement and regression gates before retrospective scoring.
Both historical reserved tests remain unused in Phase 14. Final access needs a
locked passing candidate, contamination review and a retained selection declaration;
failed final quality blocks dependent phases. No tuning follows final feedback
without an approved fresh acceptance design. See [evaluation](docs/EVALUATION.md).

PyArrow is an optional evaluation extra only. No architecture, production tokenizer,
training behavior, paid API or historical evidence changes. Five preserved dense
artifacts are retrospectively measured without promotion; merged LoRA remains
distinct from live-adapter arithmetic. Phase 15 needs verified Phase 14 publication
and separate explicit owner/master-planning authorization.
