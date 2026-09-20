# Phase 6 fresh-chat handoff

Preparation only. **Do not begin Phase 6 in the Phase 5 closure chat.** First publish
and verify the exact owner-approved closure commit, then require an explicit owner
start in a fresh Work chat. Closure push approval alone does not authorize a start.

## Starting point and evidence boundaries

Read `AGENTS.md`, `PROJECT_STATE.md`, `ROADMAP.md`, this handoff, and the local private
context/blueprint when present. Private planning stays outside tracked files and
publication metadata. Verify the actual closure commit from its approval and
publication record, not merely the implementation tag.

Validated Phase 5 implementation: `v0.6.0`,
`fa0c3ac3b7f3890ffdcad411968a656da2f74b3b`; keep that tag unchanged.
Phase 5 closure is documentation-only and keeps package version 0.6.0.
See [closure evidence](../experiments/phase-5/CLOSURE.md).

- Mac MPS: full 17,308,032-parameter English pilot, 3,000 updates and 5,761,229
  target exposures; validation loss 9.089003 → 4.731898. Final update 3,000 was
  selected in advance, despite slightly better validation at 2,500. Samples are
  repetitive and incoherent; useful assistant behavior is not established.
- Owner-reported independent Windows / RTX 4070 SUPER: PASS at exact v0.6.0;
  162 passed, two MPS-only skips, zero failures; full suite also passed in a fresh
  non-editable wheel installation. The unchanged runner completed four CUDA
  updates, 416 target exposures, generation, recovery and 20 inventory hash checks.
  This was **not reproduction of the full Mac pilot** or its scores.
- Directly verified GitHub-hosted Linux CPU [run 35515094307](https://github.com/sebastienlato/LatoS/actions/runs/35515094307):
  162 passed, two MPS-only skips; tiny CPU Phase 5 runner tests, CPU fixture
  acceptance, locked setup, lint/format, diagnostics, offline workflows and builds.
  No full-pilot reproduction, CUDA, or fresh wheel-installed suite is claimed for CI.
- Independent physical Linux remains **deferred, not performed**. No general CUDA
  determinism, cross-device numerical equivalence, sustained-pilot performance,
  language-quality, instruction-following or Phase 6 capability follows from the
  bounded Windows exercise or CPU CI.

## Preserved local inputs

Do not overwrite existing artifacts. A fresh clone does not include these ignored
files, and their existence on a different machine must be checked honestly.

- Selected base model: `outputs/phase-5-english-pilot/step-00003000/model/`;
  `model.safetensors` SHA-256:
  `f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa`.
- The enclosing checkpoint retains optimizer/shuffle state; initial and update
  1,000/2,000 checkpoints, the exact pilot runner, plan, logs and samples also stay
  under `outputs/phase-5-english-pilot/`. Check its inventory against
  `experiments/phase-5/artifacts.json`; inventory SHA-256:
  `99621c23dfaf0353da37769ee9f5eecd78f16de42f6ac4b61b6bace558a32637`.
- Tokenizer: `artifacts/tokenizers/english-bpe-v1/`, 8,192 entries; SHA-256
  `7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
- `data/processed/english-books-v1/`: retain train/validation/test separation.
  English test text remains reserved; never use it for routine SFT selection.
- `checkpoints/phase-3-pilot-initial/`: preserve the random baseline, weight SHA-256
  `99dce7bf0356bf7d641dff4b29d6616772dfbf9779877e0c085bdea9e368b5f2`.
- Preserve `outputs/phase-4-*` and `outputs/phase-5-*` evidence and logs.

Read [PRETRAINING.md](PRETRAINING.md) and [TRAINING.md](TRAINING.md). Optimizer
recovery requires matching runtime/backend/implementation/data/configuration.
A new SFT objective or dataset is not a pretraining resume. Preserve the base
model as a read-only comparison, initialize new SFT work from its model weights,
and record the new optimizer/training contract explicitly. Do not silently switch
to an intermediate pilot checkpoint or incompatible tokenizer.

Local uv is `.private/tools/bin/uv`; Python lives under `.private/python`, with the
accepted locked `.venv`. Recheck actual host memory, disk and backends. The prior
Mac has an M4 Max and 64 GiB, but do not assume the next host matches it. External
CUDA validation is not access to that machine. New paid-service budget is zero.

## Phase 6 objective, only after the fresh-chat start

Implement and measure a bounded instruction-tuning experiment from the preserved
base, following the Phase 6 roadmap. Own routine engineering decisions, execution,
validation, separate review/fixes and the local commit.

- Independently select or author English conversations, document source rights,
  provenance and any synthetic generation, and split by conversation/source before
  formatting. Reserve held-out instruction checks from optimization and retain a
  separate test reservation. Do not acquire or generate this data during closure.
- Document one shared chat serialization and its role/turn boundaries. Define
  tokenizer/special-token handling and keep training/evaluation formatting aligned.
- Implement and verify assistant-only target masking with the model's next-token
  shift; cover padding, non-assistant turns, EOS, multi-turn boundaries, truncation
  and examples with no usable assistant targets. Respect actual context limits:
  model capacity 512, previous pilot training/evaluation windows 256.
- Measure available resources, choose a runnable budget, and record seed, data,
  base/tokenizer identities, settings, checkpoint selection, token exposure,
  runtime, throughput, memory, fixed samples, failures and recovery evidence.
- Compare base and SFT on the same held-out instruction cases and scoring contract,
  plus relevant language-modeling regressions using validation only. Distinguish
  assistant-only loss from prior all-target pretraining loss. Preserve negative
  results; a successful training loop is not proof of useful instruction following.
- Keep all acquired data, weights, full logs and private context outside Git.
  Preserve the accepted base. Do not expand into Phase 7 inference work.
- After implementation, tests, separate review/fixes, evidence and local commit,
  stop at the reviewed Phase 6 push-approval checkpoint. Publication needs its own
  approval; Phase 5 approvals do not cover it.

## Fresh-chat prompt — use only after closure publication is verified

> Start LatoS Phase 6 in this fresh Work chat only after verifying the exact
> published Phase 5 closure commit against its approval/publication record and
> confirming v0.6.0 remains at fa0c3ac3b7f3890ffdcad411968a656da2f74b3b. Read
> AGENTS.md, PROJECT_STATE.md, ROADMAP.md, docs/PHASE6_HANDOFF.md, and local private
> context/blueprint when present. Windows/RTX 4070 SUPER validation passed as a
> bounded four-update, 416-target CUDA exercise; it did not reproduce the full Mac
> English pilot. Hosted Linux CPU CI passed separately within its documented scope;
> physical Linux remains deferred. No general CUDA determinism or cross-device
> equivalence is established. Preserve the selected Phase 5 base and tokenizer,
> all prior artifacts and privacy rules, and reserved test data. Implement a
> measured English instruction-tuning experiment with documented conversations,
> shared chat formatting, verified assistant-only masking/next-token alignment,
> held-out base-versus-SFT comparisons and regression checks. Use available
> resources without new paid services, complete validation and separate review,
> then stop at the reviewed Phase 6 push-approval checkpoint.
