# Phase 7 fresh-chat handoff

Preparation only. **Do not begin Phase 7 in the Phase 6 closure chat.** First obtain
approval for the reviewed closure commit, publish that exact state and verify it.
Then require explicit owner start in a fresh Work chat. Closure publication
approval does not override this gate.

## Starting point and evidence boundaries

Read AGENTS.md, PROJECT_STATE.md, ROADMAP.md, this handoff, and local private
context/blueprint when present. Private planning stays outside tracked files and
publication metadata. Verify the actual closure commit against its approval and
publication record; the implementation tag alone is not proof of closure.

Phase 6 implementation is published as annotated `v0.7.0` at
`f6636af34933b9678824cc8dd50f6ab6559a2de5`; retain that tag unchanged. Its tag object
is `bdbaa5499847ae7080e2b32bc96c5400ed14df80`. Closure is documentation-only and
keeps package 0.7.0. See [closure evidence](../experiments/phase-6/CLOSURE.md).

- Full Mac MPS SFT experiment: 200 updates / 6,188 assistant-target exposures.
  Assistant validation loss 8.354879 → 5.815233, but **0/32 exact replies** for both
  base and SFT, and English loss 4.731898 → 5.653422. All samples and the negative
  result remain preserved; the tuned checkpoint is not an improved base.
- Owner-reported independent Windows / RTX 4070 SUPER PASS at exact v0.7.0:
  **184 passed, three MPS-only skips, zero failures**, plus fresh non-editable wheel
  suite. Actual bounded CUDA SFT: **four updates / 175 assistant-target exposures**;
  49 artifact files verified and 64 responses independently replayed. Both tiny
  models scored 0/32. This did not reproduce the full Mac experiment or artifact.
- Directly inspected GitHub-hosted Linux CPU
  [run 35522033012](https://github.com/sebastienlato/LatoS/actions/runs/35522033012):
  **184 passed, three MPS-only skips**, tiny two-update CPU SFT runner/verifier
  tests, Phase 4 fixture acceptance, locked setup, lint/format, diagnostics,
  offline workflows and builds. No full SFT run, CUDA, fresh wheel-installed test
  suite or independent-process Phase 6 verifier CLI result is claimed for CI.
- Physical Linux remains **deferred, not performed**. No useful instruction
  following, general CUDA determinism, cross-device numerical equality,
  mixed-precision/distributed behavior or broader capability is established.

## Preserved local inputs

A fresh clone does not contain ignored artifacts. Verify actual availability and
hashes; do not overwrite, silently substitute, or refit missing inputs.

- Phase 5 accepted base: `outputs/phase-5-english-pilot/step-00003000/model/`;
  weights SHA-256 `f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa`.
- Phase 6 final experimental SFT: `outputs/phase-6-instruction-reviewed/step-00000200/model/`;
  weights SHA-256 `62b890214e5544292cc54b725a4e505730fca0d501b3337cdd3f051a622d09ae`.
  Clearly label its measured limits in any inference experience. Preserve the
  accepted base separately; do not treat SFT as a quality upgrade.
- Phase 6 inventory: [artifacts.json](../experiments/phase-6/artifacts.json), SHA-256
  `f0d88d3de0ab3da03d7c9f8734aea3de24f7191afc9ea76499d5c7bfccda01c7`. The run retains
  exact package source/lock, initial/update 100/update 200 recovery checkpoints,
  samples and metrics. Earlier calibration, failure and pre-review pilot stay.
- Tokenizer: `artifacts/tokenizers/english-bpe-v1/`, 8,192 entries, SHA-256
  `7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
- English corpus: `data/processed/english-books-v1/`; instructions:
  `data/processed/english-instructions-v1/`. Both test payloads remain reserved.
  No routine inference optimization or quality selection may consume them.
- Preserve the random baseline under `checkpoints/phase-3-pilot-initial/`, all
  `outputs/phase-4-*`, `outputs/phase-5-*` and `outputs/phase-6-*` artifacts.
- Shared chat format: `src/latos/chat.py`, `CHAT_CONTRACT` version 1. Keep exact
  segmented header/content encoding, BOS/EOS handling and generation-prefix
  identity with Phase 6. Read [INSTRUCTION_TUNING.md](INSTRUCTION_TUNING.md),
  [MODEL.md](MODEL.md) and [TRAINING.md](TRAINING.md) before implementation.

Read-only model loading is distinct from optimizer recovery. Earlier checkpoints
require their original runtime/implementation fingerprints for training recovery.
Do not rewrite old artifacts to fit new inference code. Recheck actual host/backends,
memory and disk; prior host was an M4 Max with 64 GiB, not a hardware guarantee.
Local uv is `.private/tools/bin/uv`, with the locked `.venv`. New paid-service budget
is zero. External validation is not access to the Windows GPU.

## Phase 7 objective — only after the fresh-chat start

Implement the roadmap's inference phase: CLI chat, streaming local interface and
KV cache. Own routine implementation, validation, separate review/fixes and local
commit preparation. Adding an interface or faster inference cannot establish
language quality or reverse the Phase 6 negative result.

- Preserve model-only checkpoint compatibility and the exact shared chat formatter.
  Make the chosen artifact explicit and label the SFT model experimental.
- Validate cached/uncached logits and generation under declared numerical tolerance
  on actually available backends, including incremental/chunked prefixes, positions,
  causal attention and cache reset between conversations. Record actual results.
- Verify EOS stopping, token budgets, context limits and multi-turn behavior. Define
  an explicit policy for history that does not fit; never silently corrupt or split
  a message. Model capacity is 512, but prior experiments used at most 256-token
  windows; no 512-token language-quality claim follows.
- Keep streaming text consistent with completed decoding, including Unicode and
  cancellation/end conditions. Keep the interface local and avoid external service
  dependency or new charges. Test the actual interface path and honest error cases.
- Measure prefill/first-token latency, subsequent-token latency/throughput and memory
  for fixed prompts/settings with appropriate synchronization; compare cached and
  uncached inference without retraining or selecting on test text.
- Preserve artifacts and provenance, record platform limits and any failures,
  complete separate review/fixes, prepare the exact local commit, and stop at the
  Phase 7 push-approval checkpoint. Phase 6 approval never authorizes Phase 7 pushes.

## Concise fresh-chat prompt — use only after verified closure publication

> Start LatoS Phase 7 in this fresh Work chat after verifying the published Phase 6
> closure commit against its approval/publication record and confirming v0.7.0
> remains at f6636af34933b9678824cc8dd50f6ab6559a2de5. Read AGENTS.md,
> PROJECT_STATE.md, ROADMAP.md, docs/PHASE7_HANDOFF.md, and local private
> context/blueprint when present. Preserve the base, experimental SFT artifact,
> tokenizer, shared chat format, prior evidence and both reserved test sets.
> Windows/CUDA PASS covered four updates / 175 assistant targets, not the full Mac
> SFT experiment; Linux CPU CI passed separately within its documented scope and
> physical Linux remains deferred. Useful instruction following, general CUDA
> determinism and cross-device equality are not established. Implement CLI chat,
> streaming local inference and KV caching with cached/uncached agreement, stopping,
> context-limit and latency checks. Use existing resources without new paid services,
> complete validation and separate review/fixes, prepare the local commit, then stop
> at the reviewed Phase 7 push-approval checkpoint.
