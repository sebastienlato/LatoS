# Project state

Updated: 2026-09-20.

## Phase and checkpoint

**Phase 7 is complete and separately reviewed locally. Publication awaits explicit
owner approval; stop at the reviewed push checkpoint.**
The owner started Phase 7 in this fresh Work chat after verification of published
Phase 6 closure `c4ed8c9564c41f26be2c10ee178f5a10ac80d3a8`. Remote main matched;
annotated v0.7.0 remains at `f6636af34933b9678824cc8dd50f6ab6559a2de5`, with tag
object `bdbaa5499847ae7080e2b32bc96c5400ed14df80` unchanged.

Phase 7 supplies streaming terminal chat, JSON events, explicit model selection,
Unicode completion, cancellation, whole-turn context rejection and a session-owned
KV cache. The exact shared Phase 6 chat formatter is unchanged. Partial, empty and
cancelled replies never enter history. The reviewed cancellation fix covers the
last token and final Unicode flush. See [inference guide](docs/INFERENCE.md),
[validation plan](experiments/phase-7/PLAN.md) and [review](experiments/phase-7/REVIEW.md).

## Validation

- **210 tests passed** in the locked development environment and **210 passed** in
  a fresh non-editable wheel installation. Lint, formatting, source/wheel builds,
  CPU/MPS diagnostics and real installed chat commands passed.
- Both preserved models passed same-backend cached/uncached logits, incremental and
  chunked prefixes through 512 positions, reset and causal checks. CPU atol/rtol=1e-5;
  MPS atol/rtol=1e-4. Maximum absolute difference was 1.812e-5. Fixed greedy IDs matched.
- Fixed 128-prefix/32-step forward-only decode: base CPU 111.3→712.8 tokens/s,
  base MPS 279.7→353.7; SFT CPU 110.8→727.5, SFT MPS 280.6→361.7. Prefill was slightly
  slower cached. First-token/subsequent latency and memory are recorded separately.
- **357 preservation hashes unchanged**, including both reserved payloads and all
  49 SFT inventory entries. Shared formatter unchanged. Tests were not used for tuning.
- Separate review fixed cancellation at the final token/Unicode flush. Final tests
  and the four retained-model measurement runs passed after those fixes. Initial
  PATH/offline-install harness failures are retained and described in the report.

[Final report](experiments/phase-7/REPORT.md) and [results](experiments/phase-7/results.json)
record exact scope. Full logs, measurements and source snapshots remain under
`outputs/phase-7-validation/`.
No new paid service or remote write has occurred. Dependencies retain their locked
versions; only the package version advances to 0.8.0.

## Prior evidence and limits — unchanged

- Full Mac MPS SFT: **200 updates / 6,188 assistant-target exposures**. Assistant
  loss **8.354879 → 5.815233**, exact replies **0/32 → 0/32**, English loss
  **4.731898 → 5.653422**. Preserve the negative result and final update 200.
- Windows / RTX 4070 SUPER Phase 6: owner-reported PASS at exact v0.7.0, **184 passed,
  three MPS-only skips**, plus fresh wheel suite. Bounded CUDA SFT: **four updates /
  175 assistant-target exposures**. It did not reproduce the full Mac experiment or
  learned artifact. No raw Windows logs were supplied here.
- GitHub-hosted Linux CPU Phase 6: separately inspected PASS for
  [run 35522033012](https://github.com/sebastienlato/LatoS/actions/runs/35522033012),
  **184 passed, three MPS-only skips**, within its tiny CPU SFT/workflow scope.
  No full Mac reproduction, CUDA or fresh wheel-installed suite was claimed for CI.
- Physical Linux remains **deferred, not performed**. No Phase 7 Windows/CUDA or
  Linux execution has run. Prior platform results do not validate new inference.
- Useful instruction following, general CUDA determinism, cross-device equality,
  mixed precision and distributed execution remain unestablished. A terminal and
  faster decoding do not reverse the SFT negative result. 512-token mechanics do
  not establish language quality beyond the prior at-most-256-token windows.

## Preserved inputs

Base: `outputs/phase-5-english-pilot/step-00003000/model/`, weights SHA-256
`f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa`.
Experimental SFT: `outputs/phase-6-instruction-reviewed/step-00000200/model/`, weights
`62b890214e5544292cc54b725a4e505730fca0d501b3337cdd3f051a622d09ae`.
Tokenizer: `artifacts/tokenizers/english-bpe-v1/`, SHA-256
`7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
The completed before/after preservation check covers 357 artifact/data files, including all
49 reviewed SFT inventory entries, earlier runs/failures and both reserved test
payloads. Test payloads are hashed only for preservation, never parsed or used for
inference selection. Ignored local artifacts are not remote backups. Original
optimizer recovery still requires its recorded runtime/implementation.

## Pending publication

- Reviewed commit message: `Complete Phase 7 local streaming inference and KV cache`.
  This state file belongs to that reviewed commit; its exact ID is supplied in the
  approval request and is available from `git log -1 --format=%H`.
- Approval: **PENDING**. No Phase 7 branch backup, push, PR, tag, release or artifact
  upload is authorized or performed.
- Existing private remote: `https://github.com/sebastienlato/LatoS.git`; branch `main`.
  Proposed new annotated tag: **v0.8.0**, pointing to the same reviewed commit.
  Preserve all existing tags, especially v0.7.0. No release or weight/data upload.
- Next action: present the reviewed local commit, validation/limits and exact
  publication details; ask **“Push Phase 7 to GitHub?”** and await approval. No Phase 8
  development begins before Phase 7 publication is approved and verified; honor any
  owner-imposed transition gate supplied with that approval.

Local uv: `.private/tools/bin/uv`; development runtime: `.venv`. No new paid services.
