# Project state

Updated: 2026-09-23.

## Active checkpoint

**Phase 11 is formally closed locally within its documented bounded scope; the
reviewed closure commit awaits explicit publication approval. Do not begin Phase 12
in this chat.** The owner-supplied independent Windows/CUDA PASS satisfies the
external-validation wait. After approved closure publication, verify exact remote
main and all unchanged tags, then stop. Phase 12 needs an explicit fresh-chat start
only after that verification; the [handoff](docs/PHASE12_HANDOFF.md) is preparation.

Phase 11 implementation is published at `41fc3cffe42a1ae7a42446c5785abfffc1f559db`,
package 1.3.0, **without a Phase 11 tag**. Verified publication supersedes historical
pending snapshots. All eleven tags remain fixed, including v1.0.0 at
`10d9ef7bf0618b364f887ff9d279408e3cbc33c9`, object
`1acb6b94f0adfbca85014ad6601dbfc716e3c651`.

Delivered closure: [report](experiments/phase-11/CLOSURE.md),
[structured evidence](experiments/phase-11/closure.json),
[separate review](experiments/phase-11/CLOSURE_REVIEW.md). Original Phase 11
report/results/samples/plan/inventories/scripts and both attempts remain unchanged.

## Scoped evidence and preserved limits

- **Owner-reported Windows RTX 4070 SUPER PASS:** 293 passed / 12 expected skips /
  zero failures, fresh exact checkout/locked install and fresh non-editable wheel.
  Phase 11: 50 passes / one MPS-only skip in each environment. Parsing, schemas,
  bounds, failures, separate metrics, CLI/packaging/privacy and scoped side-effect
  audit pass. Scripted controls: 12/12 normal tasks and 4/4 expected failures handled.
- Windows retained **tiny** base/SFT/DPO, 256 capacity; base is a random fixture.
  Original system prompt plus cases correctly rejected with context_limit before
  generation. **One fixed compact prompt** enabled 1,024 CUDA tokens/model across
  16 cases. Other controls unchanged per owner. Each model: **0/16 JSON**, no calls,
  **0/12 tasks and 0/4 learned failure handling**. Not the full Mac prompting experiment;
  full Mac artifacts/tokenizer unavailable. Exact compact text/hash, raw Windows logs,
  artifact hashes, full skip breakdown and numerical maxima not supplied.
- **Separately inspected hosted Linux CPU** [run 35862021715](https://github.com/sebastienlato/LatoS/actions/runs/35862021715):
  295 passed / ten MPS-only skips, exact implementation; Phase 11 50 CPU passes / one
  skip. Protocol/scripted/tiny CPU inference and existing workflows/builds. Verifier
  fixtures labeled base/SFT/DPO are scripts, not real model evaluation. No full Mac
  experiment, Windows compact-prompt exercise, CUDA/MPS or fresh installed-wheel suite.
  Dense fixture recovery is not tool learning or adapter/DPO resume. Physical Linux deferred.
- **Original Mac unchanged:** base/SFT/DPO each 0/16 JSON, no calls, 0/12 tasks and
  0/4 learned failure handling; 595/48/51 generated tokens. Twelve base retry-context
  failures plus four incomplete replies; SFT/DPO sixteen retry-context failures each.
  No second learned turn. Scripted 12/12 and 4/4 are mechanics only. Development and
  isolated wheel each passed 305 tests. Learned tool use remains **unestablished**.
- **59,999 local files rehashed unchanged**, including the prior 59,873 snapshot and
  126 additional distinct Phase 11 run/inventory/log/original evidence files. Both
  reserved tests opaque-hashed only, never parsed/evaluated. Windows preservation is
  separately owner-reported: 213 tracked and 2,543 prior regular report/evidence files,
  clean tree, no Windows commit/push. Preserve all models, tokenizer and chat contract 1.
- [Phase 10 closure](experiments/phase-10/CLOSURE.md) retains full Mac negative DPO:
  100 updates / 400 pairs / 2,498 targets, ranking 16/32 → 15/32, exact 0/32 → 0/32;
  English 5.653422 → 5.603780 remains worse than original base 4.731898. Distinct tiny
  Windows four-update/16-pair/169-target ranking 16/32 → 16/32 and separately scoped
  hosted Linux evidence remain unchanged. Negative SFT/LoRA/control results retained.
- Merge atol=rtol=1e-4 remains amended; original CPU 1e-5 failure retained. Cache
  bounds CPU/CUDA 1e-5, MPS 1e-4. Adapter/DPO optimizer resume unsupported; dense
  recovery version-bound. Windows OS-level Ctrl-C remains unvalidated. No useful
  instruction/tool following, factual reliability, safety alignment, general device
  determinism, cross-device equality, production performance, mixed precision,
  distributed serving or learned language quality beyond 256 is established.

## Pending publication and next action

- Commit message: `Close Phase 11 with scoped external validation`.
- Existing remote: `https://github.com/sebastienlato/LatoS.git`, **main**, public unchanged.
- Documentation/evidence only; runtime/version/tests/configs/lock/workflow/original
  evidence unchanged. No new runtime suite, training, benchmark or build for closure.
  JSON, links, preservation, source identity, complete diff and privacy checks passed.
- **Stop for explicit Phase 11 closure publication approval.** Exact commit appears
  in the approval request and `.private/phase11-closure-publication.json` once prepared.
- No tag, release, assets or Phase 12 work. After approved publication verify exact
  main and all unchanged tags, then stop. Closure approval does not override this gate.

Local uv: `.private/tools/bin/uv`; runtime `.venv`. Ignored learned artifacts and
raw logs remain local, not remote backups. See the handoff for preserved inputs.
