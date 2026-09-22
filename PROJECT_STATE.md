# Project state

Updated: 2026-09-22.

## Active checkpoint

**Phase 10 is formally closed locally within its documented scope; the reviewed
closure commit awaits explicit publication approval. Do not begin Phase 11 in this
chat.** The owner-supplied independent Windows/CUDA PASS satisfies the external
validation wait. After approved closure publication, verify exact remote main and
unchanged tags, then stop. Phase 11 needs an explicit fresh-chat start after verification.

Phase 10 implementation is published at `abed1adffba562373348477e3958bcd1d5886b11`,
package 1.2.0, **without a Phase 10 tag**. Verified publication supersedes the original
tracked pending snapshot. All eleven existing tags remain fixed; annotated v1.0.0
at `10d9ef7bf0618b364f887ff9d279408e3cbc33c9`, object
`1acb6b94f0adfbca85014ad6601dbfc716e3c651`.

Delivered closure: [report](experiments/phase-10/CLOSURE.md),
[structured evidence](experiments/phase-10/closure.json),
[separate review](experiments/phase-10/CLOSURE_REVIEW.md), and
[Phase 11 handoff](docs/PHASE11_HANDOFF.md). Original Phase 10 report, results,
samples, verification, inventories, first attempt and checker failure are unchanged.

## Evidence and preserved limits

- Full Mac DPO remains **negative**: 100 updates / 400 pairs / 2,498 final-response
  targets. Preference loss 0.693147 → 0.718657; ranking **16/32 → 15/32**; exact replies
  **0/32 → 0/32**. English 5.653422 → 5.603780 improves versus SFT but stays worse than
  original base 4.731898. No preference-learning or instruction-following improvement.
  Mac development and isolated wheel: 254 passed; full learned evidence retained.
- Owner-reported Windows RTX 4070 SUPER PASS at exact implementation: **243 passed /
  11 expected skips / zero failures**, fresh exact checkout/locked install and fresh
  non-editable wheel suite. Tiny CUDA **4 updates / 16 pairs / 169 response targets**;
  frozen CUDA reference, actual updated policy, exact same-CUDA reload and validation
  preservation. Ranking **16/32 → 16/32**, exact **0/32 → 0/32**. Not full Mac 100-update
  reproduction; full learned SFT/reference and English corpus not reproduced. Raw logs,
  hashes, skip breakdown and numerical maxima not supplied. DPO resume unsupported.
- Separately inspected hosted Linux CPU [run 35742827336](https://github.com/sebastienlato/LatoS/actions/runs/35742827336):
  **245 passed / 9 MPS-only skips**, exact implementation. Tiny CPU DPO two-update/
  eight-pair integration and scalar/exposure oracle, CLI, offline workflows/builds.
  No fresh installed-wheel suite, full learned experiment/oracle or CUDA/MPS.
  Dense fixture recovery is not DPO/adapter resume. **Physical Linux deferred.**
- **54,356 distinct local files rehashed unchanged**, including the prior 54,236-file
  snapshot, 109 Phase 10 inventory entries and 11 original evidence/scripts. Both
  reserved tests integrity-hashed only, never parsed/evaluated. Windows preservation
  is separately owner-reported: 2,017 prior evidence/report files, 194 tracked files,
  clean tree, no Windows commit/push. Preserve all models, tokenizer and chat contract 1.
- Original negative SFT (200 updates / 6,188 targets, exact 0/32, English regression)
  and Phase 9 base/LoRA/full results (all exact 0/32, English
  4.731898 / 4.773844 / 5.653425) remain unchanged. See prior closure for historical
  tiny Windows/CUDA versus hosted Linux scope; no full Mac experiment substitution.
- Merge **atol=rtol=1e-4** remains amended; original CPU 1e-5 failure retained.
  Cache bounds stay CPU/CUDA 1e-5, MPS 1e-4. Adapter and DPO optimizer resume remain
  unsupported; original dense recovery is version-bound. Scalar/float64 diagnostics
  are not mixed precision. Windows console Ctrl-C remains unvalidated.
- Useful quality, DPO/LoRA superiority, safety alignment, factual reliability,
  general accelerator determinism, cross-device equality, production performance,
  mixed precision/distributed serving and language quality beyond 256 remain
  unestablished. Familiar synthetic templates, shared validation prompts and
  length-sensitive scoring limit conclusions. No new paid service.

## Pending publication and next action

- Commit message: `Close Phase 10 with scoped external validation`.
- Existing remote: `https://github.com/sebastienlato/LatoS.git`, **main**, public unchanged.
- Documentation/evidence only; package/runtime/tests/configs/lock/workflow/original
  evidence unchanged. No new runtime tests, training, benchmarks or builds for closure.
  JSON, links, preservation, source identity, complete diff and privacy checks passed.
- **Stop for explicit Phase 10 closure publication approval.** Exact commit is in the
  approval request and `.private/phase10-closure-publication.json` once prepared.
- No tag, release, asset upload, visibility change or Phase 11 development. After
  approved publication and verification, stop and provide the prepared fresh-chat
  prompt; closure approval does not override the current gate.

Local uv: `.private/tools/bin/uv`; runtime `.venv`. Ignored learned artifacts are
local inputs/outputs, not remote backups. See the handoff for preserved inputs.
