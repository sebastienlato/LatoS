# Project state

Updated: 2026-09-21.

## Active checkpoint

**Phase 9 is formally closed locally within its documented scope; the reviewed
closure commit awaits explicit publication approval. Phase 10 has not begun and
must not begin in this chat.** The independent Windows/CUDA wait is satisfied.
After an approved closure push, verify exact remote main and unchanged tags, then
stop. Phase 10 requires an explicit fresh-chat start after verified closure.

Phase 9 implementation is published at `0fbedd31f746d30bb5f071fbda30defa9036a451`,
package 1.1.0, **without a Phase 9 tag**. Its verified publication supersedes the
old tracked pending snapshot. All eleven existing tags remain fixed, including
annotated v1.0.0 at `10d9ef7bf0618b364f887ff9d279408e3cbc33c9`, tag object
`1acb6b94f0adfbca85014ad6601dbfc716e3c651`.

Delivered closure: [attributed report](experiments/phase-9/CLOSURE.md),
[structured evidence](experiments/phase-9/closure.json),
[separate review](experiments/phase-9/CLOSURE_REVIEW.md), and
[Phase 10 handoff](docs/PHASE10_HANDOFF.md). Original Phase 9 report, results,
samples, verification, failure evidence, inventories and implementation are unchanged.

## Evidence and preserved limits

- Mac CPU/MPS implementation evidence: **236 passed** in development and installed
  wheel; full learned base/adapter/control comparison and merge/cache checks through
  512. Both methods: 200 updates / 6,188 targets. Base/LoRA/full exact replies remain
  **0/32**; English loss 4.731898 / 4.773844 / 5.653425. No useful assistant.
- Owner-reported Windows RTX 4070 SUPER PASS at exact implementation: **226 passed,
  10 expected skips**, zero failures; fresh exact checkout/locked install and fresh
  non-editable wheel suite. Tiny CUDA LoRA/full: **4 updates / 175 targets each**,
  **7,856 frozen base / 192 trainable updated adapter parameters**, all 0/32 exact.
  Tiny fraction ~**2.44399%**; independently checked **0.851951279%** applies to the
  full pilot architecture only. Not full Mac learned-experiment reproduction;
  full learned 512-position merge on Windows was not validated. Raw logs/artifact
  hashes were not supplied here. Windows console Ctrl-C remains unvalidated.
- Separately inspected hosted Linux CPU [run 35617504810](https://github.com/sebastienlato/LatoS/actions/runs/35617504810):
  **228 passed, 8 MPS-only skips** at exact implementation. Tiny CPU LoRA/comparison,
  CLI and offline workflows/builds; no fresh installed-wheel suite, full learned
  verification or CUDA/MPS. **Physical Linux remains deferred.**
- Merge **atol=rtol=1e-4** remains the amended contract; original CPU 1e-5 failure
  is retained, not reclassified. CPU/CUDA cache is separately 1e-5; MPS cache 1e-4.
  Adapter optimizer resume is unsupported. Float64 diagnostics are not mixed precision.
- Original Phase 6 SFT remains negative: final update 200 / 6,188 targets, assistant
  loss 8.354879 → 5.815233, exact 0/32 → 0/32, English 4.731898 → 5.653422.
  Preserve that checkpoint separately; do not promote SFT or the new adaptations.
- **743 distinct local files rehashed unchanged**; reserved tests integrity-hashed
  only, never parsed/evaluated. Windows preservation is separately owner-reported:
  1,513 prior evidence/report files, 173 tracked files, clean tree, no Windows push.
  Keep all base/SFT/LoRA/control models, tokenizer, chat contract and prior attempts.
- Useful quality, LoRA superiority, general accelerator determinism, cross-device
  equality, production performance, mixed precision/distributed serving and language
  quality beyond 256-token windows remain unestablished. No new paid service.

## Pending publication and next action

- Reviewed commit message: `Close Phase 9 with scoped external validation`.
- Existing remote: `https://github.com/sebastienlato/LatoS.git`, **main**, public.
- Documentation/evidence only; package/runtime/tests/configs/lock/workflow and
  original experiment evidence unchanged. No new runtime tests/training/builds
  for closure; JSON, links, preservation, diff and privacy checks passed.
- **Stop for explicit Phase 9 closure publication approval.** Exact commit is in
  the approval request and `.private/phase9-closure-publication.json` once prepared.
- No tag, release, asset upload, visibility change or Phase 10 development. Preserve
  all eleven tags. After approved publication and verification, stop and provide
  the prepared fresh-chat prompt; closure approval does not lift the current gate.

Local uv: `.private/tools/bin/uv`; runtime `.venv`. Ignored learned artifacts are
local inputs, not remote backups. See the handoff for preservation and future scope.
