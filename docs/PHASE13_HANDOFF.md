> **Superseded historical handoff — do not execute this research kickoff.**
> The owner redefined Phase 13 as Repository Repositioning & Roadmap 2.0 on
> 2026-09-23. The old open-ended extension below is retained only as history.
> Follow [Roadmap 2.0](../ROADMAP.md) and [current state](../PROJECT_STATE.md).
> Phase 14 requires verified Phase 13 publication and explicit owner/master-planning
> authorization; Phase 13 push approval alone does not authorize it.

# Next research extension — fresh-chat handoff (Phase 13)

Preparation only. **Do not begin another Phase 12+ experiment in the closure chat.**
First approve and publish the reviewed Phase 12 closure commit, verify exact remote
main and all eleven unchanged tags, then stop. Use this handoff only in an explicitly
started fresh chat after publication verification. No next experiment is selected.

## Starting checkpoint and required context

Phase 12 correction is published at `b222c1fa844289247531b922d5cb18134e99c7b1`, package
1.3.0; there is no Phase 12 tag. This is not the separate closure commit. Obtain the
exact closure commit from the owner's publication confirmation or local
`.private/phase12-closure-publication.json`; verify that exact state on remote main.
Verified publication supersedes historical tracked pending snapshots.

Read AGENTS.md, PROJECT_STATE.md, ROADMAP.md, this handoff, Phase 12
[closure](../experiments/phase-12/CLOSURE.md), [structured evidence](../experiments/phase-12/closure.json),
[original report](../experiments/phase-12/REPORT.md),
[portability correction](../experiments/phase-12/PORTABILITY_CORRECTION.md),
and local private context/blueprint/publication records. Preserve all eleven tag
objects and peeled targets; v1.0.0 remains at
`10d9ef7bf0618b364f887ff9d279408e3cbc33c9`, object
`1acb6b94f0adfbca85014ad6601dbfc716e3c651`.

## Evidence and boundaries to preserve

- Original Mac pinned learned base/SFT/DPO and English tokenizer: **44 learned retry
  blocks removed** at 512, with no successful learned tool use. Parsing remains
  0/41, 0/64, 0/64 emitted replies; each model 0/12 tasks and 0/4 failure handling.
  Both runs and the original hypothesis/result remain fixed; H1 passes, H2 fails.
- Corrected owner-reported Windows RTX 4070 SUPER PASS: checkout and fresh wheel
  each **301 passed / 13 skips**. Mac learned artifacts/tokenizer unavailable, not
  substituted. The original random synthetic 512-capacity fixture has **16 initial
  blocks removed**, zero replies at 256 → one at 512, 1,024 CUDA tokens, 0/16 JSON,
  no calls, 0/12 tasks and 0/4 learned failure handling. **Not Mac-style learned
  retries.** Keep synthetic sessions and scripted retry/tool controls separate.
- Windows CUDA reaches 512 positions, rejects 513/over-budget inputs without
  truncation, passes reservation boundaries and cache atol=rtol=1e-5 with approximate
  maximum absolute difference 1.78814e-7. This is bounded engineering evidence,
  not general CUDA determinism or cross-device equality. Earlier Windows path
  failure/evidence remains preserved; no fresh experiment arose from that correction.
- Separately inspected hosted Linux CPU run 35880051473: **303 passed / 11 MPS skips**,
  Phase 12 eight CPU passes/one skip; synthetic inference, inventory/scripted/cache
  mechanics and workflows/builds. Not the full Mac experiment, Windows CUDA exercise,
  fresh installed-wheel suite or full artifact/guard audit. Physical Linux deferred.
- Rehash before reuse; preserve all models, adapters, merged/control/random models,
  tokenizer/chat contract 1, prior evidence, failure records and both reserved tests.
  Original inputs are identified in Phase 12 [plan](../experiments/phase-12/plan.json)
  and [inventory](../experiments/phase-12/artifacts.json). Inventory paths are relative
  to `outputs/phase-12-context-reviewed`. A Git clone does not restore ignored models.
  Both reserved tests remain excluded from routine tuning/evaluation.
- Historical evidence retains source fingerprints: use its original checkout or
  captured source for full historical replay. Do not rewrite inventories or bypass
  source-identity checks to make changed code appear to have generated earlier runs.
- Prior negative SFT/LoRA/control/DPO and distinct Phase 11 Mac/compact-prompt tiny
  Windows/Linux scopes stay intact. Merge 1e-4 and original CPU 1e-5 failure retained;
  cache CPU/CUDA 1e-5, MPS 1e-4; adapter/DPO resume unsupported, dense recovery
  version-bound, Windows OS Ctrl-C unvalidated. No learned tool use, useful assistant,
  learned long-context quality, safety alignment or production performance established.

## Future scope

Only after an explicit fresh-chat start, select one justified bounded research
extension with a fixed baseline, hypothesis and existing-resource budget. No new
paid services. No method, data, baseline, configuration or training is selected here.
Implement, validate, separately review/fix, preserve evidence and prepare a local
commit before the next publication approval. Never merge the distinct evidence
scopes into a learned-capability claim.

## Concise kickoff — only after closure publication and verification

> Start LatoS's next bounded research extension (Phase 13) in this fresh chat only
> after verifying remote main matches the exact approved Phase 12 closure commit
> from its publication confirmation and all eleven tags remain unchanged, including
> v1.0.0 at 10d9ef7bf0618b364f887ff9d279408e3cbc33c9. Read AGENTS.md, PROJECT_STATE.md,
> ROADMAP.md, docs/PHASE13_HANDOFF.md, Phase 12 closure/evidence and local private
> context/blueprint/publication records. Verified publication supersedes pending
> snapshots. Select one justified extension with a fixed baseline, hypothesis and
> existing-resource budget; no new paid services. Preserve all models, tokenizer,
> chat format, prior evidence, both reserved tests and capability limits. Keep Mac's
> 44 learned retry transitions, Windows's 16 synthetic initial blocks removed,
> scripted controls and hosted Linux evidence separate; no learned tool capability
> follows. Physical Linux remains deferred. Implement, validate, separately review/fix,
> prepare a local commit and publication proposal, then stop for explicit approval.
