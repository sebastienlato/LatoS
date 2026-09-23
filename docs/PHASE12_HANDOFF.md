# Phase 12 fresh-chat handoff

Preparation only. **Do not begin Phase 12 in the Phase 11 closure chat.** Obtain
explicit approval for the reviewed closure commit, publish that exact state, verify
remote main and all eleven unchanged tags, then stop. Phase 12 requires an explicit
fresh-chat start after verification. This document does not select or start an experiment.

## Starting state

Phase 11 implementation is published at `41fc3cffe42a1ae7a42446c5785abfffc1f559db`,
package 1.3.0, **without a Phase 11 tag**. The separate closure commit is identified
in the owner's publication confirmation and `.private/phase11-closure-publication.json`
when present. Verify that exact approved closure against remote main. Verified
publication records supersede historical pending snapshots; do not mistake the
implementation commit for closure. Preserve all eleven tag objects and peeled
targets, including v1.0.0 at `10d9ef7bf0618b364f887ff9d279408e3cbc33c9`, object
`1acb6b94f0adfbca85014ad6601dbfc716e3c651`.

Read AGENTS.md, PROJECT_STATE.md, ROADMAP.md, Phase 11 [closure](../experiments/phase-11/CLOSURE.md),
[structured evidence](../experiments/phase-11/closure.json), [report](../experiments/phase-11/REPORT.md),
and local private context/blueprint/publication records when present.

## Preserved evidence and boundaries

- Original Mac base/SFT/DPO and English tokenizer, original prompt: each 0/16 valid
  JSON, no proposed/executed calls, 0/12 tasks and 0/4 learned failure handling.
  Tokens 595/48/51; retry-context limits and incomplete replies retained. Mac suites:
  305 passed in development and isolated wheel. Scripted 12/12 and 4/4 are mechanics.
- Owner-reported Windows RTX 4070 SUPER PASS: 293 passed / 12 expected skips, fresh
  wheel; Phase 11 50 passes / one MPS skip. Retained tiny models, random fixture base,
  256 capacity; original prompt rejected before generation. One fixed compact prompt
  enabled 1,024 CUDA tokens/model across 16 cases. Each still 0/16 JSON, no calls,
  0/12 tasks and 0/4 learned failure handling. Other controls/scoring unchanged per
  owner. Full Mac artifacts/tokenizer unavailable. **Not the full Mac prompting
  experiment**; raw logs/hashes/exact compact prompt not supplied in this session.
- Separately inspected hosted Linux CPU run 35862021715: 295 passed / ten MPS-only
  skips; Phase 11 50 CPU passes / one skip. Protocol/scripted/tiny CPU mechanics and
  workflows/builds, not full-model evaluation, CUDA or fresh installed-wheel suite.
  Verifier fixtures labeled base/SFT/DPO contain scripts, not learned outputs.
  Physical Linux remains deferred. Windows OS-level Ctrl-C remains unvalidated.
- Full Mac DPO remains negative (ranking 16/32 → 15/32, exact 0/32; English better
  than SFT but worse than original base), distinct from tiny Windows/CUDA Phase 10
  and scoped hosted Linux evidence. Negative SFT/LoRA/control results remain unchanged.
- Rehash before reuse; preserve all base/SFT/LoRA/merged/control/DPO and random models,
  original tokenizer, chat contract 1, prior evidence and both reserved tests.
  The [model card](MODEL_CARD.md), Phase 9/10 inventories, Phase 11 [plan](../experiments/phase-11/plan.json)
  and [inventory](../experiments/phase-11/artifacts.json) identify inputs. Ignored local
  artifacts are not backed up by a clone; Phase 11 inventory paths are relative to
  `outputs/phase-11-tools-reviewed`. Reserved tests stay out of routine evaluation.
- Merge atol=rtol=1e-4 and original CPU 1e-5 failure are retained; cache bounds remain
  CPU/CUDA 1e-5, MPS 1e-4. Adapter/DPO optimizer resume unsupported; dense recovery
  version-bound. No learned tool use, useful assistant quality, factual reliability,
  safety alignment, general accelerator determinism, cross-device equality,
  production performance, mixed precision, distributed serving or learned language
  quality beyond 256 is established.

## Future scope, only after explicit fresh-chat start

Roadmap Phase 12+ permits one justified research extension at a time, with a fixed
baseline, declared hypothesis, existing-resource compute budget and reproducible
positive or negative result. No extension, method, dataset, baseline, configuration
or training has been selected or started here. The future task should choose based
on preserved evidence, use zero new paid services, implement and validate within a
bounded protocol, perform a separate review/fixes, and prepare the local commit
before requesting its own publication approval.

## Concise kickoff — use only after closure publication and verification

Use the exact closure commit from the publication confirmation; never substitute
the Phase 11 implementation commit.

> Start LatoS Phase 12 in this fresh chat only after verifying remote main matches
> the exact approved Phase 11 closure commit from the publication confirmation and
> all eleven tags remain unchanged, including v1.0.0 at
> 10d9ef7bf0618b364f887ff9d279408e3cbc33c9. Read AGENTS.md, PROJECT_STATE.md,
> ROADMAP.md, docs/PHASE12_HANDOFF.md, Phase 11 closure/evidence and local private
> context/blueprint/publication records. Verified publication supersedes pending
> snapshots. Select one justified, bounded research extension with a fixed baseline,
> hypothesis and existing-resource budget; use no new paid services. Preserve all
> models, tokenizer, chat format, prior evidence, both reserved tests and capability
> limits, including the distinct negative Mac and compact-prompt tiny Windows tool
> results and separately scoped hosted Linux evidence. Implement, validate, separately
> review/fix, prepare a local commit and concrete publication proposal, then stop
> for explicit approval.
