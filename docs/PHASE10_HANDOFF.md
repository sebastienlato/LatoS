# Phase 10 fresh-chat handoff

Preparation only. **Phase 10 has not begun and must not begin in the Phase 9 closure
chat.** First obtain explicit closure publication approval, publish that exact
commit, verify remote main and unchanged tags, then stop. Closure approval alone
does not authorize Phase 10 development in the current chat.

## Starting point

Phase 9 implementation is published at
`0fbedd31f746d30bb5f071fbda30defa9036a451`, package 1.1.0, **without a Phase 9 tag**.
Its separate closure commit is identified by the owner's approval and the local
`.private/phase9-closure-publication.json` when present. Verify that exact closure
against remote main; do not mistake the implementation commit or pending snapshot
for verified closure. Annotated v1.0.0 remains at
`10d9ef7bf0618b364f887ff9d279408e3cbc33c9`, object
`1acb6b94f0adfbca85014ad6601dbfc716e3c651`; preserve all eleven existing tags.

Read AGENTS.md, PROJECT_STATE.md, ROADMAP.md, the Phase 9
[closure](../experiments/phase-9/CLOSURE.md), [structured evidence](../experiments/phase-9/closure.json),
[report](../experiments/phase-9/REPORT.md), and private context/blueprint when supplied.
A verified publication record supersedes an accepted tracked pending snapshot.

## Retained inputs and limits

Use the artifact identities in the [model card](MODEL_CARD.md),
[Phase 9 inventory](../experiments/phase-9/artifacts.json) and
[Phase 9 results](../experiments/phase-9/results.json). Full base, experimental SFT,
original tokenizer, LoRA adapter/merged/control models, random baseline and earlier
attempts/failures remain local; a clone is not their backup. Verify hashes before use.
Preserve shared chat contract 1 and both reserved test payloads; routine evaluation
uses validation only. Original optimizer recovery is version-bound; LoRA adapter
optimizer resume is unsupported. Do not silently substitute the new adaptations
for the preserved SFT or claim they are improved baselines.

- Mac: full learned experiment; 236 tests in development and installed wheel.
  Phase 9 base/LoRA/full are all 0/32 exact replies; LoRA/full both regress English
  loss. The original Phase 6 negative SFT and fixed final update 200 remain unchanged.
- Owner-reported Windows RTX 4070 SUPER: 226 passed / 10 skips, fresh wheel suite;
  tiny CUDA LoRA/full, 4 updates / 175 targets, 7,856 frozen base / 192 trainable
  adapter parameters. Tiny fraction ~2.44399%; full-architecture fraction
  0.851951279%. No full Mac learned experiment or full learned 512-position merge
  validation on Windows. Console Ctrl-C remains unvalidated.
- Hosted Linux CPU: separately inspected exact-implementation CI, 228 passed /
  8 MPS-only skips; tiny LoRA/comparison and existing offline workflows. No fresh
  installed-wheel suite, full learned-artifact verification or CUDA/MPS execution.
  Physical Linux remains deferred.
- Preserve amended float32 merge atol=rtol=1e-4 and original CPU 1e-5 failure.
  Cache tolerance is separate: CPU/CUDA 1e-5, MPS 1e-4. No useful instruction
  following, general determinism, cross-device equality, production performance,
  mixed-precision or distributed behavior follows.

## Future scope, only after an explicit fresh-chat start

The roadmap calls for a bounded DPO experiment with a fixed baseline, documented
preference data, objective checks, held-out SFT comparison and English regressions.
No Phase 10 baseline selection, dataset preparation, implementation or training has
been performed here. Use existing resources with zero new paid-service budget;
complete implementation, validation, separate review/fixes and a local commit
before requesting Phase 10 publication approval.

## Concise kickoff prompt — copy only after verified closure publication

> Start LatoS Phase 10 in this fresh chat only after verifying remote main matches
> the owner-approved Phase 9 closure commit and all existing tags remain unchanged,
> including v1.0.0 at 10d9ef7bf0618b364f887ff9d279408e3cbc33c9. Read AGENTS.md,
> PROJECT_STATE.md, ROADMAP.md, docs/PHASE10_HANDOFF.md, Phase 9 closure/evidence and
> local private context/blueprint and publication records when present. Verified
> publication records supersede pending snapshots. Implement a bounded DPO experiment
> with a fixed baseline, documented preference data, objective checks, held-out SFT
> comparison and English regressions using existing resources without new paid
> services. Preserve all models, tokenizer, shared chat format, prior evidence and
> both reserved tests. Retain negative SFT/LoRA quality results, the amended 1e-4
> merge contract and original failure, unsupported adapter optimizer resume, bounded
> tiny Windows/CUDA scope, separately scoped hosted Linux evidence, deferred physical
> Linux and all capability limits. Complete validation and separate review/fixes,
> prepare the local commit and concrete Phase 10 publication proposal, then stop
> for explicit approval.
