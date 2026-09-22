# Phase 11 fresh-chat handoff

Preparation only. **Do not begin Phase 11 in the Phase 10 closure chat.** Obtain
explicit closure publication approval, publish that exact commit, verify remote
main and all unchanged tags, then stop. Phase 11 needs an explicit fresh-chat start
after that verification. This document is not authorization or implementation.

## Starting state

Phase 10 implementation is published at
`abed1adffba562373348477e3958bcd1d5886b11`, package 1.2.0, **without a Phase 10 tag**.
The separate closure commit is identified in the owner's approval/publication
confirmation and `.private/phase10-closure-publication.json` when present. Match that
exact approved closure against remote main; do not mistake the implementation
commit or tracked pending snapshot for verified closure. Verified publication
records supersede pending snapshots. Preserve all eleven tags, including annotated
v1.0.0 at `10d9ef7bf0618b364f887ff9d279408e3cbc33c9`, object
`1acb6b94f0adfbca85014ad6601dbfc716e3c651`.

Read AGENTS.md, PROJECT_STATE.md, ROADMAP.md, Phase 10 [closure](../experiments/phase-10/CLOSURE.md),
[structured evidence](../experiments/phase-10/closure.json), [report](../experiments/phase-10/REPORT.md),
and private context/blueprint/publication records when present.

## Preserved inputs and limits

Use the [model card](MODEL_CARD.md), [Phase 9 inventory](../experiments/phase-9/artifacts.json),
[Phase 10 inventory](../experiments/phase-10/artifacts.json) and
[results](../experiments/phase-10/results.json) for identities. Rehash before reuse.
Base/SFT/LoRA/merged/control/DPO models, random baselines, original tokenizer,
prior attempts/failures and both reserved tests stay preserved. A clone is not a
backup of ignored learned artifacts. Do not silently promote an adaptation or DPO
model as a better baseline. Keep shared chat contract 1; validation only for routine
checks. Original dense recovery is version-bound; adapter/DPO optimizer resume is
unsupported.

- Mac full DPO: 100 updates / 400 pairs / 2,498 final-response targets; ranking
  **16/32 → 15/32**, exact replies **0/32 → 0/32**, preference loss worsens.
  English **5.653422 → 5.603780**, still worse than base **4.731898**. No preference-
  learning or instruction-following improvement. Negative SFT/LoRA results retained.
- Owner-reported Windows RTX 4070 SUPER: **243 passed / 11 expected skips**, fresh
  wheel suite; tiny CUDA **4 updates / 16 pairs / 169 response targets**, frozen CUDA
  reference, updated policy, exact same-CUDA reload. Tiny ranking **16/32 → 16/32**,
  exact **0/32 → 0/32**; not a reproduction of full Mac learned SFT/reference, corpus
  or ranking regression. Raw logs/hashes not supplied in this closure session.
- Separately inspected hosted Linux CPU: **245 passed / 9 MPS-only skips**, tiny
  two-update CPU DPO and offline workflows/builds. No fresh installed-wheel suite,
  full learned-artifact validation, CUDA/MPS or physical Linux. Physical Linux deferred.
- Preserve amended merge atol=rtol=1e-4 and original CPU 1e-5 failure; separate cache
  bounds CPU/CUDA 1e-5, MPS 1e-4. Windows console Ctrl-C remains unvalidated. No useful
  assistant, DPO superiority, safety alignment, general accelerator determinism,
  cross-device equality, production performance, language quality beyond 256,
  mixed precision or distributed behavior is established.

## Future scope, only after explicit fresh-chat start

Roadmap Phase 11 calls for structured calls to bounded tools, measuring parsing,
argument correctness, task success and failure behavior separately. No Phase 11
tool selection, schema, dataset, baseline selection, implementation or training has
been performed here. Use existing resources with zero new paid-service budget.
Implement, validate, separately review/fix and prepare a local commit before
requesting Phase 11 publication approval.

## Concise kickoff prompt — only after verified closure publication

Use the exact closure commit from the publication confirmation when copying this
prompt; do not use the Phase 10 implementation commit as the closure identity.

> Start LatoS Phase 11 in this fresh chat only after verifying remote main resolves
> to the exact approved and verified Phase 10 closure commit and all eleven existing
> tags remain unchanged, including v1.0.0 at 10d9ef7bf0618b364f887ff9d279408e3cbc33c9.
> Read AGENTS.md, PROJECT_STATE.md, ROADMAP.md, docs/PHASE11_HANDOFF.md, Phase 10
> closure/evidence and local private context/blueprint/publication records when present.
> Verified publication records supersede pending snapshots. Implement bounded
> structured tool calls with parsing, argument correctness, task success and failure
> behavior measured separately, using existing resources without new paid services.
> Preserve all models, tokenizer, shared chat format, prior evidence and both reserved
> tests. Retain the negative Mac DPO result, distinct tiny Windows/CUDA result,
> separately scoped hosted Linux evidence, deferred physical Linux, unsupported
> adapter/DPO optimizer resume, amended merge contract/original failure and all
> capability limits. Complete validation and separate review/fixes, prepare the local
> commit and concrete Phase 11 publication proposal, then stop for explicit approval.
