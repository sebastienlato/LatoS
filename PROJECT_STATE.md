# Project state

Updated: 2026-09-23.

## Active checkpoint

**Phase 11 is implemented, validated and separately reviewed locally; stop for
explicit publication approval. Phase 12 has not begun.** Package 1.3.0, no new tag.
The owner explicitly started this fresh chat after remote main was verified at
Phase 10 closure `87d744fb8cfd4228c78f35bbd220d2833462e037`. All eleven local/remote
tag objects and peeled targets match the verified closure publication record.
v1.0.0 remains at `10d9ef7bf0618b364f887ff9d279408e3cbc33c9`, object
`1acb6b94f0adfbca85014ad6601dbfc716e3c651`. Verified publication supersedes old
tracked pending snapshots. No Phase 11 remote write has occurred.

Delivered [bounded tools](docs/TOOLS.md): strict JSON, pure addition/lookup, four
assistant turns and two executions maximum, explicit errors/cancellation/overflow,
unchanged shared chat contract 1. [Report](experiments/phase-11/REPORT.md),
[results](experiments/phase-11/results.json), [validation](experiments/phase-11/validation.json)
and [separate review](experiments/phase-11/REVIEW.md) retain all evidence.

Fixed Mac MPS evaluation of unchanged base/SFT/DPO: each **0/16 valid JSON, 0/20
correct required calls, 0/12 normal tasks, 0/4 expected failures handled**. No calls
were proposed/executed; conditional argument/execution rates are undefined, not
perfect. Scripted control passes 12/12 tasks and 4/4 failures; that is mechanics only.
All learned sessions emitted one invalid reply; retry context exhaustion occurred
in 12 base and all 16 SFT/DPO sessions, with four incomplete base generations.
No learned multi-turn tool execution or useful tool quality is established.
Development and isolated fresh non-editable wheel suites each passed **305 tests**;
independent evidence recount verified 64 sessions and 49 inventory files.
No training, dependency upgrade, corpus evaluation or paid service. Full local
preservation includes 59,873 files; reserved tests opaque-hashed only.

## Prior Phase 10 evidence and preserved limits

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

- Commit message: `Add bounded structured tools and separate capability evaluation`.
- Existing remote: `https://github.com/sebastienlato/LatoS.git`, destination **main**;
  public visibility unchanged. No tag, release, asset upload or remote backup.
- Exact reviewed commit is supplied in the approval request and ignored local
  `.private/phase11-publication.json`; approval is pending, no push performed.
- Full development and isolated fresh non-editable wheel suites, independent saved
  evidence recount, packaging, preservation and privacy checks are in validation.json.
- **Stop and ask: Push Phase 11 to GitHub?** Publication approval covers this exact
  phase state only. Retain the reviewed checkpoint while awaiting that approval.

Local uv: `.private/tools/bin/uv`; runtime `.venv`. Learned artifacts and raw runs
stay ignored and local; a repository clone is not their backup.
