# Project state

Updated: 2026-09-23.

## Active checkpoint

**Phase 15 — Data 2.0.** Implementation, actual data preparation, tokenizer
comparison, validation and separate local review are complete. A reviewed local
source/evidence commit is prepared for explicit publication approval. No Phase 15
remote write or Phase 16 implementation has occurred.

Authoritative start: HEAD/main/origin/main and remote main matched published
Phase 14 `8eaf32d06df43449529ec7d2a1827637261a0f47`, clean. Eleven tag objects/targets
and the existing release/four assets were verified unchanged. The local publication
record superseded Phase 14's historical pending snapshot; the owner explicitly
authorized Phase 15 in this fresh Mac task. Package stays **1.3.0**; **v1.0.0** stays
at `10d9ef7bf0618b364f887ff9d279408e3cbc33c9`.

## Completed scope and evidence

- [Audited data foundation](experiments/phase-15/REPORT.md): **158,795,397 training
  pretraining bytes / 37,226,334 content tokens**, plus separate development and
  reserved splits; **14,259 training conversations**, including **734 multi-turn**.
  Pretraining bytes are 29.76 times the historical training text; conversations
  are 74.27 times the original 192. Repeated exposures are counted separately.
- [Sources and rights](docs/DATA_2.md): pinned Wikimedia Wikipedia, Dolly and OASST1;
  reviewed terms, notices, lineage, deterministic selection, language/quality/privacy
  filters, source/group/template isolation, exact/near joins and rejection ledgers.
  Acquired text and learned tokenizers remain ignored, with no payload redistribution.
- [Final audit](experiments/phase-15/contamination.json): **361,515 records**, zero
  observed remaining exact/near full-record pairs, zero cross-split groups and zero
  protected matches. All ARC splits, historical reservations, frozen suites and
  outputs are protected. Reserved access was fingerprint-only; no scoring, example
  export or final override. Lexical checks do not certify semantic absence.
- [Selected 16,384 BPE](experiments/phase-15/tokenizer-comparison.json): 9.67% fewer
  new-development tokens than new 8,192, 25.39% fewer than historical 8,192, no source
  regression versus new 8,192. Fitting used new training only. Native codec/chat
  contract 1 and fixed BPB spans remain compatible; old token IDs/weights are intact.
- [Local Phase 16 package](docs/PHASE16_INPUTS.md): `outputs/phase15-input-package`,
  with verified train/development data, selected tokenizer and source notices.
  No weights/reserved/evaluation examples bundled. Length/padding, exposure and
  hypothetical time/embedding costs are recorded; no CUDA feasibility claim.

## Validation and limits

**390 tests passed** in development and **390** in an isolated non-editable wheel,
with no failures/skips. Lint/format, locked dependencies, builds, CLI/CPU doctor,
package inventory and local links pass. A 4,000-real-candidate preparation replay
is byte-identical; full-scale final curation repeats identically. All **73,679**
historical data/artifact/evidence files rehash unchanged. See
[validation](experiments/phase-15/validation.json) and [review fixes](experiments/phase-15/REVIEW.md).

The undersized complete run, partial attempts, intermediate curation outputs and
failed entrypoint/test attempts remain local. Resource-based plan amendments did
not lower acceptance floors. Final full preparation took 1,273.29 seconds and
8.73 GB peak process RSS on this Mac; [measurements](experiments/phase-15/resources.json)
are not GPU/production benchmarks. No Phase 15 Windows/CUDA or Linux execution;
physical Linux remains deferred. No new paid resources or substantive model training.
Normal tests include disposable synthetic optimization fixtures.

Encyclopedia dominance, imperfect contributor quality/lineage, typos, omitted short
leads/lists, lexical detection limits and incomplete worldwide redistribution
clearance remain explicit. Construction acceptance is not learned-quality success.
Historical results remain **1.872944 BPB**, ARC-Easy **157/570 (27.54%)**, Challenge
**66/299 (22.07%)**, and all five artifacts **0/96 instructions**, every category
**0/16**. Separate original SFT, adaptation, DPO and learned-tool negatives persist.

## Pending publication and next action

- Proposed commit: **Build audited Data 2.0 corpora and tokenizer inputs**.
  Exact SHA belongs in the approval request and ignored local publication record;
  this file cannot embed its own commit SHA.
- Existing remote: `https://github.com/sebastienlato/LatoS.git`; destination **main**.
  No new tag, release, assets, remote backup or visibility change.
- Stop at **“Push Phase 15 to GitHub?”** for explicit approval of the exact reviewed
  commit. After approval, publish that state and verify remote main and unchanged
  tags/release/assets. Subsequent phase transitions follow the standing rules and
  any active owner-imposed gate; publication approval cannot override a later pause.

Local uv: `.private/tools/bin/uv`; runtime `.venv`. The Mac remains authoritative;
future substantive GPU work targets the existing RTX 4070 SUPER (~12 GB VRAM).
