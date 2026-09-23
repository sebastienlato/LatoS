# Project state

Updated: 2026-09-23.

## Active checkpoint

**Phase 13 — Repository Repositioning & Roadmap 2.0.** Documentation/positioning
work is complete locally, validated and separately reviewed. The reviewed local
commit awaits explicit publication approval. No remote write has occurred.

Phases 0–12 are complete within their documented scopes. Phase 12 closure is
published at `c74290ee1208dc6b0e4077ce3c7b2aad262f3c42`, verified as the clean,
synchronized starting main for this phase. Older pending-publication snapshots
are historical. Package version remains **1.3.0**; current stable release is
**v1.0.0**, fixed at `10d9ef7bf0618b364f887ff9d279408e3cbc33c9`.

## Completed scope and retained evidence

- Rewritten landing page, active identity wording, documentation/evidence index,
  corrected release guidance, historical roadmap/checkpoint snapshots and retired
  research handoff. Development and publication safeguards remain in AGENTS.md.
- [Roadmap 2.0](ROADMAP.md) defines Phases 13–21: repositioning, evaluation, Data 2.0,
  model/CUDA training, base quality, assistant quality, interoperability, justified
  alignment/tool learning, then the bounded LatoS 2.0 release. Quality failures
  block dependent phases; mechanisms and successful execution cannot waive them.
- No architecture, training behavior, weights, data, dependency, lock, test or
  workflow changes. Source edits only change CLI description/package docstring;
  package metadata only changes the description. Historical source fingerprints
  still require original checkouts. No new learned-model experiment or training run.
- SFT, LoRA and full tuning retain 0/32 exact replies. Full Mac DPO ranking remains
  16/32 → 15/32, exact 0/32; English loss improves versus SFT but remains worse than
  the original base. No useful instruction following or preference improvement.
- Phase 12 retains **44 Mac learned retry transitions** and zero learned tool
  success. Windows' **16 synthetic initial blocks removed** is separate evidence;
  scripted controls and hosted Linux CPU CI are also separate. Physical Linux
  remains deferred. See the [evidence index](docs/INDEX.md), unchanged
  [Phase 12 closure](experiments/phase-12/CLOSURE.md), and
  [prior detailed checkpoint](docs/history/PHASE12_CHECKPOINT.md).

## Validation and review

**314 tests passed** on Mac CPU/MPS. Locked sync, dependency compatibility,
lint/format, CLI/CPU doctor, source/wheel builds, archive inspection and isolated
non-editable wheel smoke checks pass. Local documentation links resolve; separate
review findings are fixed. **80,180** existing local files rehashed unchanged;
all historical experiments and both reserved test payloads remain intact. All eleven
tags and the existing release/four asset records are unchanged. No learned artifacts
or private planning content are included. [Validation/review](docs/PHASE13_VALIDATION.md)
records exact scope and limitations. No unresolved Phase 13 blocker; no new
Windows/CUDA or hosted Linux CI result is claimed.

## Pending publication and next action

- One reviewed local commit: `Reposition LatoS and define Roadmap 2.0`.
  Its exact SHA is recorded in the approval request and ignored local
  `.private/phase13-publication.json` after commit creation; the committed state does not embed its own SHA.
- Existing remote: `https://github.com/sebastienlato/LatoS.git`; destination **main**.
  No new tag, release, assets, backup branch or visibility change proposed.
- Stop at **“Push Phase 13 to GitHub?”** for explicit approval of that exact commit.
  After approval, publish only that state and verify remote main and unchanged tags/releases.
- **Do not begin Phase 14 until Phase 13 publication is approved, completed and
  verified AND the owner/master-planning process explicitly authorizes Phase 14.**
  Push approval alone never lifts this gate. No Phase 14 implementation has begun.

Local uv: `.private/tools/bin/uv`; runtime: `.venv`. Ignored learned artifacts and
logs are retained locally, not backed up by source publication. New paid-service
budget remains zero. The Mac is authoritative; planned future CUDA training uses
the existing Windows RTX 4070 SUPER (~12 GB VRAM), not rented compute.
