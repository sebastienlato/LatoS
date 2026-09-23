# Project state

Updated: 2026-09-23.

## Active checkpoint

**Phase 14 — Evaluation Foundation.** Implementation, historical evaluation,
validation and separate review are complete locally. One bounded reviewed commit
is prepared for explicit publication approval; no remote write has occurred.

Authoritative start: local HEAD/main/origin/main and remote main matched published
Phase 13 `565bf08fbd8aaa360056fc670b08b8879be9312c`, clean, with all eleven tags
unchanged. Its historical pending snapshot is superseded by verified publication.
Package stays **1.3.0**; stable **v1.0.0** remains at
`10d9ef7bf0618b364f887ff9d279408e3cbc33c9`.

## Completed scope and evidence

- [Fixed evaluation](docs/EVALUATION.md): independent next-token likelihood,
  historical LM objective, tokenizer-aware matched-text BPB, 24 generation prompts,
  96 objective instruction cases in six categories, pinned ARC-Easy/Challenge
  answer-likelihood evaluation, separate raw records/summaries and regression gates.
- [Protocol](configs/evaluation/protocol-v1.json) freezes Phase 17/18 improvement,
  minimum-quality and regression bounds before model improvement. No composite
  score. Another 48 instruction cases are reserved for final acceptance, unscored.
  Final access requires a locked candidate, passing development gates and reviewed
  contamination evidence. Both historical reserved payloads remain unused here.
- [New retrospective results](experiments/phase-14/REPORT.md): all five preserved
  base/SFT/merged-LoRA/full-control/DPO models score **0/96 instructions**, every
  category 0/16. Base legacy NLL **4.731899**, matched BPB **1.872944**, ARC-Easy
  **157/570 (27.54%)**, Challenge **66/299 (22.07%)**. No useful assistant or
  broad reasoning quality is established; no historical model is promoted.
- Five original artifacts were available; merged LoRA is explicitly the saved
  dense merge, not a new live-adapter quality measurement. Raw evidence remains
  in `outputs/phase-14-evaluation-reviewed`; original attempt retained separately.
  [Hashes](experiments/phase-14/artifacts.json) identify local evidence, not backups.
- Optional PyArrow **25.0.1** only; lock retains prior dependency versions. Two
  evaluation-only Parquet files total **141,823 bytes**, CC BY-SA 4.0, pinned
  revision/content hashes. Acquired questions and generated raw logs stay ignored.
  Development overlap audit reports zero exact/13-word matches, with explicit
  short-text/semantic limits; future training needs a fresh exclusion audit.
- All Phase 5–12 evidence and negative findings remain intact. SFT/LoRA/control
  original 0/32, DPO ranking 16/32 → 15/32, and failed learned tools are unchanged.
  No architecture, production tokenizer, training mechanism or learned weights changed.

## Validation and review

**353 tests passed** in development and **353** from a fresh non-editable wheel,
with zero failures/skips on this Mac CPU/MPS. Lint/format, locked dependencies,
builds, CPU doctor, evaluation CLI, fresh pinned data acquisition and source/archive
inspection pass. The base's separate-process repeat passes all raw/aggregate
numerical checks with identical generated IDs and predictions. All **72,947**
pre-existing artifact/evidence files rehashed unchanged; reserved payloads opaque
only. [Validation](experiments/phase-14/validation.json) and
[review fixes](experiments/phase-14/REVIEW.md) record scope and attempts.

No Phase 14 CUDA/Windows or hosted/physical Linux run is claimed. CI now includes
the optional evaluation reader tests, but has not run for this unpublished state.
MPS resource observations are single-host measurements, not production benchmarks.
No new paid service or serious training. Synthetic fixture tests are mechanics only.

## Pending publication and next action

- One local commit: **Add fixed evaluation foundation and historical baselines**.
  Exact SHA belongs in the approval request and ignored local publication record;
  this committed file cannot embed its own SHA.
- Existing remote: `https://github.com/sebastienlato/LatoS.git`; destination **main**.
  No new tag, release, asset upload, remote backup or visibility change.
- Stop at **“Push Phase 14 to GitHub?”** and await explicit approval for that exact
  commit. After approval, publish only the described state and verify remote main
  and all eleven unchanged tags.
- **Do not begin Phase 15 until Phase 14 publication is approved and verified AND
  the owner/master-planning process explicitly authorizes Phase 15.** Publication
  approval alone never lifts this gate. No Data 2.0 work has begun.

Local uv: `.private/tools/bin/uv`; runtime: `.venv`. Learned artifacts and raw
results are preserved locally, outside source publication. New paid-service budget
remains zero. Mac is authoritative; future GPU work uses the existing RTX 4070 SUPER.
