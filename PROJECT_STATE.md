# Project state

Updated: 2026-09-21.

## Active checkpoint

**Phase 9 is complete, validated and reviewed locally. Publication approval is PENDING.**
The owner explicitly started
Phase 9 after remote main and annotated v1.0.0 were verified at
`10d9ef7bf0618b364f887ff9d279408e3cbc33c9`; tag object
`1acb6b94f0adfbca85014ad6601dbfc716e3c651`. The verified Phase 8 publication record
supersedes its tracked pending-publication snapshot. No Phase 9 remote write.

Delivered: float32 attention LoRA, frozen base/adapter-only optimization, bound
adapter snapshots, non-mutating dense merge, CLI training/merge and an equal-budget
full-tuning comparison. See [guide](docs/ADAPTATION.md),
[report](experiments/phase-9/REPORT.md), [review](experiments/phase-9/REVIEW.md) and
[validation](experiments/phase-9/validation.json). Package 1.1.0; dependencies unchanged.

## Evidence and limits

- Mac CPU/MPS: **236 tests passed** in development and an isolated non-editable wheel.
  Full saved artifacts pass read-back, frozen-base, exact adapter reload and merge/cache checks.
- Both methods: same base/tokenizer/data/schedule/shuffle, 200 updates, 6,188 assistant
  targets, fixed final selection. LoRA trains 147,456 / 17,308,032 parameters (0.852%).
  Assistant losses: base 8.354879, LoRA 3.762932, full 5.814905; exact replies **0/32
  for all three**. English losses: 4.731898 / 4.773844 / 5.653425. No useful assistant.
- Full float32 merge requires atol=rtol=1e-4 on CPU/MPS. Initial CPU 1e-5 failure is
  retained with an explicit protocol amendment; float64 diagnostic agrees to
  3.73e-14. Historical CPU cache 1e-5 and MPS cache 1e-4 remain unchanged.
- Original Phase 6 SFT remains negative and unchanged: 200 updates / 6,188 targets,
  assistant loss 8.354879 → 5.815233; exact 0/32 → 0/32; English 4.731898 → 5.653422.
  Retain final update 200; do not promote it or either new adaptation as an improved base.
- **507 preserved files rehashed unchanged**, including the prior 441-file set.
  Full base/SFT/tokenizer, shared chat contract, all prior evidence/attempts and both
  reserved test sets remain local and preserved. Tests are integrity-hashed only.
  New adapters/merged/control weights stay under ignored outputs. No remote backups.
- Windows historical evidence: tiny 256-position artifacts plus synthetic 512,
  not full Mac learned artifacts; console Ctrl-C unvalidated. Hosted Linux evidence
  is separately scoped to historical tiny CPU/workflows. Physical Linux deferred.
  No Phase 9 Windows/Linux/CUDA execution. Useful quality, general determinism,
  cross-device equality, production latency, mixed precision and distributed serving
  remain unestablished. Adapter optimizer resume is unsupported.

## Pending publication and next action

- Reviewed local commit message: `Add Phase 9 bounded LoRA adaptation and merge`.
- Existing remote: `https://github.com/sebastienlato/LatoS.git`, branch **main**.
- Proposed source commit only; **no tag, release or uploaded artifacts**. Preserve
  every prior tag, release and repository visibility; no new paid services.
- Lint/format, locked setup, source/wheel build, archive inspection, installed-wheel
  full merge and CPU/MPS chat, preservation and privacy checks passed.
- **Next action: stop for “Push Phase 9 to GitHub?”** Approval is pending.
  Exact commit is in the approval request and local publication record.
  Phase 10 has not begun; this request does not authorize a Phase 10 transition.

Local uv: `.private/tools/bin/uv`; runtime `.venv`. Exact publication identity and
approval status are in the local `.private/phase9-publication.json` record.
