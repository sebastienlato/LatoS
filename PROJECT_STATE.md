# Project state

Updated: 2026-09-22.

## Active checkpoint

**Phase 10 is complete locally, separately reviewed, and awaits explicit publication
approval. Stop here. Phase 11 has not begun.** Package 1.2.0 adds a bounded DPO
experiment with a fixed SFT reference and documented synthetic preferences.

Before starting, verified remote main at Phase 9 closure
`7d19649cabcc0698bd77042779772bc2196c4bc2` and all eleven unchanged tag objects/targets.
The owner explicitly authorized this fresh-chat start. Verified publication records
supersede historical tracked pending snapshots. Annotated v1.0.0 remains at
`10d9ef7bf0618b364f887ff9d279408e3cbc33c9`, object
`1acb6b94f0adfbca85014ad6601dbfc716e3c651`.

Delivered: [report](experiments/phase-10/REPORT.md), [fixed protocol](experiments/phase-10/PLAN.md),
[data provenance](experiments/phase-10/DATA.md), [results](experiments/phase-10/results.json),
[verification](experiments/phase-10/verification.json), [review](experiments/phase-10/REVIEW.md),
[validation](experiments/phase-10/validation.json) and [usage](docs/PREFERENCES.md).

## Phase 10 evidence

- Original Phase 6 fixed-update-200 SFT initializes both policy and separate frozen
  reference; no substitution of LoRA/full control. Original tokenizer/chat contract.
  100 MPS updates / 400 pair exposures / 2,498 final-response targets; 192 train and
  32 validation pairs, fixed final selection, no sweep, new service or paid compute.
- **Negative held-out preference result:** loss 0.693147 → 0.718657; raw chosen
  ranking 16/32 → 15/32; exact replies 0/32 → 0/32. Assistant loss 5.815233 →
  5.791526; English loss 5.653422 → 5.603780, still worse than base 4.731898.
  No promoted model, useful assistant, human preference or safety claim.
- **254 passed** in development and isolated non-editable wheel. Tiny CPU/MPS DPO,
  scalar objective/gradients, masking/EOS, frozen reference, CPU repetition,
  failures, CLI and inventory rejection; reserved fixture test files absent.
  Full learned MPS round trip exact over SFT validation; independent token-score
  error <=3.864105e-6 at 1e-4 bound. All update exposures replayed.
- Separate review fixed protocol/runner inventory capture and an unsupported MPS
  float64 conversion in the checker. First learned attempt and original failure
  log retained. Reviewed repeat uses unchanged training settings, not quality selection.
- All 743 closure-preserved files and the broader 54,236-file pre-phase artifact/
  evidence snapshot rehashed unchanged. Both reserved tests integrity-hashed only,
  never parsed/evaluated. Full models, tokenizer and previous attempts remain local.

## Retained limits

Original SFT remains negative (200 updates / 6,188 targets, exact 0/32, English
4.731898 → 5.653422). Phase 9 base/LoRA/full exact replies remain 0/32; English
4.731898 / 4.773844 / 5.653425. No useful assistant or LoRA superiority established.

Phase 9 Windows RTX 4070 SUPER is owner-reported bounded tiny evidence: 226 passed /
10 skips, fresh wheel, 4 updates / 175 targets per method, 7,856 base / 192 adapter
parameters (~2.44399%, distinct from full-architecture 0.851951279%). Full Mac learned
experiment and 512-position learned merge not reproduced there; raw logs/hashes
not supplied. Windows console Ctrl-C remains unvalidated. Separately inspected
hosted Linux CPU run 35617504810: 228 passed / 8 MPS skips, tiny CPU/workflow scope,
no fresh installed-wheel suite or full learned verification. Physical Linux deferred.
No Phase 10 Windows/CUDA or Linux execution is claimed.

Merge atol=rtol=1e-4 stays the amended contract; original CPU 1e-5 failure retained.
Cache bounds remain CPU/CUDA 1e-5 and MPS 1e-4. Adapter optimizer resume is unsupported;
DPO outputs are also model-only, without optimizer resume. Scalar CPU diagnostics
are not mixed-precision training. Familiar synthetic validation templates, shared
preference/SFT validation prompts, weak SFT baseline and length-sensitive sequence
scores limit interpretation. Useful quality, language quality beyond 256, general
accelerator determinism, cross-device equality, production performance, mixed
precision and distributed serving remain unestablished.

## Pending publication

- Commit message: `Add bounded DPO experiment with fixed SFT reference`.
- Exact reviewed commit: supplied in approval request and local publication record
  `.private/phase10-publication.json` after committing; no self-referential hash here.
- Remote: `https://github.com/sebastienlato/LatoS.git`, branch **main**, public unchanged.
- **Approval pending. No Phase 10 remote write.** Propose source commit only, no tag,
  release, asset upload, visibility change or remote backup. Preserve all eleven tags.
- Next action: ask “Push Phase 10 to GitHub?” and stop for explicit approval.
  No Phase 11 implementation is included or started.

Local uv: `.private/tools/bin/uv`; runtime `.venv`. Learned artifacts under
`outputs/phase-10-dpo-reviewed` are local inputs/outputs, not remote backups.
