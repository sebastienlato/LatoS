> Historical snapshot at Phase 12 closure; superseded by [Roadmap 2.0](../../ROADMAP.md)
> and the [current project state](../../PROJECT_STATE.md). Publication-pending language
> records the original checkpoint, not the current publication status.

# Project state

Updated: 2026-09-23.

## Active checkpoint

**Phase 12 is formally closed locally within its documented bounded scope; the
reviewed closure commit awaits explicit publication approval. Do not begin another
Phase 12+ experiment in this chat.** The owner-reported corrected Windows/CUDA PASS
satisfies the external-validation wait. After approved closure publication, verify
exact remote main and all unchanged tags, then stop. The next extension (Phase 13)
requires an explicit fresh-chat start after verification; [handoff](../../docs/PHASE13_HANDOFF.md).

Validated correction `b222c1fa844289247531b922d5cb18134e99c7b1` is published, package
1.3.0, **no Phase 12 tag**. Original experiment:
`5b72872a2032f8999aae9c28ff0ab5c704e3d893`. Verified publication supersedes historical
pending snapshots. All eleven tags remain fixed, including v1.0.0 at
`10d9ef7bf0618b364f887ff9d279408e3cbc33c9`, object
`1acb6b94f0adfbca85014ad6601dbfc716e3c651`.

Closure: [report](../../experiments/phase-12/CLOSURE.md),
[structured evidence](../../experiments/phase-12/closure.json),
[separate review](../../experiments/phase-12/CLOSURE_REVIEW.md). Original experiment,
portability correction, failed attempts and evidence remain unchanged.

## Distinct evidence and preserved limits

- **Original Mac learned result:** pinned base/SFT/DPO and English tokenizer;
  256→512 enables **44 learned retry transitions**, 12/16/16 by model. At 512:
  0/41, 0/64, 0/64 valid JSON replies, no calls, each 0/12 tasks and 0/4 learned
  failure handling. Two runs, 2,915 tokens each; H1 passes, H2 fails. No research
  rerun for correction/closure. Original suites 312 each; Mac correction suites 314 each.
- **Owner-reported corrected Windows/CUDA PASS:** fresh checkout and non-editable
  wheel each **301 passed / 13 expected skips / zero failures**. Native Windows
  nested inventory roundtrip passes; missing entries/extra files/changed contents
  rejected. RTX 4070 SUPER executes all 512 positions, rejects 513/over-budget inputs
  without truncation; reservation boundaries pass. Cache atol=rtol=1e-5, maximum
  observed absolute difference approximately 1.78814e-7, scoped to this exercise.
- Windows used the original **random synthetic 512-capacity fixture**, fixed across
  both conditions; Mac learned models/tokenizer unavailable and not substituted.
  **16 initial context blocks removed**, zero replies at 256 → one at 512, 1,024 CUDA
  tokens. 0/16 JSON, no calls, 0/12 tasks, 0/4 learned failure handling. This is not
  Mac's 44 learned retry transitions. Scripted 12/12 tasks, 4/4 failures and scripted
  retry controls are mechanics only. Do not combine these into a capability claim.
- Windows owner reports 232 tracked and 3,129 prior evidence/report files preserved,
  including the original failed path-portability run. Reserved payloads unchanged,
  opaque-hashed only; CUDA access guards zero attempts. No Windows changes/commits/
  pushes/tags/uploads. Raw logs/artifact hashes and full skip breakdown not supplied.
  Earlier failure at the original commit: suite 299/13, actual CUDA not reached.
- **Separately inspected hosted Linux CPU** [run 35880051473](https://github.com/sebastienlato/LatoS/actions/runs/35880051473):
  **303 passed / 11 MPS-only skips** at exact correction; Phase 12 eight CPU passes /
  one skip. Synthetic/scripted/inventory/cache/inference checks and workflows/builds.
  Not the Mac research run, Windows CUDA exercise, fresh installed-wheel suite or
  full artifact/access-guard audit. Physical Linux deferred. Dense fixture recovery
  remains separate from tool learning and unsupported adapter/DPO optimizer resume.
- **78,433 local files rehashed unchanged**, including all prior models/tokenizers,
  original experiment/correction evidence, source snapshots and failures. Both
  reserved tests opaque-hashed only. Chat contract 1 and all runtime defaults stay fixed.
- Prior negative SFT/LoRA/control/DPO results remain: full Mac DPO ranking 16/32→15/32,
  exact 0/32; English loss improves over SFT but remains worse than original base.
  Preserve distinct tiny Windows and hosted Linux [Phase 10 evidence](../../experiments/phase-10/CLOSURE.md).
  [Phase 11](../../experiments/phase-11/CLOSURE.md) retains negative full Mac and separate
  compact-prompt tiny Windows tool results, plus separately scoped Linux evidence.
- Merge atol=rtol=1e-4 and original CPU 1e-5 failure retained; cache CPU/CUDA 1e-5,
  MPS 1e-4. Adapter/DPO optimizer resume unsupported; dense recovery version-bound;
  Windows OS-level Ctrl-C unvalidated. No learned tool use, useful instruction
  following, factual reliability, learned long-context quality, general accelerator
  determinism, cross-device equality, safety alignment, production performance,
  mixed precision or distributed serving established. Capacity relief is not learning.

## Pending publication and next action

- Commit message: `Close Phase 12 with scoped external validation`.
- Exact reviewed commit is recorded in the approval request and
  `.private/phase12-closure-publication.json` after preparation.
- Existing remote: `https://github.com/sebastienlato/LatoS.git`, **main**, public unchanged.
- Documentation/evidence only; runtime/version/tests/configs/lock/workflow/original
  evidence unchanged. No new runtime suite, training, benchmark or build for closure.
- **Stop for explicit Phase 12 closure publication approval.** No tag, release or assets.
  After approved publication verify exact main and all unchanged tags, then stop.
  Closure approval does not authorize another experiment in this chat.

Local uv: `.private/tools/bin/uv`; runtime `.venv`. Ignored learned artifacts/raw
logs are not remote backups. No next method, data, baseline or configuration selected.
