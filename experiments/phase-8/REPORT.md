# Phase 8 — reproducible educational release

Prepared locally on 2026-09-21; publication is pending explicit owner approval.
Package 1.0.0 adds reproduction/release documentation, data/model cards, a consolidated
report and an explicitly bounded tiny artifact package. No core model, tokenizer,
chat, training or inference implementation changes; dependency versions are retained.
This milestone does not establish useful assistant quality.

## Verified starting point

Remote main was read back as approved Phase 7 closure
`e4fd79ba7ab1fe5bcc19e733c2f6639a2214ba94`. Annotated v0.8.0 still points to
`04031e5ea98da8db495242165a78c216ab1d4cf4`, tag object
`485aaf36b7ed81a95c948434e1c14634b220be07`. The explicit fresh-chat start satisfies
the prior transition gate. GitHub currently reports the existing repository as
**public**; earlier state snapshots called it private. Phase 8 makes no visibility
change. No remote write or new paid service was used.

## Consolidated experiment outcomes

| Stage | Actual result | Interpretation |
| --- | --- | --- |
| Data | 12 pinned historical books; 9,171 train / 1,030 validation / 2,394 reserved test paragraphs | Narrow prose corpus; document/author separation and declared lexical dedup only |
| Tokenizer | Original 8,192-entry BPE fit only on train; validation 3.9652 UTF-8 bytes/token, zero round-trip failures | Codec correctness, not language quality |
| Tiny training | Original Phase 4 fixture overfit and same-runtime CPU recovery | Mechanics only |
| Full base | 17,308,032 parameters, random seed 17, 3,000 updates, 5,761,229 target exposures | Fixed final checkpoint; no post-hoc best selection |
| Base English validation | Random 9.089003 loss / 8857.350 perplexity; unigram 6.883256 / 975.798; trained 4.731898 / 113.511 | Same 111,523 targets and tokenizer; limited held-out evidence |
| Full experimental SFT | 200 updates, 6,188 assistant-target exposures, seed 61 | Fresh optimizer from preserved base; fixed final update 200 |
| SFT quality | Assistant loss 8.354879 → 5.815233; exact replies 0/32 → 0/32; EOS replies 0/32 → 32/32 | Better teacher-forced loss/stopping, no exact task success |
| SFT regression | English loss 4.731898 → 5.653422; perplexity 113.511 → 285.266 | Negative result; SFT is not an improved base |
| Inference | Cache/streaming/history/stops/cancellation checked within recorded scopes | No new model-quality improvement |

These are historical measured outcomes, not Phase 8 reruns. Both real test sets
remain reserved. Language-modeling and assistant-only losses are different
objectives. Training/evaluation windows were at most 256 positions; numerical
capacity 512 does not establish language quality at 512. Full fixed samples,
failures, source/runtime identities and methodology remain in
[Phase 5](../phase-5/REPORT.md), [Phase 6](../phase-6/REPORT.md) and
[Phase 7](../phase-7/REPORT.md). The [data card](../../docs/DATA_CARD.md) and
[model card](../../docs/MODEL_CARD.md) identify source terms, transformations,
checkpoint hashes and intended uses.

Historical resource results: the full base run took 417.224 seconds before final
summary/inventory hashing, with synchronized updates at 14,628.5 targets/s. SFT
took 25.902 seconds before inventory hashing, updates at 437.25 assistant targets/s.
Those are different workloads, not comparable optimization speeds. Phase 7's fixed
128-ID prefix/32-prediction workload measured CPU cache decode ratios 6.40×/6.56×
(base/SFT) and MPS 1.26×/1.29×; prefill did not improve. These are warm run medians
on one M4 Max, not production latency guarantees. Logical KV storage at batch one,
512 positions is 12 MiB, excluding temporaries. RSS and MPS boundary snapshots are
not independent memory pools or continuous accelerator peaks. Full definitions and
numbers remain in the original reports; Phase 8 ran no performance comparison.

## Platform evidence and limits

- Historical Mac CPU/MPS Phase 7: 210 tests in development and fresh wheel; full
  preserved base/SFT cache checks through 512. Same-backend tolerances CPU
  atol=rtol=1e-5 and MPS atol=rtol=1e-4; fixed greedy IDs matched.
- Historical Windows / RTX 4070 SUPER Phase 7: **owner-reported PASS**, 202 tests,
  eight skips, fresh wheel and actual CUDA CLI. Tiny base/SFT through capacity
  256, plus a separate synthetic 512-position/batch-two model. Same-CUDA logits
  met atol=rtol=1e-5; fixed cached/uncached/non-streaming greedy IDs and callback
  cancellation agreed. **Full Mac learned artifacts were unavailable and not
  validated there; Windows console Ctrl-C remains unvalidated.** Raw Windows
  logs/hashes/latency numbers were not supplied here.
- Historical GitHub-hosted Linux CPU: separately inspected
  [run 35529617332](https://github.com/sebastienlato/LatoS/actions/runs/35529617332),
  203 tests / seven MPS-only skips; tiny CPU inference/CLI/POSIX SIGINT, existing
  SFT/recovery fixtures, locked lint/build/offline workflows. No full learned
  artifacts, CUDA/MPS, retained-artifact latency matrix or fresh-wheel suite.
- Physical Linux remains **deferred, not performed**. Phase 8 has no new Windows,
  CUDA or Linux execution evidence. Useful instruction following, general CUDA
  determinism, cross-device equality, production latency, mixed precision and
  distributed serving remain unestablished.

See [Phase 7 closure](../phase-7/CLOSURE.md) for attribution. Do not present prior
platform results as testing the exact 1.0.0 package or the downloadable tiny zip.

## Phase 8 validation

Host rechecked: Mac16,6 / M4 Max, 64 GiB, macOS 26.6.2 arm64, about 627 GiB free
at start. Python 3.14.7, PyTorch 2.14.0, uv 0.12.15; CPU/MPS available, CUDA absent.
No dependency upgrades. [validation.json](validation.json) records compact results;
raw logs stay under ignored `outputs/phase-8-validation/`.

- Development: **213 passed**. Three release tests exercise exact allowlists and
  hash verification, exclusion of unrelated/private sibling material, deterministic
  zip bytes, no overwrite, corrupt/expanded inventory rejection and symlink refusal.
- Fresh index checkout containing only staged tracked files, new locked environment:
  **213 passed**; documented 400-update offline fixture and tiny JSON chat passed.
- Fresh non-editable built-wheel installation in that isolated environment:
  **213 passed**; import path verified in site-packages. Both preserved full Mac
  model-only snapshots loaded and produced one-shot JSON replies on CPU and MPS.
  These are compatibility smokes, not new quality or latency measurements.
- Fixture train loss **5.814638 → 0.010030**, 131,600 targets; exact CPU recovery
  across 303 replayed updates. Fixture validation loss worsened **5.806987 →
  11.245802**. This confirms memorization and does not establish generalization.
- All five tiny artifact files reproduced byte-for-byte in the isolated checkout.
  The actual tokenizer has **328 entries**, despite the configured maximum 512.
  Tiny model: 127,808 parameters, capacity 64. Inventory pins its exact identities.
- Locked resolution, lint/format, source/wheel build, source member identity checks,
  zip member/hash inspection, Markdown links and publication/privacy checks passed.
- **441 distinct preserved files** rehashed unchanged: the prior 357-entry set,
  Phase 5/6/7 inventories (25/49/83 entries, overlapping) and shared chat source.
  Both real tests were integrity-hashed as opaque bytes only, never parsed/evaluated.

The isolated index checkout tests the implementation snapshot before final report
wording; it is not falsely identified as a prior published commit. Final approved
asset hashes and exact candidate commit are supplied in the local publication
record/approval proposal after packaging. A final local clone validates that commit
before requesting approval. No fresh full English training or original optimizer
recovery is claimed; those require separately retained inputs and original runtimes.

## Artifacts, review and publication

Proposed release assets: `latos-1.0.0.tar.gz`, `latos-1.0.0-py3-none-any.whl`,
`latos-1.0.0-tiny-fixture.zip` and `SHA256SUMS`. Source comes only from reviewed
tracked files; wheel contains package code/license; tiny zip explicitly contains
five pinned learned fixture files, two cards, MIT license and an internal manifest.
Acquired corpora, full Mac base/SFT, English tokenizer, optimizer states, full logs,
environments and private context are excluded. Full artifacts stay preserved
locally; neither source publication nor this tiny release backs them up.

[Separate review](REVIEW.md) corrected the requested-versus-actual tiny vocabulary
count and clarified current remote visibility and historical platform scopes.
[Release guide](../../docs/RELEASE.md) separates mechanism reproduction, model-only
loading and version-bound optimizer recovery. No remaining actionable review
finding was identified within this bounded scope.

Exact publication proposal: existing `https://github.com/sebastienlato/LatoS.git`,
`main`, new annotated `v1.0.0`, GitHub release and the four reviewed assets above.
Preserve all existing tags, including v0.8.0. No PyPI upload or visibility change.
Phase 8 publication approval is pending; Phase 9 has not begun.
