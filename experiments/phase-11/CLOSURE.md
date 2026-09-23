# Phase 11 closure — scoped external validation

Recorded 2026-09-23. **Phase 11 is formally closed locally within its documented
bounded scope; the reviewed closure commit awaits explicit publication approval.
Do not begin Phase 12 in this chat.** The owner-supplied Windows/CUDA PASS satisfies
the external-validation wait. After approved closure publication, verify exact
remote main and all unchanged tags, then stop. Phase 12 requires an explicit fresh-chat
start after verification; the [handoff](../../docs/PHASE12_HANDOFF.md) is preparation only.

Validated implementation: `41fc3cffe42a1ae7a42446c5785abfffc1f559db`, published and
verified on main, package 1.3.0. **There is no Phase 11 tag.** All eleven existing tags
remain fixed, including annotated v1.0.0 at
`10d9ef7bf0618b364f887ff9d279408e3cbc33c9`, object
`1acb6b94f0adfbca85014ad6601dbfc716e3c651`. Verified publication supersedes the original
tracked pending snapshot. Original experiment files and implementation are unchanged.

## Windows/CUDA — owner-reported independent PASS

The owner reports a fresh exact-commit checkout and locked install, **293 passed /
12 expected platform/hardware skips / zero failures**, and a complete suite PASS
from a fresh non-editable wheel installation. Phase 11 contributes **50 passes and
one MPS-only skip in each environment**. GPU: NVIDIA RTX 4070 SUPER. These results
are supplied by the owner, not independently executed on this Mac.

Reported passing scope:

- Structured JSON parsing, envelope and argument-schema validation, bounded execution,
  failure behavior and separate metric accounting.
- Scripted controls: **12/12 normal tasks and 4/4 expected missing-key failures handled**.
  These are engineering mechanics, not learned model capability.
- Bounded execution and rejection without unintended file/process/network/environment-
  write events in the scoped audit. This is not a general sandbox or safety guarantee.
- Actual CUDA model generation for retained tiny base, SFT and DPO artifacts:
  **1,024 generated tokens per model across 16 cases**. Each model produced **0/16
  valid JSON**, zero proposed/executed calls, **0/12 normal tasks and 0/4 learned
  failure-handling successes**. No learned model produced a valid tool call.
- CUDA cache/inference integration and synthetic 512-position mechanics under their
  existing criteria; packaging, dependencies, CLI and privacy checks.
- **213 tracked files unchanged**, clean working tree, **2,543 prior regular report/
  evidence files preserved**. Both reserved payloads unchanged and integrity-hashed
  as opaque bytes only. Nothing committed or pushed from Windows.

### Fixed compact prompt: a different bounded exercise

The retained tiny models have **256-token capacity**. With their tokenizer, the
original Phase 11 system prompt plus development cases exceeded capacity, causing
correct `context_limit` rejection before generation. The external exercise used
**one fixed compact prompt** to permit actual bounded CUDA generation.

The owner reports that this changed only the prompt: tool registry, cases, expected
calls/answers/errors, scoring, chat history, turn/call limits, decoding settings,
output budget and model/tokenizer identities were unchanged within that exercise.
It **does not reproduce the full Mac prompting experiment**. Full Mac learned
artifacts and tokenizer were unavailable and untested. The tiny base is a random
fixture base, not the learned English base. Do not compare equal 0/16 outcomes as
proof of cross-device equality or identical experiments.

The supplied summary includes no exact compact-prompt text/hash, raw Windows logs,
artifact hashes, full individual skip breakdown, new driver/runtime details or
numerical error maxima. These are not invented or borrowed from Linux evidence.
Windows OS-level Ctrl-C remains unvalidated. No general CUDA determinism, useful
instruction/tool following, safety alignment or production-performance claim follows.

## Hosted Linux CPU — separately inspected PASS

Directly inspected exact-implementation [run 35862021715](https://github.com/sebastienlato/LatoS/actions/runs/35862021715),
[job 107184285661](https://github.com/sebastienlato/LatoS/actions/runs/35862021715/job/107184285661),
run/job metadata and logs, exact-commit workflow and relevant test/fixture bodies.
Push run at the implementation above; job completed 2026-09-23 12:42:16 UTC.
Ubuntu 24.04.5 hosted x86-64, Python 3.14.7 / PyTorch 2.14.0+cpu.
**295 passed / 10 MPS-only skips / zero failures**, 305 collected, 36.79 seconds.
Phase 11 contributes **50 CPU passes and one MPS-only skip**.

Actual Phase 11 scope: bounded JSON/Unicode/size/depth/duplicate/nonfinite rejection;
strict schemas and pure execution; scripted multi-turn feedback, retry/turn/call
limits, cancellation callbacks, incomplete-generation rejection, CLI schema/execute
exit codes, metric false-positive checks, and saved-evidence recount/corruption
fixtures. The tiny random CPU responder test compares eight-token generation against
the shared formatter/generator and preserves tensors, RNG and gradients. Its model
has 512-position capacity; separate existing synthetic cache tests exercise positions
through 512. This is synthetic CPU mechanics, not learned 512-token language quality.

The verifier unit test populates files named base/SFT/DPO with **scripted fixture
data**; those labels are not evidence of evaluating the three retained models.
The workflow does not run the standalone Phase 11 full-model experiment/verifier,
the Windows compact-prompt exercise, CUDA/MPS, a fresh installed-wheel suite, a
Windows-style side-effect audit or a full learned-artifact preservation audit.
No Linux model-quality comparison is inferred from its passing tests.

Other passing steps: locked install, lint/format, CPU diagnostics/debug model,
offline tiny data and tokenizer workflows, source/wheel builds and existing runtime
tests. A separate dense fixture acceptance runs 400 updates / 131,600 target exposures
and checks exact CPU recovery from update 97 over 303 updates, including weights,
optimizer, sampler and non-timing metrics. Validation state remains unchanged.
This is **dense fixture recovery**, not adapter/DPO optimizer resume or tool learning.
Fixture data audit is distinct from reading the two reserved full-project payloads.
**Physical Linux remains deferred, not performed.** Raw log/metadata and workflow
hashes are recorded in [closure.json](closure.json); raw copies remain local.

## Original Mac result and previous limitations are unchanged

The full Mac experiment used the original English models/tokenizer, original prompt,
256-token total context and 64-token output budget. Base/SFT/DPO each produced **0/16
valid JSON**, zero proposed/executed calls, **0/12 tasks and 0/4 failures handled**.
Their token counts remain **595 / 48 / 51**, distinct from the Windows 1,024 per
model. All emitted one invalid reply; base had twelve retry-context failures and
four incomplete generations, SFT/DPO sixteen retry-context failures each. No second
learned turn or learned tool use was demonstrated. Scripted controls passed 12/12
and 4/4. Mac development and isolated wheel each passed 305 tests. Preserve the
original [report](REPORT.md), samples, fixed plan, source snapshots and both attempts.

Retain the full negative Mac DPO result: 100 updates / 400 pairs / 2,498 targets,
ranking 16/32 → 15/32, exact 0/32 → 0/32; English 5.653422 → 5.603780 improves over
SFT but remains worse than original base 4.731898. The distinct Phase 10 tiny Windows
four-update/16-pair/169-target result, unchanged 16/32 ranking, and separately scoped
hosted Linux evidence remain in [Phase 10 closure](../phase-10/CLOSURE.md). Negative
SFT/LoRA/control findings remain unchanged.

Preserve all models, random baselines, tokenizer, chat contract 1, original evidence
and failures, and both reserved tests. Merge atol=rtol=1e-4 remains amended with the
original CPU 1e-5 failure retained; cache bounds CPU/CUDA 1e-5, MPS 1e-4. Adapter and
DPO optimizer resume remain unsupported; original dense recovery is version-bound.
No useful assistant, factual reliability, safety alignment, general accelerator
determinism, cross-device equality, production performance, mixed precision,
distributed serving or learned language quality beyond 256 is established.

## Closure checks and publication

Documentation/evidence only: package version, runtime, tests, configs, lock, workflow
and original experiments are unchanged. **59,999 distinct local files rehashed
unchanged**, comprising the previous 59,873-file snapshot plus 126 distinct Phase 11
run/inventory/log/original evidence files. Both reserved tests were opaque-hashed
only, never parsed or evaluated. Windows preservation counts remain separately
owner-reported. JSON, links, source identity, evidence counts, complete staged diff
and privacy/history checks passed; separate [review](CLOSURE_REVIEW.md) resolved scope
and handoff wording. No new runtime suite, training, benchmark or build was run for
this documentation-only closure.

Propose the reviewed closure commit to existing `https://github.com/sebastienlato/LatoS.git`,
main only, after explicit approval. No new tag, release, assets or visibility change.
After publication verify exact main and all unchanged tags, then stop. Phase 12 has
not begun; its handoff must only be used after closure publication and verification.
