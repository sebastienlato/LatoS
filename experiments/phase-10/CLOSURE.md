# Phase 10 closure — scoped external validation

Recorded 2026-09-22. **Phase 10 is formally closed locally within its documented
scope; the reviewed closure commit awaits explicit publication approval. Do not
begin Phase 11 in this chat.** The owner's independent Windows/CUDA PASS satisfies
the external-validation wait. After approved closure publication, verify the exact
remote commit and unchanged tags, then stop. Phase 11 needs an explicit fresh-chat
start after verified closure publication; the [handoff](../../docs/PHASE11_HANDOFF.md)
is preparation only.

Validated implementation: `abed1adffba562373348477e3958bcd1d5886b11`, published and
verified on main, package 1.2.0. **There is no Phase 10 tag.** All eleven existing tags
remain fixed, including annotated v1.0.0 at
`10d9ef7bf0618b364f887ff9d279408e3cbc33c9`, tag object
`1acb6b94f0adfbca85014ad6601dbfc716e3c651`. Verified publication supersedes the original
tracked pending snapshot; original reports are retained as historical evidence.

## Windows/CUDA — owner-reported independent PASS

The owner reports a fresh exact-commit checkout and locked install, **243 tests
passed / 11 expected platform or hardware skips / zero failures**, and a complete
suite PASS from a fresh non-editable wheel. GPU: **NVIDIA RTX 4070 SUPER**. This Mac
session did not execute those Windows checks. The owner's summary supplies no raw
logs, artifact hashes, individual skip breakdown, new driver/runtime details or
numerical error maxima; none are invented here.

Reported checks passed:

- Deterministic preferences, chosen/rejected integrity, train/validation isolation,
  unchanged shared chat serialization, and final-response-only masking/alignment.
- Independent sequence log probabilities and scalar DPO objective; gradient/sign checks.
- Tensor-identical initial policy/reference with independent storage. Reference
  actually executed on CUDA and stayed frozen, unchanged, gradient-free and outside
  optimizer state. Actual policy forward/backward, finite gradients, optimizer state
  and weight updates passed on the GPU.
- **Four updates / 16 preference pair exposures / 169 final-response targets** in
  the bounded tiny-model exercise, plus validation-state preservation and exact
  same-CUDA model reload. DPO optimizer resume remains unsupported and unclaimed.
- Held-out raw chosen ranking **16/32 → 16/32**, exact replies **0/32 → 0/32**.
  English/SFT/base regression-comparison mechanisms passed within the tiny local scope.
- Reserved tests stayed unused and unchanged. **2,017 prior evidence/report files**
  and **194 tracked files** were preserved, tree clean, no Windows commit or push.
- CLI, diagnostics, lint, formatting, dependency integrity, offline workflows,
  builds, packaging and privacy checks passed.

This validates the bounded CUDA engineering path, **not the full Mac 100-update
experiment**. Full Mac learned SFT/reference and English corpus were not reproduced.
Windows did not reproduce the Mac's 16/32 → 15/32 ranking result. Its unchanged tiny
ranking does not contradict or replace the Mac outcome. No useful preference or
instruction following, DPO superiority, general CUDA determinism, cross-device
equality, safety alignment, mixed-precision/distributed behavior or DPO optimizer
recovery is established. Windows console Ctrl-C remains previously unvalidated.

## Hosted Linux CPU — separately inspected PASS

Directly inspected exact-implementation [run 35742827336](https://github.com/sebastienlato/LatoS/actions/runs/35742827336),
[job 106796704207](https://github.com/sebastienlato/LatoS/actions/runs/35742827336/job/106796704207),
metadata, complete logs, exact-commit workflow and relevant test/fixture bodies.
Push run at the implementation commit above; job completed 2026-09-22 14:49:19 UTC.
Ubuntu 24.04.5 hosted x86-64, Python 3.14.7, PyTorch 2.14.0+cpu.
**245 passed / 9 MPS-only skips / zero failures**, 254 collected, 51.23 seconds.
The preference file contributes **17 CPU passes / one MPS-only skip**.

Actual scope: scalar DPO objective/gradient and sequence-score checks; masking,
padding/EOS, source splits, invalid inputs; two-update tiny random-initialized CPU
policy/reference runs with eight pair exposures, fixed reference, weight updates,
round trip, repeated CPU results, CLI, wrong-base/overwrite/failure retention and
independent tiny exposure/oracle verification with corruption rejection. Both tiny
reserved payloads are removed before the DPO runner uses its inputs. These tests
do not use the learned Mac SFT model or full English corpus. Existing CPU inference,
LoRA, SFT and dense training tests also pass.

The workflow also passes locked install, lint/format, CPU doctor/debug model,
offline data/tokenizer fixtures and source/wheel builds. Its separate Phase 4 dense
fixture acceptance runs 400 updates / 131,600 targets and checks exact CPU recovery
from update 97 over 303 updates. **That is dense fixture recovery, not DPO or adapter
optimizer resume, and not language-quality evidence.**

The workflow does **not** run a fresh installed-wheel suite, CUDA/MPS, the standalone
full Mac DPO experiment or learned-artifact oracle, independent Windows validation,
full learned-model preservation/quality comparison, or a standalone archive/privacy
audit. Test-level tiny packaging checks and builds do not establish those broader
claims. **Physical Linux remains deferred, not performed.** Raw inspected log and
metadata hashes are in [closure.json](closure.json); raw copies stay local.

## Negative results and prior limits are unchanged

The full Mac experiment remains 100 updates / 400 pairs / 2,498 response targets.
Held-out preference loss **0.693147 → 0.718657**, raw ranking **16/32 → 15/32**,
exact replies **0/32 → 0/32**. English loss **5.653422 → 5.603780** improves relative
to SFT but remains worse than original base **4.731898**. This is not preference-
learning or instruction-following improvement. Original samples, data identities,
first attempt, checker failure, fixed selection and [report](REPORT.md) are unchanged.

Keep all base/SFT/LoRA/merged/control/DPO models, random baselines, tokenizer,
chat contract 1 and both reserved tests. Negative SFT and LoRA/control quality
results remain unchanged. Merge atol=rtol=1e-4 stays the amended contract; preserve
the original CPU 1e-5 failure. Cache bounds remain CPU/CUDA 1e-5, MPS 1e-4. Adapter
and DPO optimizer resume remain unsupported. Scalar/float64 diagnostics do not
establish mixed-precision training. Familiar synthetic templates, shared development
validation prompts, length-sensitive scoring and weak baseline limit inference.
No useful quality, safety alignment, factual reliability, general accelerator
determinism, cross-device equality, production performance, language quality beyond
256 tokens, mixed precision or distributed serving is established. Earlier scoped
Windows/hosted Linux evidence is retained in [Phase 9 closure](../phase-9/CLOSURE.md).

## Closure checks and publication

Documentation/evidence only; no runtime, package version, tests, configs, lock,
workflow or original experiment evidence changes. Separate [review](CLOSURE_REVIEW.md)
checks attribution, actual CI scope, counts, negative results and the fresh-chat gate.
**54,356 distinct local files rehashed unchanged**: the previous 54,236-file snapshot,
109 Phase 10 inventory entries and 11 original Phase 10 evidence/script files.
Windows preservation counts remain separately attributed. Reserved tests were
hashed as opaque bytes only, never parsed/evaluated. JSON, Markdown links,
whitespace, complete staged diff and privacy/history checks passed. No new runtime
tests, training, benchmark or package build was needed or run for this closure.

Propose only the exact reviewed closure commit to existing public
`https://github.com/sebastienlato/LatoS.git`, **main**. No new tag, tag movement,
release, asset upload, visibility change or paid service. Stop for explicit approval;
then publish and verify the exact state and unchanged tags, and stop again.
