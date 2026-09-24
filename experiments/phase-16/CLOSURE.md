# Phase 16 completed analysis — publication pending

The owner returned the bounded Windows evidence on 2026-09-24 and authorized
Mac-side analysis, configuration selection and a reviewed local publication
checkpoint. This completes Phase 16's engineering/configuration gate, not a
learned-model quality gate. Phase 17 remains explicitly prohibited until separately
authorized, including after any Phase 16 publication approval.

## Evidence accepted and independently checked

Measured source: `3cb6a47743e698a303deac0ead2664c0c89cf5e5`.
Returned archive SHA-256:
`4ff7a907d6bf416e96ca0024d709c55a729d5164990d7151f368350c9e3e6f76`.
The 3,301,537-byte archive remains ignored locally. It contains an evidence subset,
not all checkpoint bytes. No administrative script from the return was executed
on Mac; the [original verifier](verify_return.py) independently checks its contents.

- All 650 manifest-listed return files rehash correctly; no extras/missing files.
  All 324 source files named by the transfer manifest bind to the exact Git commit.
  Eight imported-package snapshots, each with 60 Python files, match that source.
- Nested inventories agree. The only 38 omitted probe files are model/optimizer
  tensors, declared with original sizes/hashes and retained on Windows. Mac did
  not rehash absent tensors, replay CUDA, recompute logits or independently rescore
  the saved models. Assertions in the verified executed code, successful exits,
  update records and checkpoint metadata support the scoped recovery evidence.
- The actual RTX 4070 SUPER reports native BF16, compute capability 8.9, driver
  616.92, Python 3.14.7 and Torch 2.14.0+cu130. Both precision modes obey the fixed
  TF32/thread/runtime settings. CUDA tensor/backward and doctor logs pass.
- JUnit XML independently totals **388 passed / 15 skipped / zero failed** for
  development and final isolated non-editable wheel invocations. The BF16 test
  executed in both. Skips are 13 MPS cases, one POSIX SIGINT case and one unavailable
  symlink-permission case. Package/import logs bind the wheel to site-packages.
- Three build/test invocation failures were retained: missing Hatchling in the
  selected build interpreter; unsuitable test import mode; wrong working directory
  lacking uv.lock. Final corrected invocations pass without source/lock changes.
  These were not model retries. One suite command ran exactly seven prescribed
  jobs: control plus six candidates, all exit zero, no candidate timeout/retry.
- Native BF16 control: loss absolute difference 0.000117302 <=0.03; gradient
  relative L2 0.00496642 <=0.05; four-update weight relative L2 0.000460638 <=0.03.
  Accumulation, validation-state preservation, same-backend recovery and nonfinite
  failure handling passed. Original synthetic mechanics remain separate evidence.
- Actual Mac reconstruction of both selected datasets matches Windows document
  accounting and identities. A fresh seed-160 sampler reconstructs every update's
  targets and cumulative exposure: 41,454 float32 / 322,035 BF16 targets. Throughput
  recomputes from all timed updates after the first two; exposure includes warmup.
  Inference medians, checkpoint byte counts and memory/headroom gates check out.

A first local verification helper incorrectly treated worker `.log` files as source
snapshot directories; filtering directories fixed it. A subsequent strict equality
check found one float64 ULP between Windows/Mac cosine learning-rate calculations
at BF16 steps 23/45. The source/configuration match; the verification records this
host-math difference and allows one ULP for that analytic recomputation only.
No prespecified CUDA numerical, quality or selection threshold was changed.

The complete compact numerical record is [windows-verification.json](windows-verification.json).
Original Mac [checkpoint validation](validation.json), negative CLI attempt,
[report](REPORT.md) and [review](REVIEW.md) remain historical records.

## Selection and bounded next-phase budget

All three scales are feasible and inside the frozen 2% short-loss band. The frozen
rule therefore selects **8 layers / 34,087,424 parameters**, native BF16, context
512, batch two × accumulation eight. The larger 53.36M and 72.63M models are feasible,
not disqualified for OOM or asserted inferior in learned quality. We retain their
results. The smaller model costs less and the tiny exposure supplies no evidence
requiring the extra parameters. The 50–80M exploration was actually performed.

[Selection](selection.json) and the [training contract](../../docs/TRAINING_2.md)
define the exact model, data, precision, inference targets and bounds. Full accepted
train/development layout accounting was run without model initialization or training:
**37,811,418** one-pass targets, **119,748** windows, **7,485** updates with a final
partial accumulation. All accepted pretraining documents fit; no exclusions or
reserved reads. All 60 runtime Python files remain byte-identical to the
CUDA-measured source; documentation and wheel metadata have been updated.

The selected measured update throughput projects 21.74 minutes of updates and
23.90 minutes with periodic full-development/checkpoint costs. A twofold margin
plus preprocessing gives 52.79 minutes. Allocate two training/recovery hours and
one paired evaluation hour; planned storage <=6 GiB and hard new-artifact limit
20 GiB. One seed, one attempt, one full pass; at most two same-attempt external-
interruption resumptions. These are honest projections and prospective limits,
not execution or quality claims. Full-data reader/tail conservation must be checked
before any future optimization; no Phase 17 run was implemented or launched here.

Keep fixed Phase 14 quality/regression/final-access gates. Historical 1.872944 BPB,
ARC-Easy 157/570, Challenge 66/299 and all five models' 0/96 instruction outcomes
remain unchanged. Nothing here establishes that the selected model will pass them.

## Review and publication boundary

A distinct final review checked source/protocol identity, evidence subset limits,
selection counterexamples, full-corpus target arithmetic, tail handling, runtime
projection, storage and the Phase 17 stop gate. The final validation record is
[closure-validation.json](closure-validation.json). No new CUDA run is claimed.
No runtime dependency, trained weight, reserved suite, historical evidence, tag,
release or asset was changed. Package remains 1.3.0. Physical Linux is deferred.

Prepare the reviewed local commit, then request explicit publication to the existing
origin/main only. No tag, release or assets. After any approved exact publication,
verify the remote and stop: **Phase 17 requires separate explicit authorization**.
