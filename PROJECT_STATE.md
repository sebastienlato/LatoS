# Project state

Updated: 2026-09-24.

## Active checkpoint

**Phase 16 — Model & CUDA Training 2.0 is complete locally, reviewed and awaiting
explicit publication approval.** The owner returned Windows evidence and authorized
Mac analysis and selection. The actual CUDA external gate is satisfied within the
bounded protocol. **Phase 17 has not begun and requires separate explicit owner
authorization after approved, verified Phase 16 publication. Push approval does
not lift that gate.**

Measured implementation: `3cb6a47743e698a303deac0ead2664c0c89cf5e5`, built from
published Phase 15 `d3c63d8655415252ac4a9efac023107831e1ad75`. Mac remains authoritative;
Windows used the exact reviewed transfer without source changes. Package remains
**1.3.0**, and **v1.0.0** remains at `10d9ef7bf0618b364f887ff9d279408e3cbc33c9`.
The earlier tracked handoff/pending snapshots retain their historical meanings.

## Outcome and evidence

- [Completed analysis](experiments/phase-16/CLOSURE.md): original CUDA BF16 autocast,
  float32 weights/AdamW state, recovery, synchronized measurement and bounded
  Data 2.0 document coalescing; no training-runtime changes after Windows validation.
- [Independent return verification](experiments/phase-16/windows-verification.json):
  650 returned files verified, source manifest bound to 324 Git files, eight 60-file
  executed-source snapshots identical; all six prescribed candidate jobs and BF16
  control pass, with no candidate retries/timeouts. All per-update exposure and
  throughput aggregates reconstruct. The 38 tensor files stay on Windows; Mac
  verified their recorded identities, not absent bytes or a numerical replay.
- [Selection](experiments/phase-16/selection.json): **8 layers / 34,087,424 parameters**,
  width 512, eight heads, FFN 1,408, 16,384-token vocabulary, context 512, native
  CUDA BF16, batch two × accumulation eight. All three scales pass and lie within
  the frozen 2% early-loss band; select the smallest eligible. Larger candidates
  are feasible; these short probes do not rank eventual learned quality.
- Selected probe: **28,983.1 useful targets/s**, **1.088 GiB peak reserved VRAM**,
  2.470 GiB host peak, 5.59 ms first-token latency and 168.12 decode tokens/s in
  the fixed synthetic inference resource test. No general performance guarantee.
- [Training contract](docs/TRAINING_2.md): full accepted corpus has **37,811,418
  one-pass targets**, **119,748 windows**, **7,485 planned updates** with a two-
  microbatch final update. Fresh seed 160; one serious attempt, one full pass;
  <=2 same-attempt interruption resumptions. 23.90-minute central / 52.79-minute
  conservative training projection; two-hour training and one-hour evaluation
  limits, 6 GiB planned / 20 GiB hard new-artifact budget. These are projections.

## Validation and limits

Windows development and isolated wheel: **388 passed / 15 skips / zero failures**;
BF16 numerical/recovery test executes. The 15 skips are 13 MPS cases, POSIX SIGINT
and unavailable symlink permission. Three initial build/test invocation failures
are retained and explained; final passing commands changed no source or dependency.
The Mac implementation checkpoint had **402 passed / one CUDA skip** in both
environments. Final analysis validation is recorded in
[closure-validation.json](experiments/phase-16/closure-validation.json).

The complete accepted train/development layout was counted without creating or
training a model. Its full-data reader/cache and final partial accumulation must
pass identity/target/recovery checks before future Phase 17 optimization. CUDA full-
run duration, sustained thermals, larger contexts and final model quality remain
unmeasured. No new CI result or physical Linux execution; physical Linux deferred.
No new paid resources, dependency upgrades, reserved scoring or Phase 17 work.

Fixed Phase 14 gates and all historical negatives remain: **1.872944 BPB**,
ARC-Easy **157/570**, Challenge **66/299**, all five historical models **0/96
instructions**. One modest, encyclopedia-dominated data pass may fail those gates;
failure must be retained and block dependent work, not weaken thresholds.

## Pending publication and next action

Prepare the reviewed final local commit; exact SHA belongs in the approval request
and ignored local publication record. Existing remote:
`https://github.com/sebastienlato/LatoS.git`, destination **main** only. No proposed
tag, release, asset, remote backup, PR, visibility change or other GitHub write.
The earlier implementation commit is still local and is included in this final state.

Stop at **“Push Phase 16 to GitHub?”** for explicit approval of the exact reviewed
state. After approval, publish that state, verify remote main and unchanged eleven
tags/release/four assets, then **STOP**. Separate explicit owner authorization is
required before Phase 17. Until approval, all work stays local.

Local uv: `.private/tools/bin/uv`; runtime `.venv`. Private context, returned raw
logs and acquired/learned artifacts remain ignored. The compact Windows return
is an evidence subset, not a backup of the only tensor checkpoints.
