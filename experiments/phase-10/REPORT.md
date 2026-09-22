# Phase 10 — bounded DPO with a fixed SFT reference

Completed locally on 2026-09-22; package 1.2.0, publication pending explicit approval.
Remote main was verified at Phase 9 closure
`7d19649cabcc0698bd77042779772bc2196c4bc2`, with all eleven tag objects and peeled
commits unchanged, before development. v1.0.0 remains at
`10d9ef7bf0618b364f887ff9d279408e3cbc33c9`. Verified publication records supersede
historical pending snapshots. This fresh-chat start was explicitly authorized.

## Fixed experiment

The [protocol](PLAN.md) fixes 100 updates of full-parameter float32 DPO, beta 0.1,
constant learning rate 1e-5, fresh AdamW, zero weight decay, clip norm 1, batch four
pairs, seed-101 epoch shuffle and maximum length 256. Final update 100 is selected
in advance; no validation selection or hyperparameter search. Only the original
Phase 6 SFT model initializes both policy and separate frozen reference. Base/SFT,
LoRA/merged/full control, tokenizer and all historical attempts remain unchanged.

SFT weight SHA-256: `62b890214e5544292cc54b725a4e505730fca0d501b3337cdd3f051a622d09ae`.
Tokenizer SHA-256: `7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
17,308,032 policy parameters; same number in the frozen reference. Same chat contract 1.
No new dependency versions, data acquisition, paid service or remote write.

[Documented original preferences](DATA.md) comprise 192 train / 32 validation pairs.
Preferred exact answer versus a different same-family answer from the same split;
no human feedback or external model labels. Prompt/earlier replies remain identical.
DPO scores only final assistant content plus EOS. Assistant regression preserves
SFT's original all-assistant-turn metric. Both reserved tests remain unused.
Validation template overlap and training-prompt reuse are explicit limitations.

Accepted run: `outputs/phase-10-dpo-reviewed`, Mac M4 Max / 64 GiB, MPS, one CPU
thread, Python 3.14.7 / PyTorch 2.14.0. The fixed baseline is re-evaluated on the same
backend. The [results](results.json) pin runtime, dirty source parent plus exact
implementation/source snapshot, lock, configuration, datasets and model identity.
Dirty means the local phase implementation was not yet committed; exact source
files were captured before evaluation. No published baseline state was changed.

## Results, including the negative outcome

| Validation measure | Preserved SFT | DPO update 100 |
| --- | ---: | ---: |
| Preference loss, lower is better | 0.693147 | 0.718657 |
| Raw chosen likelihood wins | 16/32 | 15/32 |
| Reference-relative margin wins | 0/32 (32 ties) | 9/32 (0 ties) |
| Mean beta-scaled relative margin | 0 | -0.044940 |
| Exact generated answers | 0/32 | 0/32 |
| Replies ending in EOS | 32/32 | 32/32 |
| Assistant loss, 121 targets | 5.815233 | 5.791526 |
| English loss, 111,523 targets | 5.653422 | 5.603780 |
| English perplexity | 285.266 | 271.451 |

**Held-out preference quality worsened; exact instruction following did not improve.**
Training preference loss fell from 0.693147 to 0.480671 and raw train ranking stayed
192/192, but this did not transfer to held-out words. Chosen train log probability
fell from -0.125233 to -0.701896 while rejected fell further (-8.966028 to -14.520039).
Reducing preference loss does not require increasing absolute chosen probability.
On validation, both rose slightly, but rejected rose more; per-pair values are retained.

English loss slightly recovered relative to the poor SFT baseline, yet remains
well above the preserved Phase 5 base's 4.731898. This does not overturn the original
negative SFT result or establish a better general language model. All 64 baseline/DPO
[samples](samples.json), including failures, are retained. Greedy generation is capped
at 32 tokens; recall uses gold earlier replies. The 32 preference prompts are also
the 32 SFT validation prompts, not an independent second benchmark or fresh test.
No useful assistant, alignment, factual reliability or safety improvement is established.

## Accounting and numerical checks

100 updates processed 400 pair exposures / 800 response sequences / 2,498 final-
response target exposures, including EOS on both branches. All 192 distinct pairs
and 1,200 distinct branch target positions were visited. DPO response targets are
not directly comparable to SFT's all-assistant target budget. [Independent replay](verification.json)
reconstructs shuffle indices, epochs and every update's counts, verifies 47 saved
inventory files, and checks all 32 validation pairs with token-by-token scalar
log-sum-exp after float32 MPS inference. Maximum response-score discrepancy is
3.864105e-6, below predeclared 1e-4; scalar objective agrees within 1e-6. This scalar
CPU diagnostic is not mixed-precision training or general cross-device equality.

The policy starts exactly equal to reference. Reference weights remain bitwise
unchanged, with no gradients or optimizer membership, while policy weights change.
Saved-model reload reproduces every SFT validation logit exactly on the same MPS
backend. Tiny CPU repeatability is tested separately; general accelerator determinism
or historical optimizer recovery is not claimed. Model-only DPO output has no resume.

Synchronized updates took 5.831 seconds (68.60 pair exposures/second); total before
inventory was 14.876 seconds. One-run timing, not a stable speedup or production
claim. Observed training RSS high-water mark: 921,485,312 bytes. MPS allocation:
285,515,008 bytes; driver allocation: 2,277,195,776 bytes. MPS values are boundary
snapshots, not continuous peaks or additive pools. Tests were idle during training.

## Review, attempts and retained boundaries

The separate [review](REVIEW.md) fixed protocol/runner inventory capture and the
independent checker's unsupported MPS float64 conversion. The first learned run,
`outputs/phase-10-dpo`, and the original checker TypeError log remain preserved.
The accepted repeat uses identical optimization and selection settings after the
metadata fix, not improved-result selection. Both runs show the same negative
outcome; small MPS differences are not reclassified as exact recovery.

Development and installed-wheel checks, packaging, privacy and preservation are
recorded in [validation.json](validation.json). New mathematical checks independently
verify loss and gradients, shift/sum/masking/padding/EOS, baseline ties and invalid
inputs. Tiny integrations run with both test payloads removed and test failure
retention, CLI, frozen reference, CPU repetition and MPS round trip. All earlier
runtime tests remain in the suite. Learned artifacts/logs stay local and ignored;
[artifact hashes](artifacts.json) are identities, not remote backups or uploads.

The [Phase 9 closure](../phase-9/CLOSURE.md) remains authoritative for historical
platform evidence: owner-reported tiny Windows RTX 4070 SUPER (226 passed / 10 skips;
4 updates / 175 targets per method, 7,856 base / 192 adapter parameters), separate
hosted Linux CPU (228 passed / 8 MPS skips), no full Mac experiment/512-position
learned merge reproduction on Windows, and physical Linux deferred. No Phase 10
Windows/CUDA or Linux execution is claimed. Windows console Ctrl-C remains unvalidated.
Tiny adapter ratio ~2.44399% is distinct from the full architecture's 0.851951279%.

Retain amended float32 merge atol=rtol=1e-4 and the original CPU 1e-5 failure;
cache tolerances stay CPU/CUDA 1e-5 and MPS 1e-4. Adapter optimizer resume remains
unsupported. Original SFT (200 updates / 6,188 targets, exact 0/32, English regression)
and negative LoRA/full-control quality results remain unchanged. Useful instruction
following, language quality beyond 256-token windows, general accelerator determinism,
cross-device equality, production performance, mixed precision and distributed
serving remain unestablished. Phase 11 is not started.

Propose the reviewed local source commit to existing
`https://github.com/sebastienlato/LatoS.git`, branch `main` only. No tag, release,
asset upload, visibility change or remote backup. Stop for explicit Phase 10 approval.
