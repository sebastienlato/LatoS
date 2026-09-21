# Phase 9 fixed protocol

Declared before the learned-model experiment on 2026-09-21. Starting publication:
remote main and annotated v1.0.0 resolve to
`10d9ef7bf0618b364f887ff9d279408e3cbc33c9`; tag object
`1acb6b94f0adfbca85014ad6601dbfc716e3c651`. The verified Phase 8 publication record
supersedes that commit's pending-publication wording. Existing Mac CPU/MPS only;
no new paid service, external model, tokenizer, data or dependency.

## Question and controls

Can attention LoRA reduce trainable/optimizer storage while retaining a functioning
assistant-only adaptation/merge pipeline? Quality is measured, not an acceptance
threshold. Compare fresh full tuning against LoRA, both starting from the same
preserved Phase 5 final base, never from the negative Phase 6 SFT artifact.

- Base weights SHA-256: `f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa`.
- Original frozen tokenizer SHA-256:
  `7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
- Same original Phase 6 conversations, 192 train / 32 validation, shared chat
  contract 1 and assistant-only targets including EOS. Both reserved tests stay
  unopened by the experiment; preservation checks hash them as opaque bytes only.
- Same fresh AdamW, seed 61, 200 updates, batch 4, accumulation 2, context 256,
  peak learning rate 0.0003, warmup 10, cosine to 0.1 times peak, weight decay 0.01,
  beta1=0.9, beta2=0.95, epsilon=1e-8, norm clip 1. See the committed
  [configuration](../../configs/training/adaptation-pilot.json).
- LoRA rank 8, alpha 16, initialization seed 91; Gaussian A standard deviation
  0.02, zero B; fused QKV and attention output in all eight blocks. Shared QKV A;
  no dropout, embedding, feed-forward, norm or bias adaptation. Float32 only.
- Final update 200 selected in advance. One run per method, no tuning/search or
  early stopping. This is an equal-data/update/schedule comparison, not equal
  wall time, an optimized learning-rate comparison or general method ranking.
- Separate sequential processes on MPS, one CPU thread. Compare identical starting
  metrics, dataset identities, per-update sampler hashes and target exposures.
  Preserve both summaries and every generated sample, including failures.

## Acceptance and measurements

Tiny CPU numerical/gradient tests precede full runs; CPU and MPS check frozen base,
zero initialization, adapter save/load, merge and KV inference. Reject wrong bases,
wrong tokenizers, corrupt tensors, invalid ranks and attempted overwrites.

Measure all 32 validation exact replies and EOS stops, full assistant loss and
English validation loss on the unchanged 111,523-target regression set. Greedy
32-token generations, seed 71; recall uses gold preceding replies. No language
quality claim beyond 256 positions. Original SFT stays negative and preserved.

Require bitwise unchanged frozen base tensors and absent base gradients/optimizer
slots; saved adapter round-trip logits exact on the same backend. Compare all
vocabulary logits on all instruction validation windows after merge with
atol=rtol=1e-5 CPU / 1e-4 MPS. Also exercise cached inference through model capacity
512 with fixed synthetic IDs, separately from language quality. Save a new merged
model; never mutate or overwrite the original model or adapter.

Record trainable parameters, optimizer tensor bytes, adapter/full file bytes,
synchronized update seconds and target throughput. RSS is each process's lifetime
peak; MPS memory is boundary snapshots, not continuous peak. Do not add RSS and MPS
or infer production performance from a single unreplicated run. Adapter snapshots
have no optimizer recovery; a failure keeps evidence and requires a new run directory.

Finish full tests, locked lint/build, installed-wheel smoke, preservation hashing,
a separate review/fixes, and a local commit. Stop for Phase 9 publication approval.
No release assets or tag are required; proposed publication is the reviewed source
commit to existing main only. Do not begin Phase 10 in this request.

## Numerical validation amendment after independent review

The original CPU merge bound (atol=rtol=1e-5) failed on the saved full learned
adapter, despite the tiny CPU fixture passing: the first failing window had 66 of
376,832 values outside the combined bound, with worst failing absolute difference
1.40816e-5. Preserve that failure. MPS validation passed its declared 1e-4 bound.
The CPU cache bound is not a suitable universal merge bound: low-rank and merged
projections use different floating-point multiplication/reduction orders.

Use atol=rtol=1e-4 for **full float32 merge** on CPU and MPS, report maximum errors,
and independently check a float64 unmerged/merged reference on CPU. Retain CPU
cache atol=rtol=1e-5, MPS cache 1e-4 and exact same-backend adapter reload. This is a
post-observation acceptance amendment, not a claim the initial CPU threshold passed.
It changes no training setting, checkpoint selection, loss result or historical
inference tolerance. The original protocol hash is preserved in the run evidence.

The first comparison overlapped a validation suite; retain it as preliminary and
exclude its timings from the performance comparison. The accepted repeat ran with
validation jobs idle. The fixed hyperparameters and final-update rule were unchanged.
