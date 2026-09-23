# Phase 12 fixed protocol — inference context budget

Fixed before model evaluation on 2026-09-23, following verified closure
`e2d61100a7c0c74d759bdc4ded5ddcda89cb112b` and all eleven unchanged tags.

One extension: increase the **inference session budget** from 256 to 512 tokens,
within the existing model configurations' 512-position capacity. This is not new
training, a model architecture change, RoPE scaling, or a claim of longer-context
language quality. Phase 11's full Mac evaluation could not emit a second reply:
44 sessions hit the retry context limit and four stopped with incomplete generation.
This observed confound justifies testing additional history capacity before training.

Fixed baseline: the exact original base, SFT and DPO weights, English tokenizer,
original system prompt, 16 synthetic development cases and literal scoring from
Phase 11. Identities come from its tracked plan and inventory. Baseline 256 must
reproduce saved sessions/scores/metrics/token counts before the 512 condition runs;
otherwise stop and retain the discrepancy. Do not retune to match a result.

Hypotheses, fixed before running:

- H1 (mechanics): at least one previously context-limited session emits a second
  reply at 512. Count paired transitions, attempts, prompt lengths and generated
  tokens. More turns alone are not tool use.
- H2 (capability): added history room alone suffices for at least one successful
  normal task across the three models (baseline 0/36). Report rejection if still
  zero. Parsing, envelopes, arguments, calls, execution and expected failure
  handling remain separate; no claim based solely on additional tokens or turns.

Each model runs 256 then 512, greedy cached float32 generation, seed 0, one CPU
thread, original chat contract 1, 64 new tokens per turn, four turns/two calls
maximum. Primary backend: existing Mac MPS. Tiny engineering tests cover CPU/MPS;
no Phase 12 CUDA or Linux result is inferred. No prompt rewrite, repair, forced
JSON, truncation, data acquisition, checkpoint selection or parameter sweep.
Development cases are reused and already observed; not an untouched benchmark.
Both reserved tests are opaque-hashed only. Scripted controls validate mechanics.

Budget: existing local machine and installed locked environment only; $0 new
services; no training; at most 96 learned sessions, 384 generations, 24,576 generated
tokens per complete run. Ten minutes for model evaluation, checked before every
attempt (one active kernel cannot be preempted). Stop on RSS above 8 GiB at attempt
boundaries; record MPS boundary allocation separately, not as continuous peak.
At most two complete runs, second only for a review fix/reproducibility check;
the two-run bound is 49,152 generated tokens. New evidence under 2 GiB.
Preserve failed attempts. Validation and preservation hashing are outside evaluation time.

Implement an experiment-local runner and saved-evidence verifier, capture exact
source/plan/config hashes and all raw sessions/attempts, rehash prior artifacts
before and after, test budget/overflow and tampered evidence, run the full suite
and fresh installed-wheel suite, separately review/fix, and prepare the exact
local commit. Public runtime/defaults/version remain 1.3.0. No dependency change.
No general accelerator determinism, cross-device equality, production performance,
learned long-context quality, useful assistant or safety capability follows.
Preserve the distinct negative original Mac and tiny compact-prompt Windows
results, separately scoped hosted Linux evidence, deferred physical Linux and
all prior limits. Stop for explicit Phase 12 publication approval, main only;
no tag, release, assets, remote backup or visibility change.
