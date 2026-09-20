# Phase 7 validation plan

Declared 2026-09-20 before numerical acceptance/latency runs. Engineering inference
checks only; no training, quality selection, test-set evaluation or paid service.
The initial manual CLI smoke used one authored prompt, without scoring quality.

- Preserve the Phase 5 base, Phase 6 final experimental SFT, tokenizer, shared chat
  contract and all prior artifacts. Hash both reserved payloads only for integrity;
  do not parse or infer on them. Verify 49 Phase 6 inventory entries.
- Same-backend float32 cache agreement: CPU atol=1e-5, rtol=1e-5; MPS
  atol=1e-4, rtol=1e-4. Compare all vocabulary logits for full, single-token and
  chunked prefixes, including final capacity position 512 and changed future IDs.
  Require identical greedy IDs on the fixed cases, without a general equality claim.
- Test EOS, zero/exhausted budgets, invalid settings/IDs, reset, cancellation,
  failed appends, Unicode boundaries/invalid bytes, complete decoding equality,
  history rollback/overflow, and the installed CLI via real subprocesses.
- CPU and MPS on the actual Mac only. No Phase 7 CUDA, physical Linux or hosted
  CI result can be claimed from Phase 6 evidence.
- Benchmark both preserved models, cached and uncached, one CPU thread and MPS,
  greedy temperature 0, seed 73, budget 32. Three original fixed prompts: a short
  question, a longer prose request, and a two-turn conversation. No dataset access.
  Warm each case once; retain three measured repetitions, including EOS-short runs.
- Also measure 32 fixed forced continuations after a synthetic 128-ID prefix so
  early EOS cannot hide decode cost. These are numerical workloads, not language
  samples. Synchronize devices at timing boundaries. Report forward-only prefill,
  first-token time including sampling/text decoding, subsequent-token latency and
  throughput, logical KV bytes, process-lifetime peak RSS, and MPS boundary memory.
  No speedup threshold; record slower paths honestly. Exclude load time and terminal
  consumer time from per-token timings. Do not add unified-memory counters together.
- Run the full regression suite, lint/format, build, fresh non-editable wheel suite,
  separate review/fixes and artifact rehashing before the local commit. Preserve logs
  in outputs/phase-7-validation; publish compact reports only after owner approval.
