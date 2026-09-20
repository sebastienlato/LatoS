# Roadmap

These are planned capabilities, not claims about the current package. Each phase
requires working deliverables, recorded checks, a separate review and fixes,
updated state, and a local commit before its publication approval. After an
approved push is verified, development proceeds to the next phase unless the owner
has imposed a pause or transition gate, as recorded in PROJECT_STATE.md.

Phase 7 implementation is published as v0.8.0 at
`04031e5ea98da8db495242165a78c216ab1d4cf4`. The owner reports Windows/RTX 4070 SUPER
PASS using tiny 256-position base/SFT artifacts and a separate synthetic 512-position
model, not the unavailable full Mac learned artifacts. Windows console Ctrl-C is
unvalidated. Hosted Linux CPU CI passed separately within its tiny CPU/workflow
scope; physical Linux remains deferred. All negative quality and broader capability
limits remain. [Phase 7 closure](experiments/phase-7/CLOSURE.md) is ready locally and
requires its own publication approval. **Do not begin Phase 8 in this chat.** After
closure publication is verified, Phase 8 requires explicit owner start in a fresh
Work chat. See [the handoff](docs/PHASE8_HANDOFF.md) and [state](PROJECT_STATE.md).

| Phase | Deliverable | Acceptance evidence |
| --- | --- | --- |
| 0: Foundation | Installable package, CLI doctor, environment report, CPU CI | Clean install, imports, CLI behavior, backend smoke checks, locked versions, reviewed packaging |
| 1: English data | Source manifest, acquisition, cleaning, deduplication, document splits, tiny fixture | Terms and provenance, reproducible counts/hashes, exact and near-duplicate checks across splits |
| 2: Tokenizer | Independently trained byte-level BPE with explicit special tokens | Train-only fitting, Unicode round trips under declared normalization, save/load, compression and vocabulary/hash report |
| 3: Transformer | Original dense PyTorch decoder with configuration and sampling | Causality, shapes, finite gradients, next-token loss, parameter count, save/load, attention reference comparison |
| 4: Training | Optimization, scheduling, accumulation, validation, resumable checkpoints | Tiny-fixture overfit, resumed vs. uninterrupted comparison on a declared backend, validation leaves weights unchanged |
| 5: Pretraining | Measured English pilot from random initialization; larger run only if feasible | Held-out baseline comparison, token exposure, fixed samples, runtime, throughput, memory, failures |
| 6: Instruction tuning | Documented English conversations, shared chat format, SFT | Assistant-only masking and next-token alignment, held-out instruction checks, comparison with base, regressions |
| 7: Inference | CLI chat, streaming local interface, KV cache | Cached/uncached agreement within stated tolerance, stopping, multi-turn context limits, latency |
| 8: Release | Reproduction guide, data/model cards, experiment report, permitted artifacts | Fresh-checkout reproduction, artifact hashes, explicit platform support and measured capability limits |
| 9: Adaptation | LoRA training and merge | Frozen-base checks, adapter round trips, merge agreement, full-tuning comparison |
| 10: Preferences | DPO experiment with a fixed baseline | Objective checks, documented data, held-out SFT comparison and regressions |
| 11: Tools | Structured calls to bounded tools | Parsing, argument correctness, task success and failure behavior measured separately |
| 12+: Experiments | One justified research extension at a time | Fixed baseline, hypothesis, available compute budget, reproducible positive or negative result |

Phases 0–7 propose tags v0.1.0–v0.8.0; Phase 8 proposes v1.0.0. Tags are created
only when included in that phase's explicit publication approval.

## Engineering and evidence

Start with CPU correctness and float32. Validate accelerator operations on actual
hardware before relying on them. A tiny debug configuration comes first; select
pilot and release sizes from measured memory and throughput, not an assumed budget.
No new paid services are authorized.

The initial model direction is a dense causal decoder with RMSNorm, rotary
positions, multi-head attention, and SwiGLU. Final details belong to Phase 3 and
must be derived from primary research with attribution. Mixed precision, CUDA,
multi-GPU operation, export adapters, and longer context each need separate evidence.

Select data independently. Record source revisions, terms, language, acquisition,
and hashes. Split by document identity before chunking; fit the tokenizer only on
training data. Reserve the test set from routine tuning. Track run configuration,
seed, source commit, environment, artifact identities, token exposure, selection
rules, samples, resource measurements, and failures. Perplexities are comparable
only with compatible evaluation data and tokenization.

Evaluation begins as soon as a model is trainable. A working pipeline does not
establish useful language quality. Keep failed experiments and revise plans openly.
Large datasets, weights, and logs stay outside Git with durable artifact references.
