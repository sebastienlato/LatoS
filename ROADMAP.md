# Roadmap

These are planned capabilities, not claims about the current package. Each phase
requires working deliverables, recorded checks, a separate review and fixes,
updated state, and a local commit before its publication approval. After an
approved push is verified, development proceeds to the next phase unless the owner
has imposed a pause or transition gate, as recorded in PROJECT_STATE.md.

Phase 9 implementation and scoped closure are published. Closure commit
`7d19649cabcc0698bd77042779772bc2196c4bc2` and all eleven unchanged remote tags were
verified before the owner's explicit fresh-chat Phase 10 start. Verified publication
records supersede the historical tracked pending snapshots. See
[Phase 9 closure](experiments/phase-9/CLOSURE.md) for bounded Windows/CUDA, separately
scoped hosted Linux evidence and deferred physical Linux.

Phase 10 closure is published at `87d744fb8cfd4228c78f35bbd220d2833462e037`.
Exact remote main and all eleven unchanged tags were verified before the explicit
fresh-chat Phase 11 start. Verified publication supersedes historical pending snapshots.
[Scoped closure](experiments/phase-10/CLOSURE.md) retains the negative full Mac DPO
result, distinct tiny Windows/CUDA PASS, separately inspected hosted Linux CPU scope
and deferred physical Linux. Annotated v1.0.0 remains unchanged.

Phase 11 implementation is published at `41fc3cffe42a1ae7a42446c5785abfffc1f559db`,
package 1.3.0, without a Phase 11 tag. [Scoped closure](experiments/phase-11/CLOSURE.md)
records owner-reported bounded Windows/CUDA PASS with a fixed compact prompt and
separately inspected hosted Linux CPU evidence. This Windows exercise is not the
full Mac prompting experiment; both retain 0/16 JSON per model and no learned tool
use. Scripted successes are mechanics only. Physical Linux remains deferred.
Phase 11 closure is published at `e2d61100a7c0c74d759bdc4ded5ddcda89cb112b`.
Exact main and all eleven tags were verified before the explicit fresh-chat Phase 12 start.
Verified publication supersedes the historical pending snapshot.

Phase 12's [fixed inference context-budget experiment](experiments/phase-12/REPORT.md)
is published at `5b72872a2032f8999aae9c28ff0ab5c704e3d893`. Increasing only the session
budget from 256 to 512 allowed all 44 previously context-limited sessions to retry,
but no model produced valid JSON or a successful tool task. Original weights,
prompt, tokenizer, chat contract, cases and defaults are unchanged. This tests
available history space within existing capacity, not general learned long-context
quality. The path-portability correction is published at
`b222c1fa844289247531b922d5cb18134e99c7b1`; no Phase 12 tag. Corrected Windows/CUDA
PASS is owner-reported within a distinct random synthetic fixture scope: 16 initial
blocks removed, not the Mac's 44 learned retry transitions. Scripted controls remain
separate. [Closure](experiments/phase-12/CLOSURE.md) also records inspected hosted
Linux CPU scope; physical Linux remains deferred. Original failure/evidence preserved.

Phase 12 is formally closed locally within that scope; the reviewed closure commit
awaits explicit publication approval. **No next experiment in this chat.** After
approved closure publication verify exact main and all unchanged tags, then stop.
The [Phase 13 fresh-chat handoff](docs/PHASE13_HANDOFF.md) is preparation only, with
no extension selected. Verified publication records supersede historical pending
snapshots. See [current state](PROJECT_STATE.md).

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
