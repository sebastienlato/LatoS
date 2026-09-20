# Phase 7 — local streaming inference

Originally completed locally on 2026-09-20; subsequently published as v0.8.0 at
`04031e5ea98da8db495242165a78c216ab1d4cf4`. The measurements below retain their
original Mac scope. Later Windows/CUDA and Linux CPU CI evidence is recorded
separately in [CLOSURE.md](CLOSURE.md); closure publication awaits approval.
CLI chat, streaming text and inference-only KV caching are implemented and checked
against both preserved learned artifacts. This is an engineering result, not an
instruction-following improvement. [Machine-readable results](results.json),
[predeclared checks](PLAN.md), [separate review](REVIEW.md) and
[usage/reproduction](../../docs/INFERENCE.md) give the detailed scope.

## Verified start and preserved inputs

Before development, remote main matched the owner-approved Phase 6 closure
`c4ed8c9564c41f26be2c10ee178f5a10ac80d3a8`. Annotated v0.7.0 still resolved to
`f6636af34933b9678824cc8dd50f6ab6559a2de5`, with unchanged tag object
`bdbaa5499847ae7080e2b32bc96c5400ed14df80`. The owner explicitly started Phase 7
in a fresh Work chat, satisfying the transition gate.

Before/after SHA-256 checks match for **357 retained artifact/data files**, including
all **49** reviewed SFT inventory entries, base, tokenizer, random baseline, prior
training outputs/failures and both reserved test payloads. Disposable environment
files and bytecode are outside this inventory. Reserved payloads were hashed as
opaque bytes for integrity, never parsed or used for inference/quality selection.
The shared `src/latos/chat.py` is byte-for-byte unchanged from closure.

| Input | SHA-256 |
| --- | --- |
| Base update 3,000 weights | `f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa` |
| Experimental SFT update 200 weights | `62b890214e5544292cc54b725a4e505730fca0d501b3337cdd3f051a622d09ae` |
| Tokenizer | `7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab` |

No learned artifact was rewritten or trained. Regression tests do run their own
temporary tiny training fixtures. Model-only checkpoints still load unchanged;
old optimizer recovery remains tied to its recorded source/runtime.

## Delivered behavior

- Streaming terminal chat and one-shot JSON line events with explicit model/tokenizer
  selection, artifact hashes, known-SFT identification and visible quality limits.
- Per-reply KV cache, absolute rotary positions and explicit causal masks for appended
  chunks. Cache state never enters saved model parameters; reset starts at position 0.
- EOS, reply-budget and context stopping; whole-message history plus full reply-budget
  reservation. Overflow rejects the new turn without changing history. Only nonempty
  EOS-completed replies enter history; cancellation and partial output roll back.
- Incremental UTF-8 buffering with a final replacement-character flush matching batch
  decoding, including invalid byte fragments. Human terminal output escapes controls;
  JSON preserves exact decoded text. Ctrl-C requests cancellation at safe boundaries.

The interface is a terminal, without a browser or server. Each reply prefills full
history; cache reuse across turns, concurrent serving and paged/preallocated storage
are not implemented. Dependencies remain at their existing locked versions.

## Validation and numerical agreement

Host: Apple M4 Max / Mac16,6, 64 GiB unified memory, macOS ARM64. Available backends
were CPU and MPS; CUDA unavailable. Python 3.14.7, PyTorch 2.14.0, package 0.8.0,
float32, one CPU thread for the retained-model measurements.

- **210 passed in 14.84 s** in the locked development environment.
- **210 passed in 14.52 s** from a fresh non-editable wheel installation; imported
  package path confirmed inside the fresh environment, not the editable source.
- Lint, formatting (88 Python files), source/wheel builds and CPU/MPS diagnostics
  passed. Real fresh-wheel chat commands passed for both artifacts on both backends.
- The 23 new tests cover CPU/MPS chunked and incremental prefixes, causality, capacity,
  batch independence, cache reset/error rollback, normal weight mutation detection,
  sampled/greedy agreement, RNG isolation, Unicode/invalid bytes, stops, history,
  subprocess JSON/interactive errors/reset/EOF and real POSIX SIGINT cancellation.
- Separate review fixed cancellation during the final token and final Unicode flush;
  regression tests and the complete suite were rerun. No remaining actionable finding
  was identified within this review's bounded scope.

All-vocabulary logits were compared on the same backend against uncached full
prefixes, for one-token increments through 32 positions and chunks through all
512 positions. Changed future tokens left earlier logits unchanged; reset checks
also passed. Tolerance means `abs(actual-reference) <= atol + rtol*abs(reference)`;
the largest absolute difference can exceed atol alone while satisfying that bound.

| Artifact/backend | atol / rtol | Incremental max absolute error | 512-position max absolute error |
| --- | --- | ---: | ---: |
| Base CPU | 1e-5 / 1e-5 | 1.431e-5 | 1.621e-5 |
| Base MPS | 1e-4 / 1e-4 | 1.717e-5 | 1.526e-5 |
| SFT CPU | 1e-5 / 1e-5 | 1.717e-5 | 1.669e-5 |
| SFT MPS | 1e-4 / 1e-4 | 1.526e-5 | 1.812e-5 |

Cached/uncached greedy token IDs matched for each fixed prompt on each backend,
including the warm-up and three measured repetitions. This is not bitwise logit
identity, arbitrary-prompt sampling determinism, or a cross-device comparison.

## Latency and memory

Each artifact/backend ran in a fresh process. Synchronization brackets device
work. Three repetitions follow one warm-up per case; loading/imports and terminal
consumer delay are excluded. Timings describe this host/run, not deployment SLAs.
No competing validation commands ran during the final accepted measurement matrix.

The fixed synthetic workload uses a 128-ID prefix and 32 forward predictions
(one prefill plus 31 appended predictions), avoiding early-EOS bias. Subsequent
throughput here is **forward-only**, excluding sampling/text decoding. The table
reports the median of three run-level results.

| Artifact/backend | Prefill uncached / cached (ms) | Decode uncached / cached (tokens/s) | Decode ratio |
| --- | ---: | ---: | ---: |
| Base CPU | 7.760 / 7.880 | 111.3 / 712.8 | 6.40× |
| Base MPS | 3.562 / 3.621 | 279.7 / 353.7 | 1.26× |
| SFT CPU | 7.740 / 7.827 | 110.8 / 727.5 | 6.56× |
| SFT MPS | 3.597 / 3.739 | 280.6 / 361.7 | 1.29× |

Caching improved subsequent decoding here but **did not improve prefill**. Short
replies may see much less benefit; larger models, batches and other devices were
not measured.

For the three authored chat cases (25, 65 and 51 prompt tokens), first-token latency
includes forward, synchronization, CPU sampling and incremental text decoding.
Subsequent latency is the median across measured subsequent tokens. These aggregate
different lengths and are descriptive, not a controlled model-quality comparison.

| Artifact/backend | First token uncached / cached (ms) | Subsequent token uncached / cached (ms) |
| --- | ---: | ---: |
| Base CPU | 4.243 / 4.298 | 5.185 / 1.327 |
| Base MPS | 3.278 / 3.648 | 3.374 / 2.937 |
| SFT CPU | 4.135 / 4.210 | 3.672 / 1.352 |
| SFT MPS | 3.383 / 3.436 | 3.346 / 2.817 |

Base replies all used the 32-token cap; SFT replies reached EOS after 4, 3 and 2
tokens. All fixed prompts, decoded samples and IDs are in results.json. They were
not scored or used to choose a model, prompt, checkpoint or inference setting.
The Phase 6 negative result remains the quality evidence.

Logical KV storage at 512 positions, batch 1, is **12,582,912 bytes (12 MiB)**;
at the last forced-workload prefix (159 positions), **3,907,584 bytes**. Cache
concatenations and attention temporaries add memory. Process-lifetime peak RSS
for base CPU/MPS and SFT CPU/MPS was respectively **559,251,456 / 732,790,784 /
564,477,952 / 685,260,800 bytes**, including loading and numerical validation.
The measured MPS boundary maxima were **76,573,184 allocated bytes** and
**1,116,422,144 / 1,116,438,528 driver bytes** for base/SFT respectively. They are allocator snapshots, not
continuous peaks or cache-only increments; do not add unified-memory counters to RSS.

## Failures, reproducibility and remaining limits

The first full-suite invocation used the environment's pytest executable without
putting its installed CLI on PATH: seven subprocess command-lookup failures and
202 passes. Running through the locked environment corrected the harness; the final
full suites passed. An offline fresh-wheel dependency install also failed because
some locked packages were absent from the local cache; installing the same locked
versions from public registries succeeded without paid services. Both failure logs
remain preserved, as do pre-review measurements and the cancellation review record.

`outputs/phase-7-validation/` holds raw accepted `*-accepted.json` measurements,
logs, diagnostics, source/lock/test snapshots and the preservation result. Compact
public results include exact source hashes and raw-report SHA-256 identities.
`artifacts.json` inventories retained evidence. Source commit in measurements is
the Phase 6 closure plus recorded dirty source hashes; the final Phase 7 commit is
provided separately in the push checkpoint. No unmeasured commit is passed off as
the execution source.

At the original Mac checkpoint, no Phase 7 Windows/CUDA, hosted Linux CI or physical
Linux run had occurred. The subsequent [closure](CLOSURE.md) supersedes that execution
status within its stated scopes; physical Linux remains deferred. Prior
Windows/CUDA PASS covered four SFT updates / 175 assistant-target exposures, not
full Mac SFT reproduction; prior hosted Linux CPU CI passed separately within its
documented tiny workflow scope. Physical Linux remains deferred. No useful
instruction following, general CUDA determinism, cross-device equality, mixed
precision or distributed behavior is established. Prior experiments used at most
256-token windows; 512-position numerical agreement does not establish 512-token
language quality. At that original checkpoint no remote publication or artifact
backup had occurred; implementation publication is now verified, while ignored
artifacts remain local and are not remote backups.
