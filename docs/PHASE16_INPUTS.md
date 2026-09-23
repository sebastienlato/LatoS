# Phase 16 input package

Phase 15 supplies data and a selected codec only. Phase 16 owns model architecture,
CUDA mixed precision, streaming/packing integration, memory and throughput probes.
The authoritative development host remains the Mac. Substantive future GPU work
targets the existing RTX 4070 SUPER (~12 GB VRAM); no new paid resources.

## Local artifacts and identities

- Complete audited corpus: `data/processed/data2-v1-ready`.
- Selected codec: `artifacts/tokenizers/data2-v1/new16384`.
- Portable local inputs: `outputs/phase15-input-package`.
- `bundle.json` verifies every copied file. No weights, raw data or reserved
  evaluation payloads are included. Corpus reservations remain in the full corpus.
- [Source terms](DATA_2.md), [audit](../experiments/phase-15/contamination.json),
  [token counts](../experiments/phase-15/token-accounting.json) and
  [exact identities](../experiments/phase-15/artifacts.json) accompany the handoff.

Tokenizer SHA-256: `23b9182d5943e7f8b32c6f415f108b7d06f229c80f5cd8f236258db40a6b94c4`.
Corpus report SHA-256: `af7882b5fdcbab25bd1412ced2d0af0e2c1fd352a9e1457d708bd62c5c24a84b`.

The native LatoS loader and chat contract 1 are required. These token IDs are not
compatible with the old learned weights. Original corpus readers retain their old
bounds; integrate Data 2.0 explicitly through its verified train/development reader.
Both legacy final reservations and new reservations stay out of routine selection.

## Actual sequence constraints

Pretraining rows: 332,782; content tokens 37,226,334. Length including BOS/EOS:
median 98, p95 222, p99 320, maximum 2,582. The historical isolated-paragraph
window policy would spend much of a long fixed batch on padding:

| Context | Windows/pass | Useful target-slot fraction |
| --- | ---: | ---: |
| 256 | 342,278 | 43.03% |
| 512 | 333,088 | 22.07% |
| 1024 | 332,790 | 11.03% |
| 2048 | 332,783 | 5.51% |

Evaluate within-document coalescing or validated packing before translating
a theoretical token budget into wall time. Any cross-document attention/boundary
policy needs tests; no packing speedup or quality is measured in Phase 15. Small
lead paragraphs and lists were omitted during prose extraction, so some retained
paragraphs have limited standalone context. Group/document/paragraph IDs permit
ordered reconstruction of retained material, not recovery of omitted text.
The 8,000-character chunks can cut words; concatenate consecutive chunks of the
same paragraph before applying a future sequence-boundary policy if needed.

Instruction streams: median 135, p95 633, p99 1,062.1, maximum 5,075 tokens.
Keep entire conversations and record context exclusions; never truncate answers.

| Context | Whole conversations that fit | Overflow | Assistant targets available |
| --- | ---: | ---: | ---: |
| 256 | 10,516 | 3,743 | 692,504 |
| 512 | 13,111 | 1,148 | 1,214,346 |
| 1024 | 14,101 | 158 | 1,655,190 |
| 2048 | 14,242 | 17 | 1,771,611 |
| 4096 | 14,257 | 2 | 1,787,020 |

All 14,259 conversations carry 1,787,217 potential assistant targets. A 512-token
limit retains about 92% of conversations but only about 68% of those targets.
Two conversations exceed the native formatter’s current 4,096-token maximum.
No context length or future SFT configuration is selected here.

## Bounded exposure/time options

One isolated-record pretraining pass has 37,559,116 targets including EOS.
The options below reuse the same 37,226,334 content-token positions; extra passes
do not create unique data. These are planning scenarios, not training commitments.

| Passes | Target exposures | At 1,000 useful targets/s | At 5,000/s | At 10,000/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 37,559,116 | 10.43 h | 2.09 h | 1.04 h |
| 2 | 75,118,232 | 20.87 h | 4.17 h | 2.09 h |
| 3 | 112,677,348 | 31.30 h | 6.26 h | 3.13 h |

Rates are hypothetical sustained **nonpadding target** throughput, unmeasured on
the RTX 4070 SUPER. Times exclude evaluation/checkpoint I/O and assume the chosen
batch/window implementation actually achieves the listed useful throughput.
Different padding, context, vocabulary and precision can change that substantially.
The unique corpus remains small relative to the roadmap’s exploratory model-size
range; repetition is not broader coverage and cannot guarantee the Phase 17 gates.

## Vocabulary cost examples, not architecture choices

Tied input/output embeddings count once in parameters, but a larger vocabulary
also raises output-projection/logit work. Relative to 8,192 entries:

| Illustrative width | Additional parameters at 16,384 | Extra float32 weights | Extra weights + gradients + two Adam moments |
| --- | ---: | ---: | ---: |
| 384 | 3,145,728 | 12 MiB | 48 MiB |
| 512 | 4,194,304 | 16 MiB | 64 MiB |
| 768 | 6,291,456 | 24 MiB | 96 MiB |

These arithmetic costs exclude activations, allocator/workspace, logits, optimizer
overheads and mixed-precision master copies. They do not establish 12 GB feasibility.
Use measured probes before choosing the model or substantive training schedule.

Fixed Phase 14 quality gates, benchmark revisions and scoring remain unchanged.
Phase 15’s construction acceptance cannot substitute for Phase 17/18 learned
quality. No model pretraining, SFT, DPO, LoRA, RL, tool learning, interoperability
or CUDA optimization began in Phase 15.
