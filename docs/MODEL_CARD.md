# Model card — LatoS v1.0.0

Historical release record, recorded 2026-09-21. v1.0.0 is now published; its
asset bytes and the measurements below remain unchanged. See the
[release guide](RELEASE.md) and [later evidence](INDEX.md). This is an educational
software release, not a claim of useful assistant quality. Original code is MIT; dependencies retain their own terms.
The tiny fixture model/tokenizer are distributed under MIT with provenance. Full
English base/SFT weights and the English tokenizer remain preserved locally and
are not included in the published release assets. Their distribution is deferred;
this is a release scope choice, not a determination that redistribution is prohibited.

## Full English artifacts

Both use an original 17,308,032-parameter dense decoder: eight layers, width 384,
six heads, SwiGLU width 1,024, RMSNorm, rotary positions, tied input/output weights,
float32. Capacity is 512 positions; training and quality evaluation used at most
256. Algorithm/dependency attribution is in [THIRD_PARTY.md](../THIRD_PARTY.md).
No pretrained weights or tokenizer were imported.

| Identity | Selected artifact / SHA-256 |
| --- | --- |
| Base | Phase 5 fixed update 3,000, `outputs/phase-5-english-pilot/step-00003000/model/` |
| Base weights | `f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa` |
| Experimental SFT | Phase 6 fixed update 200, `outputs/phase-6-instruction-reviewed/step-00000200/model/` |
| SFT weights | `62b890214e5544292cc54b725a4e505730fca0d501b3337cdd3f051a622d09ae` |
| Shared English BPE | 8,192 entries, `artifacts/tokenizers/english-bpe-v1/` |
| Tokenizer JSON | `7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab` |

The independently fitted tokenizer uses byte-level BPE, identity normalization,
PAD=0, BOS=1, EOS=2 and UNK=3. The LatoS loader restores literal-special-token
handling. Chat contract version 1 encodes role headers and content separately;
BOS appears once and each completed turn ends with EOS. Use the shared formatter,
not concatenation followed by one encoding. See [chat details](INSTRUCTION_TUNING.md).

Base training began from seed-17 random weights and saw 5,761,229 target exposures
over 3,000 updates (4.511389 corpus-equivalent passes). SFT began from that fixed
base with a fresh optimizer, seed 61, 200 updates and 6,188 assistant-target
exposures. Final updates were selected before the runs, despite better intermediate
validation measurements. All earlier attempts and failures remain preserved.

| Full validation metric | Random base | Trained base | Experimental SFT |
| --- | ---: | ---: | ---: |
| English loss, 111,523 targets | 9.089003 | 4.731898 | 5.653422 |
| English perplexity | 8857.350 | 113.511 | 285.266 |
| Assistant-only loss | Not measured | 8.354879 | 5.815233 |
| Exact final replies | Not measured | 0/32 | 0/32 |
| Replies ending in EOS | Not measured | 0/32 | 32/32 |

SFT improved stopping and teacher-forced assistant loss while failing every held-out
exact task and worsening English loss. It is not an improved base. Perplexities
are comparable only with the same tokenizer, data and objective. Neither reserved
test set was evaluated. Samples are all retained, including repetition, incoherence
and factual errors; see the [consolidated report](../experiments/phase-8/REPORT.md).

## Downloadable tiny fixture artifact

The separate `latos-1.0.0-tiny-fixture.zip` contains a 127,808-parameter model
(two layers, width 64, four heads, feed-forward width 192, capacity 64), its
328-entry tokenizer (512 requested; the tiny corpus yields fewer merges), cards, MIT license and hashes. It is trained for 400 CPU
updates on three original fixture paragraphs, using the existing Phase 4
acceptance exercise. It is a memorization/mechanics demonstration, not the English
pilot, an SFT model or an instruction assistant. The release's
[tiny inventory](../experiments/phase-8/tiny-artifacts.json) pins every learned file.
Its source inputs are included in the source distribution. No optimizer state,
corpus, execution logs or private files are bundled in this zip.

## Supported evidence, not blanket platform certification

| Environment | Evidence and boundaries |
| --- | --- |
| Mac M4 Max CPU/MPS | Full retained English models passed Phase 7 cache checks through 512; current release checks are separately recorded in Phase 8 evidence. |
| Windows RTX 4070 SUPER CUDA | Owner-reported Phase 7 PASS at v0.8.0: 202 tests, eight skips, fresh wheel and CUDA CLI. Tiny base/SFT capacity 256 plus a separate synthetic 512-position/batch-two model. Full Mac learned artifacts were unavailable and not tested. Windows console Ctrl-C remains unvalidated. |
| GitHub-hosted Linux x86-64 CPU | Separately inspected Phase 7 CI: 203 tests, seven MPS-only skips; tiny inference/CLI/POSIX SIGINT, offline workflows/builds. No full learned artifacts, CUDA/MPS, retained-artifact latency matrix or fresh-wheel suite. |
| Physical Linux | Deferred, not performed. |

Phase 7 same-backend cache tolerances were CPU/CUDA atol=rtol=1e-5 and MPS
atol=rtol=1e-4, on their respective artifacts. Fixed greedy ID agreement is not
cross-device equality or general CUDA determinism. No Phase 8 Windows/Linux
execution is implied by historical checks. The dependency lock selects macOS
arm64, Linux x86-64 CPU and Windows x86-64 CUDA wheels only.

Appropriate use: learning and bounded reproducibility experiments. Useful
instruction following, factual reliability, safety alignment, broad English quality,
512-token language quality, production latency, mixed precision and distributed
serving remain unestablished. At v1.0.0 there was no tool protocol or hosted serving
layer. All inference is local. Later main adds a bounded tool protocol; see the
addendum below. Model-only loading is distinct from optimizer recovery, which requires the original recorded implementation/runtime. See [release guide](RELEASE.md).

## Phase 10 — Model comparison

See the [bounded preference experiment](../experiments/phase-10/REPORT.md) and
[preference provenance](../experiments/phase-10/DATA.md). The original base, SFT,
tokenizer and their identities above remain unchanged. DPO is a separate local
experimental model, not a promoted replacement. Both reserved tests remain unused.

## Post-release capability evidence through Phase 12

The [LoRA/full-tuning comparison](../experiments/phase-9/REPORT.md) produced 0/32
exact replies for both methods. [DPO](../experiments/phase-10/REPORT.md) worsened
full Mac held-out ranking from 16/32 to 15/32 and retained 0/32 exact replies;
English loss improved versus SFT but remained worse than the original base.
No useful assistant or preference improvement was established.

The [tool protocol](TOOLS.md) provides bounded local mechanics, not learned tool
competence. [Phase 11](../experiments/phase-11/REPORT.md) retained 0/16 valid JSON
for each original Mac model. [Phase 12](../experiments/phase-12/REPORT.md) removed
44 blocked retries by raising the inference-session budget from 256 to 512 within
existing capacity; valid JSON remained 0/41, 0/64, 0/64 emitted replies for
base/SFT/DPO, with no calls and each 0/12 tasks and 0/4 learned failure handling.
The [closure](../experiments/phase-12/CLOSURE.md) separately records Windows synthetic
initial-block relief and hosted Linux CPU checks. None establishes learned tools,
general long-context language quality or an improvement to the retained base.
[Roadmap 2.0](../ROADMAP.md) is future work; this card does not describe a 2.0 model.

## Phase 17 — preserved but unaccepted recovered model

The [bounded experiment closure](../experiments/phase-17/CLOSURE.md) records a
34,087,424-parameter recovered final artifact with SHA-256
`d37141054b26f777403e855ee5903f45a8b5d22205d48d6ab37f9e73fc39f4e6`.
It uses the Phase 15 16,384-entry tokenizer and selected native context-512 model.
Its tensor file is 136,355,352 bytes. It is retained locally on Mac and Windows,
not distributed as a release or accepted as LatoS Base Model 2.0.

The original final state was not saved after its supervisor failure. Recovery from
step 7,000 produced this model, but 351 of 485 replayed updates exceeded the frozen
`1e-5` loss-difference gate. Source/configuration/counters matching and valid finite
float32 serialization do not waive that failed gate. Do not substitute this artifact
for the missing original final state or promote it into Phase 18.

No fixed paired development or final acceptance scores exist. New-corpus monitoring
loss 3.964917 / perplexity 52.715883 is not an acceptance result or a cross-tokenizer
comparison. Logical exposure is 37,811,418 targets; total physical work including
replay is 40,238,071, with repeated work counted separately. The divergence cause
is unestablished. All historical baselines remain unchanged.
