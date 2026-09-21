# Model card — LatoS v1.0.0 candidate

Recorded 2026-09-21. This is an educational software release, not a claim of useful
assistant quality. Original code is MIT; dependencies retain their own terms.
The tiny fixture model/tokenizer are distributed under MIT with provenance. Full
English base/SFT weights and the English tokenizer remain preserved locally and
are not proposed release assets. Their distribution is deferred; this is a release
scope choice, not a determination that redistribution is prohibited.

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
serving remain unestablished. There is no tool-use or hosted serving layer. All
inference is local. Model-only loading is distinct from optimizer recovery, which
requires the original recorded implementation/runtime. See [release guide](RELEASE.md).
