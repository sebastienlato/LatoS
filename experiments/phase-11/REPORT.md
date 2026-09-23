# Phase 11 — bounded structured tools

Completed locally on 2026-09-23; package 1.3.0, explicit publication approval pending.
The fresh-chat start followed verification that remote main exactly matched Phase 10
closure `87d744fb8cfd4228c78f35bbd220d2833462e037`, with all eleven annotated tag
objects and peeled targets unchanged. v1.0.0 remains at
`10d9ef7bf0618b364f887ff9d279408e3cbc33c9`. Verified publication supersedes historical
pending snapshots. No remote write, new paid service or dependency upgrade occurred.

## Deliverable and fixed experiment

The [interface](../../docs/TOOLS.md) adds strict structured calls to two pure local
tools: bounded integer addition and a three-entry lookup table. Four assistant turns
and two executions maximum; no shell, network, arbitrary code or model-selected
files. Invalid arguments never execute, expected tool failures get structured
feedback, and context exhaustion/cancellation/incomplete generation fail explicitly.
The original tokenizer and shared chat contract 1 are unchanged; host results enter
as ordinary user messages following assistant calls, not a new privileged tool role.

The [plan](PLAN.md) fixed 16 original synthetic development cases before model
evaluation: eight single-call tasks, four lookup/add chains and four missing lookups.
No training, constrained generation, JSON repair, prompt sweep or checkpoint selection.
No corpus payload or reserved test was read. These cases are development validation,
not a generalization benchmark. Independent literal answers and canonical call
sequences define success; equivalent alternative strategies can score as incorrect.
The gold-script control checks mechanics only and is never counted as model output.

The base, original SFT and separate DPO model identities are pinned in [plan.json](plan.json)
and checked before loading; none is promoted as a better baseline. The original
17,308,032-parameter float32 models run on Mac M4 Max MPS with one CPU thread,
Python 3.14.7 / PyTorch 2.14.0, greedy cached generation, seed 0, at most 64 generated
tokens per turn within 256 total context tokens. No truncation is allowed. State
comparisons confirm unchanged tensors and no gradients for all three models.

Accepted local run: `outputs/phase-11-tools-reviewed`. It records the source parent
plus exact source/test/runner/verifier/protocol/lock/package hashes captured before
evaluation. The parent is the published closure and the implementation was dirty
local Phase 11 work at run time. [Artifact identities](artifacts.json) refer to files
relative to that local run, not downloadable artifacts. Raw outputs and source
snapshots remain ignored and local; all 64 session traces are also in [samples.json](samples.json).

## Separate measurements

| Measure | Base | SFT | DPO | Scripted control only |
| --- | ---: | ---: | ---: | ---: |
| Valid JSON / emitted turns | 0/16 | 0/16 | 0/16 | 36/36 |
| Valid envelopes / emitted turns | 0/16 | 0/16 | 0/16 | 36/36 |
| Valid arguments / proposed calls | 0/0 | 0/0 | 0/0 | 20/20 |
| Correct tool and arguments / required call positions | 0/20 | 0/20 | 0/20 | 20/20 |
| Successful executions / executed calls | 0/0 | 0/0 | 0/0 | 16/20 |
| Exact final answer or error / cases | 0/16 | 0/16 | 0/16 | 16/16 |
| Successful normal tasks | 0/12 | 0/12 | 0/12 | 12/12 |
| Expected missing-key failure handled | 0/4 | 0/4 | 0/4 | 4/4 |

**No learned tool use was demonstrated.** The dispatcher works for correct input;
none of the three unchanged models generated a valid JSON value in these cases.
Zero-denominator rates are null/undefined, not perfect argument correctness or
execution. The scripted control's four unsuccessful executions are the intended
missing-key results; all four were correctly reported in the subsequent final error.
Normal success requires both the exact calls and exact final answer. Parsing is
counted independently even for a token-limited response, which still cannot execute.

All 48 model sessions emitted one reply. Base stopped after four token-limited
incomplete replies and twelve context overflows on attempted retry. SFT and DPO each
stopped at sixteen context overflows after invalid first replies. The fixed system
prompt plus failure feedback and a reserved 64-token continuation did not fit the
192-token prompt budget on retry. No second learned turn occurred, and **learned
multi-turn tool execution was not demonstrated**. Successful chains are scripted
mechanics evidence only. The prompt/context budget itself limits this experiment;
no prompt or context sweep was performed to improve the recorded outcome.

Base generated 595 tokens, SFT 48 and DPO 51 (including any EOS), versus zero training
updates or optimizer exposures. Evaluation took approximately 2.015 / 0.174 / 0.182
seconds respectively, 3.289 seconds total including setup/capture. These are single-run
measurements, not production benchmarks. RSS high-water mark was 648,593,408 bytes;
end-of-run MPS driver allocation was 1,099,644,928 bytes and live tensor allocation
zero after model deletion. MPS boundary readings are not continuous peaks or
additive memory pools. Exact values are in [results.json](results.json).

## Validation, review and retained attempts

The independent [verifier](verify.py) rehashes the run inventory and captured source,
recounts all 64 sessions from raw JSON and literal arithmetic/catalog results, checks
call/turn bounds and reconstructs all aggregate fractions. Its [result](verification.json)
is scoped to fixed saved evidence, not model re-execution or a second generic parser.
A corruption fixture verifies rejection of altered evidence.

Unit tests cover duplicates/nonfinite numbers, Unicode/size/depth limits, strict
integer types, extra keys, unknown tools, path-like keys, expected errors, callback
cancellation, incomplete-generation rejection, turn/call exhaustion, recovery,
false-positive metric traps and CLI exit behavior. Tiny CPU/MPS inference compares
the responder with the unchanged chat formatter/generator and checks tensor/RNG/
gradient preservation; synthetic 512-position integration is not learned language
quality beyond 256. The full suite and isolated fresh non-editable wheel results,
packaging and preservation checks are recorded in [validation.json](validation.json).

The [separate review](REVIEW.md) added a cap on retained oversized callback text and
captured the verifier/tests/package metadata in the experiment inventory. The first
run remains at `outputs/phase-11-tools`; the reviewed rerun retained identical raw
model responses, token counts and metrics, with no prompting or selection change.
An initial full-suite invocation had seven CLI lookup failures because the local
environment's command directory was absent from PATH; its log is retained. The
locked-environment invocation resolved those harness failures without test deletion.

## Prior evidence and limitations remain

The full Mac DPO result stays negative: 100 updates / 400 pairs / 2,498 final-response
targets, ranking 16/32 → 15/32, exact replies 0/32 → 0/32, preference loss
0.693147 → 0.718657. English 5.653422 → 5.603780 improves over SFT but remains worse
than original base 4.731898. Original negative SFT and LoRA/control findings remain.

[Phase 10 closure](../phase-10/CLOSURE.md) separately records owner-reported tiny
Windows RTX 4070 SUPER PASS: 243 passed / 11 skips, fresh wheel, four CUDA updates /
16 pairs / 169 response targets, ranking 16/32 → 16/32, exact 0/32. It did not reproduce
the full Mac experiment; raw Windows logs/hashes were not supplied. Separately
inspected hosted Linux CPU run 35742827336 passed 245 / nine MPS-only skips, within
tiny CPU DPO and workflows/builds, without a fresh installed-wheel suite or full
learned experiment. Physical Linux is deferred. **No Phase 11 Windows/CUDA or
hosted/physical Linux validation is claimed.**

All previous models, random baselines, tokenizer, evidence, original failed attempts
and both reserved tests remain preserved. Adapter and DPO optimizer resume remain
unsupported; original dense recovery is version-bound. Merge atol=rtol=1e-4 remains
the amended contract and the original CPU 1e-5 failure remains retained. Cache bounds
remain CPU/CUDA 1e-5 and MPS 1e-4. Windows console Ctrl-C is unvalidated. No useful
assistant, factual reliability, safety alignment, general accelerator determinism,
cross-device equality, production performance, mixed precision, distributed serving
or learned language quality beyond 256 is established. Strict dispatch bounds are
not a sandbox for arbitrary Python callbacks or evidence of model safety.

Propose reviewed source to existing `https://github.com/sebastienlato/LatoS.git`,
branch `main` only. No tag, release, artifact upload, visibility change or remote
backup. Stop for explicit Phase 11 publication approval; Phase 12 has not begun.
