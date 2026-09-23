# Phase 14 — frozen evaluation plan

Fixed before any Phase 14 learned-model scoring on 2026-09-23. Starting source:
565bf08fbd8aaa360056fc670b08b8879be9312c. No training or checkpoint selection.
The machine-readable [protocol](../../configs/evaluation/protocol-v1.json) pins
suite/input hashes, scoring, decoding and numerical gates. Suite changes require
a new version and paired baseline reruns; never silently improve an old score.

Evaluate all available historical dense artifacts: Phase 5 final base, Phase 6
reviewed SFT, Phase 9 merged LoRA and full-tuning control, Phase 10 reviewed DPO.
Do not replace unavailable artifacts. Native merged LoRA is measured as the preserved
merged artifact, not a claim of bitwise equivalence to the live adapter. Keep all
old evidence unchanged. Prior development cases remain observed historical evidence.

Use existing Mac MPS, float32, one CPU thread, scoring batch eight, context 256.
Run base twice in separate processes; after scoring fixes rerun all affected results
and preserve prior attempts. Require same-host per-row NLL atol 1e-5 / rtol 1e-6,
identical predictions/greedy IDs and integer counts. CUDA is untested in this phase.
No new paid resources. Budget each complete run to 30 minutes; diagnose rather than
silently shorten an evaluation if resources fail. Record actual time and memory.

Development LM: all 1,030 paragraphs in the existing two-book validation split.
Keep the old EOS-inclusive 256-token objective for continuity. Add independent
192-UTF-8-byte spans, reset BOS at each span, no EOS score, for matched-text BPB.
Span boundaries are tokenizer independent; every text byte and target is accounted
once. New tokenizers must be lossless and fit the same spans in context 256.
Per-document metrics expose aggregation regressions; two books cannot establish
population uncertainty or broad language proficiency.

Generation: all 24 original prompts, greedy/cached 64 tokens, no alternate samples.
Instruction: all 96 original cases, six equally sized categories, chat contract 1,
greedy/cached 64 tokens; exact or strict JSON scoring. Empty/constant/echo controls
are separate non-model baselines. Freeze an additional 48 original acceptance cases
in six categories, unscored until Phase 18's locked final selection. Their templates
are distinct from development; they are small synthetic fixtures, not human judgments.

External: all ARC-Easy 570 and ARC-Challenge 299 validation questions. Pin the
AI2-hosted dataset revision and actual Parquet bytes. Zero-shot question/answer
prefix; likelihood of each complete space-prefixed answer; exclude prefix and EOS.
Joint BPE must preserve the prefix boundary. Raw sum is primary; likelihood per
answer UTF-8 byte is a secondary length-bias diagnostic. Lowest-index ties. Retain
all choice scores and exact IDs. Overflows count as failures, never disappear from
denominators, and block acceptance. No ARC training or test download; no retrieval
corpus. Uniform-chance and first-choice baselines recorded separately for each split.
Do not call these settings interchangeable with external leaderboard protocols.

Prespecified numeric gates, not targets derived from retrospective outputs:
Phase 17 needs at least 10% lower matched BPB, no book worse by over 2%, ARC-Easy
raw accuracy >=35% and >=5 percentage points above historical base, descriptive
95% lower bound above chance, no external raw/byte-normalized drop over 3 points.
Phase 18 needs >=60% instruction accuracy, >=40% in each category, >=30-point gain
on the strongest historical instruction result; matched BPB at most 10% above the
accepted Phase 17 base, each book at most 15% above it, and external drops <=3 points.
Both limit empty generation to 5%, repeated-trigram fraction to 25%, and increase
in repetition to 10 points. These are project minimums for a bounded capability
claim, not validated universal definitions of a useful assistant. Fixed minimums
plus improvement and regression limits avoid near-random promotion or a weighted
composite. A failed gate blocks the dependent phase; only an explicit roadmap
amendment can change thresholds, never post-result tuning.

Final acceptance is separate from routine development. Do not open either historical
reserved payload now. After a candidate passes development gates and is locked with
its data/contamination/selection record, one paired final comparison may score the
historical reserved English split with the same LM protocols. Phase 18 additionally
scores the 48 new reserved instruction cases against paired historical references;
the old 32-case instruction test remains reserved and is not silently repurposed.
No training or selection may follow final feedback without an approved amendment
and a genuinely fresh acceptance design. Public fixture source is not secret;
exclusion from training and disciplined access, not obscurity, establish the policy.
