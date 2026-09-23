# Phase 14 — Evaluation Foundation

New retrospective evaluation of preserved historical artifacts, recorded on the
Mac on 2026-09-23. These are **not original Phase 5–12 measurements**. The fixed
framework and gates establish a scoreboard, not useful learned capability.
All five historical models fail every one of the 96 new objective instruction
cases. No training, checkpoint selection or model promotion occurred.

## Protocol and inputs

The [pre-run plan](PLAN.md), [frozen protocol](../../configs/evaluation/protocol-v1.json)
and [methodology](../../docs/EVALUATION.md) specify independent next-token scoring,
legacy LM continuity, matched-text BPB, all 24 fixed generation prompts, six
instruction categories, two ARC difficulty partitions, numerical uncertainty,
contamination policy, failure handling and prespecified Phase 17/18 gates.
All 869 external questions and 96 development instructions fit the existing
256-token scoring/session protocol; no example was filtered for length or outcome.

[Model identities](models.json) bind the unchanged Phase 5 base, original reviewed
SFT, preserved merged LoRA, separate full-tuning control and reviewed DPO. All were
available locally. The live unmerged adapter is not a separate quality run: prior
merge validation remains scoped numerical evidence, and the new result describes
the exact saved merged artifact. The original random/unigram results remain in
Phase 5; new analytical uniform-token, random-choice expectation, first-choice,
empty/constant/echo controls are separately labeled.

## Measurements

| Preserved artifact | Legacy NLL/token | Matched BPB | ARC-Easy raw | ARC-Challenge raw | New instructions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 5 base | 4.731899 | 1.872944 | 157/570 (27.54%) | 66/299 (22.07%) | 0/96 |
| Phase 6 SFT | 5.653422 | 2.186640 | 155/570 (27.19%) | 68/299 (22.74%) | 0/96 |
| Phase 9 merged LoRA | 4.773844 | 1.890576 | 153/570 (26.84%) | 63/299 (21.07%) | 0/96 |
| Phase 9 full control | 5.653425 | 2.186633 | 155/570 (27.19%) | 68/299 (22.74%) | 0/96 |
| Phase 10 DPO | 5.603781 | 2.175885 | 159/570 (27.89%) | 66/299 (22.07%) | 0/96 |

| Preserved artifact | Easy byte-normalized | Challenge byte-normalized | Empty continuations /24 | Mean repeated-trigram fraction |
| --- | ---: | ---: | ---: | ---: |
| Base | 135/570 | 65/299 | 0 | 38.82% |
| SFT | 139/570 | 85/299 | 1 | 5.40% |
| Merged LoRA | 142/570 | 70/299 | 0 | 45.32% |
| Full control | 139/570 | 85/299 | 1 | 5.40% |
| DPO | 137/570 | 82/299 | 0 | 13.33% |

Primary external accuracy uses summed answer log likelihood; the second table
shows a fixed byte-length diagnostic, not a post-result scoring substitution.
These near-chance/poor outcomes do not establish reasoning or broad knowledge.
Lower repetition in the tuned models does not establish better replies; all six
instruction categories remain 0/16 for every model. Full per-book statistics,
Wilson intervals, controls, paired comparisons and resources are retained in
[results.json](results.json); all generated outputs and per-case likelihoods remain
in the identified local run directories.

Legacy NLL matches the old objective within reduction precision. It includes
111,523 targets and EOS; matched BPB uses a different, tokenizer-independent text
segmentation and excludes EOS. Their perplexities must not be mixed. The preserved
base remains best on both language measures. The original SFT/LoRA/control 0/32,
negative DPO preference ranking and failed learned-tool evidence remain intact.

## Contamination, reservation and interpretation

The [development overlap audit](contamination.json) found no exact full-text or
13-word contiguous matches against historical book/instruction training for 989
queries. Of those, 102 are too short for the 13-word check. This is limited lexical
evidence, not absence of semantic overlap or a clearance inherited by future data.
Questions, answers, generated outputs and evaluation source families must be
excluded from Phase 15 training with a new provenance/near-duplicate audit.

Both historical reserved test payloads remain opaque-hashed only, never parsed or
scored in Phase 14. The additional 48 final instruction cases remain unscored.
Final acceptance requires a locked passing candidate and reviewed data exclusion,
then one paired evaluation under the frozen access policy. Phase 14 does not report
a final-test baseline, useful assistant quality or a passed Phase 17/18 gate.

## Evidence, validation and limits

The complete checks and resource scope are in [validation.json](validation.json)
and the [separate review](REVIEW.md). Source snapshots, raw results, inventories and
all attempts are retained outside Git; tracked identities are not artifact backups.
The first run remains `outputs/phase-14-evaluation`; the reviewed run is identified
in results/validation. Base repetition compares all raw scores, choices and generated
IDs in separate processes with frozen tolerances and without timing equality.

The only new dependency is optional PyArrow 25.0.1 (Apache-2.0); existing training
versions remain unchanged. The two evaluation-only Parquet files total 141,823 bytes,
CC BY-SA 4.0, pinned to an exact author-hosted revision and content hashes. No acquired
questions, training corpus or weights enter Git. Original fixture JSON uses MIT.

Mac CPU/MPS tests and MPS historical measurements are actual local evidence.
CUDA/Windows and physical/hosted Linux Phase 14 execution are untested. Package
version remains 1.3.0; no new tag, release or artifact upload is proposed. Phase 15
requires verified Phase 14 publication **and separate explicit owner/master-planning
authorization**. Push approval alone does not lift that transition gate.
