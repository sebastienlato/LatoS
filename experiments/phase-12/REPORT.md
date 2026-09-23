# Phase 12 — fixed inference context budget

Completed locally on 2026-09-23; publication approval pending. Phase 11 closure
`e2d61100a7c0c74d759bdc4ded5ddcda89cb112b` and all eleven remote tag objects/targets
were verified before the owner's explicit fresh-chat start. Verified publication
supersedes historical pending snapshots. No remote write or new paid service.

## Question and controlled extension

Does additional conversation history room remove the retry bottleneck and suffice
for successful tool use? The [pre-run plan](PLAN.md) fixed 256 versus 512 total tokens,
64 generated tokens per turn, four turns and two calls maximum. All three preserved
17,308,032-parameter base/SFT/DPO models already have 512-position capacity but lack
evidence of useful learned language quality beyond 256. No configuration or weight
was changed. The runtime/default context stays unchanged; this is an experiment-local
inference budget ablation, not context training or an architectural extension.

Original Phase 11 prompt, tokenizer, chat contract 1, greedy cached float32 generation,
seed 0, one CPU thread, tools, feedback, 16 development cases and scoring stay fixed.
Exact model/tokenizer/config/metadata/source identities and environment are in
[plan.json](plan.json). Existing Mac M4 Max MPS, Python 3.14.7 / PyTorch 2.14.0;
no dependency changes or external data. The runner checks each model's 256 baseline
against saved Phase 11 sessions/scores/metrics before its 512 condition. Run order
was base, DPO, SFT, with 256 then 512 for each. No case, model or result selection.

## Results

The 256 baseline exactly reproduced original sessions, scores, metrics and tokens.
All 44 formerly context-limited sessions emitted additional replies at 512: 12 base,
16 SFT and 16 DPO. **H1 passed. H2 failed:** none of the models completed a normal task.

| Measure | Base 256 → 512 | SFT 256 → 512 | DPO 256 → 512 |
| --- | --- | --- | --- |
| Sessions with multiple emitted replies / 16 | 0 → 12 | 0 → 16 | 0 → 16 |
| Valid JSON / emitted replies | 0/16 → 0/41 | 0/16 → 0/64 | 0/16 → 0/64 |
| Correct calls / 20 required positions | 0 → 0 | 0 → 0 | 0 → 0 |
| Successful normal tasks / 12 | 0 → 0 | 0 → 0 | 0 → 0 |
| Learned missing-key failure handling / 4 | 0 → 0 | 0 → 0 | 0 → 0 |
| Generated tokens including EOS | 595 → 1,834 | 48 → 192 | 51 → 195 |
| Context-limit stops / 16 | 12 → 0 | 16 → 0 | 16 → 0 |

No envelope, proposed call or executed call appeared. Argument validity and execution
rates remain undefined (0/0), not perfect. At 512 the base stopped with 14 incomplete
generations and two turn limits; SFT and DPO each exhausted all four turns in every
case. Maximum observed prompt lengths were 338/342/342; respectively 13/32/32 replies
started with more than 256 prompt tokens. These are measured generation positions,
not language-quality evidence. More capacity removed overflow but did not fix syntax
or semantics. This rules out insufficient retry room as the sole explanation for
failure under this fixed protocol; it does not establish that all context limits
or all other prompts would behave the same way.

Scripted controls remain **12/12 tasks and 4/4 failures handled**, with 36/36 JSON,
20/20 schema-valid/correct calls, 16/20 successful executions and four expected
missing-key errors. They bypass model generation/token budgeting and establish only
protocol/scoring mechanics; they do not demonstrate learned tool use at either length.

## Resources, artifacts and verification

Accepted run: `outputs/phase-12-context-reviewed`. Initial run:
`outputs/phase-12-context`. Both are retained. Each emitted 2,915 tokens across 96
learned sessions; both together emitted 5,830, below the fixed two-run bound 49,152.
Zero training updates or optimizer exposures. The second run followed review fixes;
all raw sessions, scores, metrics, attempt records and token counts match the first.

The reviewed evaluation took 16.436 seconds, RSS high-water 740,458,496 bytes,
MPS end-boundary driver allocation 1,108,033,536 bytes and live tensor allocation
zero after model deletion. Full-suite engineering tests ran concurrently with this
reviewed experiment; timings are observations, not isolated performance benchmarks.
MPS readings are not continuous peaks and must not be added to RSS. Both runs stayed
within the ten-minute evaluation/8 GiB RSS budgets; new evidence remained under 2 GiB.
Time/RSS limits are checked at attempt boundaries and completion; an active kernel
cannot be preempted. Missing RSS support is reported explicitly on other platforms.

[Results](results.json), [all raw sessions and attempts](samples.json),
[artifact inventory](artifacts.json), and [verification](verification.json) are tracked.
Inventory paths are relative to the accepted local run and are not download links.
Captured source uses the closure parent plus exact hashes of the locally edited
experiment, tests and packaging metadata; it is not a claim of a clean run-time commit.
Model loading checks pinned hashes, evaluation compares unchanged tensors and absent
gradients, and before/after preservation hashes cover all 78,036 pre-existing files.
Both reserved tests are opaque-hashed only, never parsed/evaluated. No model is promoted.

The new verifier checks complete inventories, source identities, fixed conditions,
resource bounds, replayed prompt lengths/overflow and per-case scores. It replays 96
learned sessions without inference; the unchanged independent Phase 11 scorer recounts
128 sessions including scripted controls. This independent scorer is scoped to these
saved outputs, not an independent general JSON parser. Seven new engineering tests
cover overflow/replay, forged evidence, inventory corruption, budget exhaustion,
model capacity and direct tiny CPU/MPS generation beyond 256. Validation, initial
failure retention and installed-wheel checks are in [validation.json](validation.json).
The [separate review](REVIEW.md) records fixes and verification boundaries.

## Reproduction and scope

From a locked checkout with the original local artifacts restored at their recorded
paths, choose a new nonexisting output directory:

```sh
uv run --locked python experiments/phase-12/run.py --output-dir outputs/phase-12-repeat
uv run --locked python experiments/phase-12/verify.py outputs/phase-12-repeat
```

Default is MPS; `--device cpu` is an explicitly different experiment, not a claimed
full-model CPU replication. Exact baseline mismatch aborts; retain it rather than
changing thresholds. These commands require original private local learned artifacts;
a Git clone alone does not contain them. The verifier needs only the retained run,
matching source checkout and original tokenizer, not model weights. Both reserved
test payloads are excluded. Development cases were already observed, not held-out
fresh tests; no generalization or statistical population claim follows.

Original [Phase 11 closure](../phase-11/CLOSURE.md) remains unchanged: original full
Mac prompt/models yield 0/16 JSON each, distinct from owner-reported tiny Windows RTX
4070 SUPER compact-prompt results (also 0/16 each; 1,024 tokens/model; original prompt
rejected at tiny 256 capacity). Exact Windows compact prompt/logs/hashes were not
supplied. Separately inspected hosted Linux CPU evidence is limited to its recorded
protocol/tiny inference/workflows, not either full experiment. No Phase 12 Windows,
CUDA or Linux run is claimed. Physical Linux remains deferred.

Prior negative SFT/LoRA/control/DPO evidence remains unchanged, including full Mac
DPO ranking 16/32 → 15/32, exact 0/32, and English loss still worse than original base;
tiny Windows Phase 10 is separately scoped. Merge atol=rtol=1e-4 and original CPU
1e-5 failure stay retained; cache bounds CPU/CUDA 1e-5, MPS 1e-4. Adapter/DPO optimizer
resume unsupported; dense recovery version-bound; Windows OS-level Ctrl-C unvalidated.
No useful assistant, factual reliability, safety alignment, general determinism,
cross-device equality, production performance, mixed precision, distributed serving,
learned tool use or general learned language quality beyond 256 is established.

Public runtime, defaults, package version 1.3.0 and lock unchanged; source packaging
adds the experiment. Propose the reviewed commit to existing
`https://github.com/sebastienlato/LatoS.git`, branch `main` only. No tag, release,
asset upload, backup branch or visibility change. Stop for explicit approval.
