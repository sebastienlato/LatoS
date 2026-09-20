# Measured English instruction tuning

Phase 6 is a bounded experiment initialized from the fixed Phase 5 model. Its
[result](../experiments/phase-6/REPORT.md) is negative for held-out exact task
completion and for English language-modeling regression. It is not a useful chat
assistant. The original base remains the accepted pretraining artifact.

## Reproduce from retained inputs

Use the locked environment from the repository root. No new dependency or paid
service is required. Learned inputs are ignored artifacts; a fresh clone must
separately recover or reproduce the exact Phase 5 base and tokenizer. Defaults pin
their hashes and refuse substitution. All destinations must be new directories.

```sh
uv sync --locked
uv run --locked python experiments/phase-6/data.py --output-dir data/processed/english-instructions-v1
uv run --locked python experiments/phase-6/run.py --benchmark --output-dir outputs/phase-6-calibration-new --device mps
uv run --locked python experiments/phase-6/run.py --output-dir outputs/phase-6-reproduction --device mps
uv run --locked python experiments/phase-6/verify.py --run-dir outputs/phase-6-reproduction --output outputs/phase-6-reproduction-verification.json
```

Skip data materialization when the verified directory already exists; it refuses
to overwrite it. Use `--conversations` to select a different newly generated copy.
The runner also accepts CPU/CUDA explicitly; only local CPU tests and the full
local MPS experiment were exercised in Phase 6. Default 200 updates are fixed in
`configs/training/instruction-pilot.json`; calibration uses ten updates in a fresh
run and discards those weights. The actual calibration and repeated reviewed run
are both retained. The small authored dataset and old base limit interpretation.

## Original synthetic conversations

`experiments/phase-6/data.py` defines 48 training, eight validation, and eight test
word groups before expanding each into copying, label extraction, first-letter
and two-turn recall conversations. Counts are 192/32/32. The templates and word
lists were independently authored by Codex for this project on 2026-09-20; no
external dataset or model-generation API was used. They are original project
material under MIT, not human-authored demonstrations. The generated JSONL and
rights/provenance manifest stay in ignored storage.

Related conversations stay in their word source group. Full conversation hashes
reject exact duplicates during generation. Train/validation/test source groups
and conversation IDs are checked for overlap when reading a split. Similarity of
templates across splits is deliberate, not a near-duplicate-free benchmark claim.
The experiment measures new words in familiar tasks, not new tasks. Neither reader
nor runner opens test JSONL: a test reservation exists in metadata, and tests prove
the complete runner succeeds without either instruction or English test files.

## Shared format and objective

The authoritative implementation is `latos.chat.format_chat`, with versioned
`CHAT_CONTRACT`. The token stream is defined by separately encoding segments:

```text
BOS
encode("\nSystem:\n") + encode(system text) + EOS       # optional system
encode("\nUser:\n") + encode(user text) + EOS
encode("\nAssistant:\n") + encode(assistant text) + EOS
... alternating user and assistant turns ...
```

BOS is ID 1 once; EOS is ID 2 at each turn boundary. Header strings are ordinary
vocabulary tokens. No vocabulary entry or embedding is added. Literal special-token
spellings in content stay ordinary text. Content/header boundaries are deliberately
encoded separately: decoding the stream and re-encoding it as a single string is
not the serialization contract because BPE merges can differ at segment seams.
These textual roles are formatting conventions, not a security boundary.

Training conversations require a nonempty final assistant response. The optional
initial system is followed by alternating user and assistant turns. Empty content,
unknown roles, invalid order and overlong sequences fail explicitly. There is no
silent truncation, packing, overlap, partial-answer training or invented EOS after
a cut. Actual experiment windows are at most 256 tokens, within capacity 512.

Only assistant **content and its EOS** are eligible targets. Headers, system/user
content and their EOS, BOS, and right padding are masked with -100. Previous
assistant responses in multi-turn examples also contribute. Eligibility remains
at the token's own position: the model pairs logits at position t with labels at
t+1 exactly once. Thus the last assistant-header token predicts the first response
token, and the last response token predicts EOS. The next role header is masked.

The shared engine counts eligible targets for accumulation, validation and saved
exposure, rather than counting all nonpadding tokens. Causal attention prevents
valid tokens from seeing right padding; conversations occupy independent batch
rows. The immutable masks are hashed into dataset identity and bound to recovery.
Pretraining datasets with no masks retain their prior identities and objective.

Generation uses the same formatter through the final user EOS and appends only the
assistant header. It then calls the existing uncached sampler; it is not a Phase 7
chat interface. Tests verify exact training-prefix equality, including multi-turn
history and first-response alignment.

## Measurement and checkpoints

The [predeclared plan](../experiments/phase-6/PLAN.md) fixes the final checkpoint,
hyperparameters and scoring. Teacher-forced validation covers every assistant
content/EOS target. Exact-match scoring covers each conversation's final reply,
case-sensitive after stripping outer whitespace, with greedy decoding capped at
32 new tokens. EOS stopping is a separate count. Recall uses the gold previous
assistant turn. All 32 samples per model are reported without cherry-picking.
English regression scores the full existing validation split, 111,523 all-target
transitions in 1,067 windows, with the same tokenizer/256-token policy before and
after. Assistant loss and all-target English loss are distinct objectives.

`plan.json` records base/tokenizer/config/data/implementation identities, the source
commit and dirty status, runtime and selection. `source/` preserves exact package
Python files, `runner.py` the runner, and `uv.lock` the locked versions. Initial and
update 100/200 directories are full recovery checkpoints. `metrics.jsonl` records
every update, actual targets, synchronized timing, validation and memory. Run
summary, baselines, full samples and an exact file/hash inventory are preserved.
No generated data, weights or full logs are committed.

The fresh optimizer starts at update zero with no moments. This is not resumption
of pretraining. Recovery uses `latos.training.checkpoint.load_checkpoint` with
`prepare_conversations` train/validation datasets and the unchanged runtime/config.
The verifier exercises that API by replaying the intermediate checkpoint through
the selected final update. It is a verifier, not a merged resumed-run reporting CLI.
The fresh-run runner intentionally refuses existing destinations; handled failures
retain `failure.json` and any completed recovery checkpoints. As in Phase 4, hard
termination or power loss cannot guarantee a failure record.

Strict package/implementation fingerprints mean old Phase 5 optimizer checkpoints
require their original v0.6.0 environment, not the changed Phase 6 package. All old
artifacts are preserved; model-only loading permits the new SFT objective. CPU
recovery is checked for exact equality in the tiny exercise. Actual MPS replay is
compared at atol=1e-6, rtol=1e-5 on this host. Neither implies CUDA determinism or
cross-device equivalence.

Timing synchronizes MPS before/after updates. Update throughput excludes evaluation
and checkpoint I/O; loop throughput includes them. Total recorded runtime includes
preparation, scoring and sample generation, but excludes imports/argument parsing
and final inventory hashing. Peak RSS is process lifetime. MPS allocated/driver
numbers are maximum boundary snapshots, not continuous peaks, and should not be
added to RSS as independent unified-memory pools.
