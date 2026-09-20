# Phase 6 measured English instruction-tuning experiment

2026-09-20. **Engineering validation passes; useful instruction following does not.**
The reviewed 200-update run completes from the selected Phase 5 base with a fresh
optimizer. Assistant validation loss improves, but held-out exact completion stays
at zero and ordinary English validation regresses. The tuned checkpoint is retained
as a negative experiment artifact; it does not replace the accepted base.

## Fixed comparison

| Measure | Preserved base | Final SFT |
| --- | ---: | ---: |
| Assistant-only validation loss, 121 targets | 8.354879 | 5.815233 |
| Assistant-only validation perplexity | 4250.8683 | 335.3694 |
| Exact final replies | 0 / 32 | 0 / 32 |
| Replies stopping at EOS | 0 / 32 | 32 / 32 |
| Copy / extraction / first letter / recall | 0/8 each | 0/8 each |
| English all-target validation loss, 111,523 targets | 4.731898 | 5.653422 |
| English all-target validation perplexity | 113.5109 | 285.2661 |

English loss increases by **0.921524** on the same full validation set and tokenizer.
Assistant-only loss is a different objective from all-target pretraining loss.
The large fall in assistant loss is not task success: the model often substitutes
training words for the unseen requested word, such as answering `window` when
asked to copy `anchor`. All 64 base/SFT samples and prompts appear in
[samples.json](samples.json); none were selected for favorable quality.

The source-group split holds out eight new words in familiar templates. It is not
an unseen-task, natural-dialogue, safety, or broad knowledge benchmark. Recall uses
gold assistant history and checks only the final continuation, not an autonomous
conversation. Each exact match strips outer whitespace but otherwise preserves case
and punctuation. Greedy generation uses a fixed 32-token cap and shared formatter.
No instruction or English test predictions/loss were computed.

## Inputs and budget

- Phase 5 closure verified on remote main before starting:
  `3878a0167a78f6ad149e8523b4feb97cda77ef5e`. Annotated v0.6.0 remains at
  `fa0c3ac3b7f3890ffdcad411968a656da2f74b3b`.
- Selected base: `outputs/phase-5-english-pilot/step-00003000/model/`, SHA-256
  `f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa`.
  The initial SFT checkpoint is tensor-for-tensor equal to this model.
- Tokenizer: existing 8,192-entry English BPE, SHA-256
  `7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab`.
- Original deterministic Codex-authored English templates: 192 training, 32
  validation and 32 reserved test conversations; source word groups disjoint.
  All related tasks for a word stay in its split. Shared templates and generic
  responses overlap intentionally. Synthetic provenance/rights are documented in
  the generator and [guide](../../docs/INSTRUCTION_TUNING.md).
- Fresh AdamW, seed 61, float32, batch 4, accumulation 2, 256-token maximum,
  200 updates, peak LR 0.0003, ten warmup updates and cosine decay to 10%, weight
  decay 0.01, gradient clipping 1.0. No hyperparameter search.
- **6,188 assistant target exposures**, 1,600 conversation exposures, all 744
  distinct training target positions seen, approximately 8.317 dataset-equivalent
  target exposures. Counts include assistant EOS and repeated exposure.
- Final update 200 was chosen before running. Earlier validation measurements
  were better; the final checkpoint remains selected. This behavior is consistent
  with overfitting this tiny corpus, not evidence of improved lexical generalization.

## Host and timing

Actual Mac16,6 / Apple M4 Max, 64 GiB, macOS 26.6.2; Python 3.14.7, PyTorch 2.14.0,
one CPU thread, MPS, float32, no MPS fallback. CUDA is unavailable. Initial free
disk was approximately 627 GiB. No new paid service or external GPU was used.
The retained environment and dependency versions are unchanged except LatoS 0.7.0.

The successful ten-update calibration took 2.199 seconds including setup and
314 targets, with 1.491 synchronized update seconds. It was discarded as a model.
The reviewed full run took **25.902 seconds before inventory hashing**, including
14.152 synchronized update seconds and 16.013 loop seconds including validation
and checkpointing. Throughput was 437.25 assistant targets/s for updates and
386.44 targets/s for the loop. These tiny-workload numbers are not sustained
large-training performance estimates.

Process lifetime peak RSS: 1,383,251,968 bytes. Maximum observed MPS allocated:
217,526,784 bytes; driver: 1,216,249,856 bytes. MPS figures are boundary snapshots,
not continuous peak memory; they must not be added to RSS as independent physical
memory usage on this unified-memory host.

## Validation and separate review

- Complete development suite: **187 passed**, including 23 added Phase 6 checks.
  Coverage includes explicit preceding-logit alignment, zero gradient on ignored
  prediction positions, padding invariance, both assistant turns/EOS, literal token
  spellings, exact generation prefixes, invalid roles, empty targets, overflow
  rejection, target-weighted accumulation, data isolation and integrity, recovery
  identities, runner failure retention and no-overwrite behavior.
- Tiny masked-objective CPU recovery is exact on this runtime. Local MPS smoke
  recovery passes at atol=1e-6, rtol=1e-5. Complete tiny CPU runner and verifier
  work when both generated test files are absent.
- Independent process verifier: **49 artifact files hashed**, all 200 updates and
  6,188 targets accounted for, initial weights equal selected base, both objectives
  recomputed with zero recorded loss difference, and all **64 generated responses
  replayed exactly**. Update 100→200 MPS recovery has maximum weight difference
  1.1920928955078125e-7 and passes atol=1e-6, rtol=1e-5.
- Review corrected masked-data boundary metadata and expanded response verification;
  the reviewed run also retains its exact package Python source. The identical
  predefined experiment was repeated from the original base after these fixes.
  The earlier successful run is retained; its assistant loss was 5.815108 and
  English loss 5.653430, also 0/32 exact. Small run differences reinforce that
  general MPS determinism is not claimed. Neither result was used to select a
  better checkpoint or change hyperparameters.
- Fresh non-editable wheel suite and lint/build/privacy results are recorded in
  [validation.json](validation.json). See [separate review](REVIEW.md).
- **19,806 pre-existing files** in retained Phase 4/5 outputs, the tokenizer and
  random-baseline directories were rehashed unchanged. The committed Phase 5
  inventory and selected base/tokenizer identities verified. Existing English
  corpus data was not modified; English test text was not opened by tuning.

Subsequent independent Windows/CUDA engineering validation is **PASS**, owner-reported
at exact v0.7.0: 184 tests passed, three MPS-only skips, fresh wheel suite, four CUDA
updates / 175 assistant-target exposures, 49 artifact checks and 64 response replays.
Both tiny Windows models scored 0/32. This did not reproduce the full Mac experiment
or learned artifact. Hosted Linux CPU CI was separately verified with 184 passed /
three skips, within its tiny CPU SFT/workflow scope. Physical Linux remains deferred.
No useful instruction following, general CUDA determinism, cross-device equality,
mixed-precision/distributed behavior or broader capability follows from these checks.
[Closure evidence](CLOSURE.md) records attribution and scope; original result JSON,
samples, inventory and the pre-publication validation snapshot remain unchanged.

## Retained artifacts and failures

The reviewed run is `outputs/phase-6-instruction-reviewed/`; its selected model is
`step-00000200/model/`, weight SHA-256
`62b890214e5544292cc54b725a4e505730fca0d501b3337cdd3f051a622d09ae`.
Initial/update 100/update 200 checkpoints retain full optimizer/shuffle state.
[artifacts.json](artifacts.json) hashes the completed run; inventory SHA-256
`f0d88d3de0ab3da03d7c9f8734aea3de24f7191afc9ea76499d5c7bfccda01c7`.
[verification.json](verification.json) and [results.json](results.json) contain
machine-readable evidence. Exact package implementation fingerprint:
`d1fee8c14d22e4662fbfb554fb5f1bda13f45dffae19d7d5ca4760d0c57e36dc`.
Git provenance correctly records the Phase 5 closure parent with dirty local
Phase 6 changes; preserved source files bind the actual implementation used.

An initial calibration attempt failed before optimization because the generator
added a redundant newline to JSONL records. It is preserved at
`outputs/phase-6-calibration/failure.json`, with the draft data at
`data/processed/english-instructions-v1-draft/`. The corrected data and calibration
are in `data/processed/english-instructions-v1/` and `outputs/phase-6-calibration-2/`.
The initial successful pilot and verifier remain at `outputs/phase-6-instruction-pilot/`
and `outputs/phase-6-verification.json`; its package source was preserved separately
in `outputs/phase-6-pre-review-source/` before the metadata fix. No training failure
occurred in either complete pilot. No artifact here is a remote backup.

A direct development test invocation initially lacked the installed CLI on PATH;
using the documented locked launcher resolved it. A new numerical accumulation
check initially used the engine's higher default LR and saw a 9.44e-7 batching
roundoff difference in one parameter. It now uses the actual SFT LR (0.0003),
retaining the original tolerance; explicit cross-entropy/gradient alignment checks
pass independently. All earlier diagnostics remain local, not hidden as success.

The implementation was published after approval as main and annotated v0.7.0 at
`f6636af34933b9678824cc8dd50f6ab6559a2de5`. The documentation-only closure now awaits
its own approval, with no new tag, tag movement, release or artifact upload.
Phase 7 must not begin in this chat, including after closure publication.
