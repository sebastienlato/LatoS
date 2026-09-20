# Phase 5 closure

Recorded 2026-09-20. Phase 5 implementation and validation are closed locally.
This documentation-only closure awaits explicit publication approval. **Do not
begin Phase 6 in this chat**, including after closure publication.

## Validated checkpoint

Tag `v0.6.0`, commit `fa0c3ac3b7f3890ffdcad411968a656da2f74b3b`, package 0.6.0.
The implementation was published after owner approval. Remote main and the peeled
annotated tag were checked again before closure; both resolve to that exact commit.
The tag object is `28f0468fff6e103a71f15078069ce7c17ea42b16`. Keep v0.6.0 fixed;
v0.5.0 also remains unchanged at `cb585c321c92f5d774fb59234f76c1d3783a635a`.

No runtime source, tests, experiment scripts, configurations, dependency lock,
package version, or CI workflow changes belong to this closure. Original pilot
metrics, samples and inventories remain unchanged.

## Evidence by environment

| Environment | Evidence source and result | Actual scope |
| --- | --- | --- |
| Local Mac CPU/MPS | Previously recorded: 164 tests passed in development and fresh wheel installation | Full 3,000-update English pilot, 5,761,229 target exposures; artifact and trained recovery checks |
| Independent Windows / RTX 4070 SUPER | **PASS**, owner's external summary for exact v0.6.0 | 162 passed, two MPS-only skips, zero failures; fresh wheel suite; bounded four-update CUDA exercise, 416 target exposures |
| GitHub-hosted Linux CPU | **PASS**, completed run metadata, exact-commit workflow and logs directly inspected | 162 passed, two MPS-only skips; tiny CPU Phase 5 tests and Phase 4 fixture acceptance at the Phase 5 commit |
| Independent physical Linux | **DEFERRED — NOT PERFORMED** | Hosted CI does not replace this validation |

### Owner-reported Windows/CUDA PASS

The owner reports all of the following at the unchanged validated commit:

- Fresh checkout and locked installation passed; 162 tests passed, two MPS-only
  skips, zero failures. The complete suite also passed from a fresh non-editable
  wheel installation.
- Actual training and generation ran on an NVIDIA RTX 4070 SUPER. The unchanged
  Phase 5 runner completed a bounded **four-update, 416-target-exposure** exercise.
- Forward/backward, AdamW updates, accumulation, scheduling, finite numerical
  state, and validation-state preservation passed.
- Fixed samples and CUDA generation passed, as did checkpoint creation/loading,
  independent verification, and separate-process CUDA recovery.
- All **20 artifact inventory hashes** passed. Synchronized optimization timing
  and CUDA memory measurement mechanisms passed.
- Reserved test data remained preserved and was not used for training or model
  selection. CLI, lint, formatting, dependency integrity, offline data/tokenizer
  workflows, builds, packaging and privacy checks passed.
- All **108 tracked files** remained unchanged, the working tree stayed clean,
  and no commits or pushes were made from Windows.

This is owner-reported independent evidence, not CUDA execution by this Mac
session. Raw Windows logs, driver/runtime details, and numerical performance
measurements were not supplied here; none are inferred.

The full **5.76-million-target Mac English pilot was intentionally not reproduced
on Windows**. Backend validation does not require rerunning that full pilot or
matching its exact scores. This bounded exercise verifies the reported mechanisms;
it establishes no general CUDA determinism, cross-device numerical equivalence,
sustained-pilot performance, language-quality, instruction-following, or Phase 6
capability claim. No new exact CUDA equality result is inferred from this PASS.

### GitHub Linux CPU CI verified separately

[Run 35515094307](https://github.com/sebastienlato/LatoS/actions/runs/35515094307)
completed successfully at `fa0c3ac3b7f3890ffdcad411968a656da2f74b3b` on 2026-09-20;
the CPU job completed at 14:00:13 UTC. The inspected workflow targets Ubuntu 24.04;
logs identify Python 3.14.7 and PyTorch 2.14.0+cpu. The suite collected 164 tests:
**162 passed, two MPS-only skips, zero failures**.

Successful steps were locked setup, Ruff lint/format, pytest, CPU doctor and debug
model check, offline data acquisition/preparation/audit, source/wheel builds,
offline tokenizer fitting/validation, and offline CPU training acceptance.

The three Phase 5 tests passed: train-only unigram scoring; a **two-update tiny CPU
pilot** with test text absent, baseline/final samples, checkpoint/inventory checks
and overwrite refusal; and an injected update failure preserving initial recovery.
The broader suite includes separate-process CLI recovery. The explicit training
acceptance step invokes `experiments/phase-4/validate.py`, at the Phase 5 commit:

- 127,808 parameters, 400 updates, 131,600 target exposures.
- Training loss 5.8146375958 → 0.0100304254; held-out fixture loss
  5.8069872746 → 11.2483315773. This is fixture memorization, not language improvement.
- Resume from update 97 matched all 303 remaining non-timing update metrics and
  final weights, optimizer and sampler state within that CPU run.
- Validation preserved weights, optimizer, gradients and RNG, and restored mode.

This CI did **not** reproduce the full English pilot, execute CUDA/MPS, validate
an independent physical Linux machine, or establish sustained-pilot quality or
performance. Building a wheel is not a fresh non-editable wheel test run. Standalone
Phase 5 `verify.py` execution and separate archive/privacy inspections are not
steps in this workflow and are not claimed for CI. Tiny test-fixture preparation
and auditing are distinct from training on or selecting against reserved test text.

## Closure decision and remaining limits

The measured English pilot and the requested independent Windows/CUDA validation
have passed within their declared scopes. No unresolved Phase 5 failure is reported.
The Mac pilot remains the full-run evidence: validation loss 9.089003 → 4.731898,
fixed final update 3,000 selected despite the slight regression after update 2,500,
and repetitive/incoherent samples. See [REPORT.md](REPORT.md), [results.json](results.json),
and [samples.json](samples.json). External checks do not expand those quality claims.

Physical Linux remains explicitly deferred. Existing local checkpoints, tokenizer,
corpus and logs stay preserved and ignored. External GPU validation is not access
to that machine for subsequent training. No paid service or artifact upload is
part of closure. [closure.json](closure.json) is the structured evidence summary;
earlier validation JSON files retain their original pre-publication scope.

## Separate closure review and publication

A separate documentation review checked owner-report attribution, exact refs and
counts, actual CI step scope, artifact identities, consistent limits, links,
privacy, and the fresh-chat gate. All 25 pilot inventory files, the selected weights,
random baseline and tokenizer were rehashed locally with their recorded identities
unchanged. JSON parsing, relative links, whitespace and privacy checks passed.
No actionable closure finding remains open. No new training, runtime test
suite, or build result is claimed for this documentation-only closure.

Publish only the reviewed closure commit to existing private
`https://github.com/sebastienlato/LatoS.git`, branch `main`, after explicit approval.
**No new tag or tag movement.** Verify the exact remote closure commit and unchanged
v0.6.0/v0.5.0 after publication, record verification locally, and stop.

Phase 6 requires explicit owner start in a **fresh Work chat after closure
publication is verified**. [PHASE6_HANDOFF.md](../../docs/PHASE6_HANDOFF.md) preserves
the inputs and reusable prompt. It is preparation, not authorization to start now.
