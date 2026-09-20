# Phase 5 separate review

Completed 2026-09-20 after implementation and the main run. This was a separate
review pass within the same Work session; no external reviewer is claimed.

## Scope and findings

- Checked the runner against the frozen protocol: seed, initial tensor identity,
  unchanged core model/engine, tokenizer binding, train-only optimization, complete
  validation, no test-file access, final-checkpoint selection, fixed samples,
  exposure accounting, synchronized timing, and preserved checkpoints.
- Checked random/unigram/final comparison alignment: identical validation targets,
  unigram counts derived only from training, and token-weighted scoring. The online
  training curve is labeled separately from the fixed train diagnostic.
- Confirmed the final score is slightly worse than update 2,500 and retained the
  predeclared final checkpoint. The report includes this regression and all four
  qualitative sample failures. No successful assistant claim is made.
- Corrected the sample-stop summary against the retained IDs: the letter sample
  ended on EOS at 60 tokens; the other three reached the 64-token limit.
- Corrected timing descriptions to exclude interpreter/import startup, output
  creation, summary serialization, and inventory hashing. Explicitly labeled MPS
  memory readings as boundary observations rather than continuous peaks.
- Clarified that the first 128 training windows are a diagnostic subset, and that
  256-token training does not establish useful 512-token behavior.
- Historical handoff language is marked as consumed; the public state now records
  the verified closure and this phase's pending approval. No extra Phase 6 start
  approval was introduced and no Phase 6 work was performed.
- Verified privacy across publishable files, historical patches/commit text, and
  historical filenames. The private context remains ignored and untracked.
- The main runner is byte-identical to its executed local copy. Runtime, tokenizer,
  corpus-window, source-base, lock, and artifact hashes are retained. The source
  working tree was dirty during the run and is explicitly reported as such.

No actionable implementation or evidence issue remains open.

## Actual validation

- Locked development suite: **164 passed, zero skips/failures**. Includes existing
  CPU/MPS numerical/recovery tests and three new pilot checks: train-only unigram
  scoring; full execution with test text absent and overwrite refusal; deliberate
  update failure preserving the initial recovery checkpoint.
- Fresh environment with locked dependencies and a non-editable wheel install:
  **164 passed, zero skips/failures**. Import location was verified inside that
  environment's installed package, not the working source tree.
- Independent pilot verification: **25 files** rehashed; **3,000 updates** and total
  exposure reconciled; initial weights equal fresh seed-17 initialization; initial
  and final full validation reproduced; trained recovery update 2,001 replayed
  with identical scalar loss/exposure on this host. Scope is recorded explicitly.
- MPS environment doctor, lint, formatting, and whitespace checks passed.
- Source distribution and wheel builds passed. Archive contents were inspected
  for private context, acquired text, learned artifacts, checkpoints, environments,
  logs, unexpected executable payloads, and private identifiers.
- Original random weight and tokenizer hashes remain unchanged. No dependency
  versions changed; the lockfile changes only the LatoS root version to 0.6.0.

## Failures and corrections retained

- Initial direct test invocation lacked the environment's CLI in PATH: seven CLI
  tests failed to launch while 157 tests passed. Running the documented locked
  environment command resolved all seven; no product-code patch was needed.
  Both logs remain under ignored `outputs/phase-5-*`.
- An offline free-resolution wheel-install attempt could not resolve uncached
  package-index metadata. Installing dependencies from the existing exact lock,
  then installing the built wheel without dependency resolution, succeeded using
  cached packages. No new download, dependency substitution, or paid service.
- Initial runner lint found long lines, corrected before the main run. Main pilot
  execution itself had no interruption, nonfinite loss, OOM, or retry.

Full Phase 5 training was MPS-only. Earlier owner-reported Windows/CUDA PASS and
verified hosted Linux CPU CI remain Phase 4 evidence. Physical Linux is deferred,
not performed. No new cross-platform or general determinism guarantee is claimed.
