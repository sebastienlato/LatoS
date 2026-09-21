# Phase 8 separate review and fixes

Reviewed 2026-09-21 after implementation, in a separate read-only inspection pass
followed by fixes and targeted rechecks. Performed locally; no independent reviewer
or external platform execution is claimed.

## Findings resolved

1. The first model-card draft confused requested vocabulary size 512 with the tiny
   tokenizer's actual 328 entries. Corrected against generated metadata and model
   configuration; the 127,808-parameter count is unchanged.
2. Historical state called the remote private. A fresh read-only GitHub lookup
   reports public. Current state/report now disclose the observed visibility and
   propose no change. Earlier phase evidence is preserved as historical snapshots.
3. Packaging/reproduction instructions now explicitly distinguish the wheel's
   platform-neutral LatoS code from platform-specific locked dependencies, and warn
   that `uv run` may restore the editable installation during wheel verification.
   Actual wheel tests invoke the installed interpreter with its CLI on PATH.

## Reviewed evidence and boundaries

- Runtime source and shared chat formatter are unchanged from the Phase 7 closure;
  only package version and source distribution scope change in package metadata.
  The dependency lock differs only in the LatoS version.
- Tiny packager uses a literal destination/source allowlist, verifies all five
  learned-file hashes and sizes, rejects symlinked files/parents, writes fixed zip
  metadata, excludes extra files and refuses an existing destination. Negative
  tests exercise corrupt inputs, extra inventory entries and symlinked directories.
- Source archive members match the isolated tracked checkout byte-for-byte; the
  wheel's package files match its source. No untracked folder archive, dependencies,
  acquired corpus, full learned artifacts, optimizer state, logs or private content
  are proposed. Tiny zip contents and manifest hashes are checked separately.
- Fresh-checkout guide tested with a new locked environment and tracked inputs only;
  tiny acceptance and JSON chat work without either real test set. Five generated
  artifact identities agree across the two local reproductions.
- Development, isolated-checkout and non-editable wheel suites each pass 213 tests.
  Full retained Mac base/SFT CPU/MPS model-only chat smokes pass. There is no new
  full experiment, historical optimizer recovery or performance comparison claim.
- Data/model cards and release notes preserve the negative SFT result, historical
  source terms, intended educational scope, actual platform attribution, Windows
  tiny/synthetic artifact boundary and unvalidated console Ctrl-C, separate hosted
  Linux scope and deferred physical Linux. No CUDA determinism, cross-device
  equality or useful assistant claim is introduced.
- Prior artifacts and both reserved test payloads are unchanged under 441 distinct
  preservation hashes. Test payloads were not parsed/evaluated. Metadata and all
  old attempts/failures remain available locally; release assets do not back them up.
- Relative documentation links, JSON validity, staged filenames/content, repository
  history and outgoing publication text were inspected for private planning content.
  All release assets require their own exact hash record before publication.

No remaining actionable finding was identified within this review's scope. Final
local-commit clone validation and asset identities are recorded outside Git to avoid
self-referential archive hashes. Publication approval remains pending.
