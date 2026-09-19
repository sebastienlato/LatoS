# Phase 3 fixture checkout correction

Historical correction evidence. The owner subsequently reported v0.4.2 Windows/CUDA
PASS and deferred physical Linux testing. See [CLOSURE.md](CLOSURE.md) for the current
status and Phase 4 hold; this supersedes the pending validation instructions below.

Recorded 2026-09-19. Base checkpoint:
`14c76284d1ec670e4022584638a5030a9da87140`, tag `v0.4.1`.
Local package version: 0.4.2. Phase 4 remains paused.

## External evidence

The owner reports successful locked installation of Python 3.14.7 / PyTorch
2.14.0+cu130 on Windows 11 Home 25H2, with driver 616.92 and CUDA runtime 13.0.
CUDA availability returned true and the RTX 4070 SUPER was detected. The earlier
dependency blocker is resolved, but this is not a CUDA tensor/model execution pass.

Pytest stopped after nine passes with a setup error in
`test_repeatable_preparation_and_no_split_overlap`: the `fixture-test` raw size
or checksum did not match. Read-only Windows investigation found system
`core.autocrlf=true`. The Git blob and manifest both specify 864 bytes for
`test.txt`; the checkout contained 871 bytes, from seven LF-to-CRLF conversions.
The owner reports no edits, commits, or pushes from that validation machine.

## Root cause and correction

The repository had no checkout attributes. Git could therefore rewrite text
line endings according to machine configuration before LatoS acquired the fixture.
LatoS correctly rejected those changed bytes before cleaning.

The new root `.gitattributes` sets `text eol=lf` only for
`/data/fixtures/**/*.txt` and `/data/fixtures/**/*.json`. The former protects the
three payloads; the latter preserves the fixture manifest's own provenance bytes.
These rules travel with fresh clones and are included in the source distribution.
Git's [attribute documentation](https://git-scm.com/docs/gitattributes#_checking_out_and_checking_in)
defines this per-path checkout behavior independently of `core.autocrlf`.

No fixture payload or manifest byte, size, or hash changed. No runtime source code,
model setting, dependency version/source/hash, CUDA routing, or numerical tolerance
changed. The lock diff updates only the project version. No checksum fallback,
pre-verification newline conversion, global Git reconfiguration, or repository-wide
renormalization was introduced. Unrelated text files retain their prior Git policy.

## Validation and review

- The original failure was reproduced in an isolated local clone of the exact
  v0.4.1 commit on Mac, checking out with `core.autocrlf=true`: 871 bytes and seven
  CRLF pairs for `test.txt`.
- A new Git-backed test writes the actual fixture inputs and candidate attributes
  to an isolated index, checks them out into an empty directory, and compares
  every payload plus manifest against index bytes. All manifest sizes/hashes match.
- Four combinations pass: autocrlf true/eol crlf, input/crlf, false/crlf, and false/lf.
  An unrelated control file converts to CRLF under autocrlf true, proving that
  conversion is active while fixture bytes remain LF. With text unspecified and
  autocrlf false, the unrelated control is not normalized by core.eol alone.
- Acquisition, preparation, and audit pass from each resulting checkout, yielding
  3 train / 3 validation / 4 test records. Deliberately converting a checked-out
  fixture to CRLF still fails the original raw-byte verification without changing
  its manifest.
- The complete Mac regression suite passes: **132 tests**, including existing MPS,
  model, tokenizer, data, and platform-lock checks. Lint/format and dependency
  compatibility checks also pass.
- A fresh locked environment installed the built wheel and ran all **132 tests**
  from the extracted source distribution, outside the development checkout. The
  archive contains `.gitattributes` and the exact original fixture bytes/hashes.
- A second local clone of the complete base tree was populated with the candidate
  attributes in its index before checkout. With autocrlf true, fixture payloads
  stayed at 1,062 / 613 / 864 bytes while the unrelated README converted to CRLF.
  The originally failing test passed from that checkout using the installed wheel.

Byte counts, hashes, Git version, and validation boundaries are recorded in
[checkout-correction.json](checkout-correction.json).

Tests invoke the installed Git executable and isolate system/global settings and
the index; they do not edit the user's Git configuration. No identity or Git commit
is needed for their temporary repositories. The source archive includes the policy
so the same tests work outside a Git checkout.

The separate review checked attribute scope, unchanged pinned inputs and verifier,
negative corruption checks, packaging inclusion, and the publication/pause gate.
This validates Git's conversion mechanism on Mac, not native Windows execution.
The user's next fresh Windows checkout remains the native acceptance check.

## Publication and next action

Proposed tag: `v0.4.2` at the new reviewed correction commit. Keep `v0.4.0` and
`v0.4.1` fixed. Stop for approval before any push. After approved publication,
verify the exact remote references and pause for fresh Windows/RTX 4070 SUPER and
native Linux results. Phase 4 requires both passes and explicit owner authorization.
Retest instructions are in [SETUP.md](../../docs/SETUP.md#windows-validation-powershell).
