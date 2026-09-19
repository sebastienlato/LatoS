# Project state

Updated: 2026-09-19.

## Active phase

Phase 3 — fixture checkout correction complete locally; push approval pending.
Phase 4 remains paused and is not authorized.

## External Windows result

The owner tested v0.4.1 at `14c76284d1ec670e4022584638a5030a9da87140` on Windows 11
Home 25H2, Python 3.14.7, PyTorch 2.14.0+cu130, RTX 4070 SUPER, driver 616.92,
and PyTorch CUDA runtime 13.0. Locked install and GPU detection passed. Pytest
stopped at its first setup error after nine passes: fixture-test size/hash mismatch.
System core.autocrlf=true converted seven LF endings, changing test.txt from 864
to 871 bytes. Committed blobs matched the manifest. The owner reports no edits,
commits, or pushes. CUDA tensor/model execution is still NOT VALIDATED.

## Correction and evidence

- Package 0.4.2 adds root .gitattributes with text eol=lf only for fixture .txt and
  .json paths. The attributes are also included in the source distribution.
- All three fixture payloads and their manifest remain byte-identical to v0.4.1.
  Original size/hash verification is unchanged. No src/, model settings, numeric
  tolerances, dependency pins, index routes, or CI configuration changed.
  uv.lock differs only in the LatoS package version.
- Original 871-byte CRLF failure reproduced with real Git on Mac. Candidate rules
  preserve 1,062 / 613 / 864 bytes under autocrlf true, input, and false. An unrelated
  control still converts under true; intentionally damaged fixtures still fail checksums.
- 132 tests passed on Mac and in a fresh installed-wheel environment running tests
  from the source archive. Lint, format, package builds, fixture/archive hash checks,
  and dependency compatibility pass. The originally failing test also passed from
  an isolated full-tree checkout using autocrlf true and the candidate attributes.
- Separate review complete; repository-wide renormalization, runtime normalization,
  manifest changes, and changes to user Git settings were avoided.
- Evidence: [checkout correction](experiments/phase-3/CHECKOUT_CORRECTION.md).
  The native Windows correction retest and external Linux validation remain pending.

## Publication history and proposed correction

Private origin: https://github.com/sebastienlato/LatoS.git.
Published Phase 3: deed6e9b0303967dd82f7a3205e03b9c4f05ffe7 at v0.4.0, then the
Windows dependency correction 14c76284d1ec670e4022584638a5030a9da87140 at v0.4.1.
Both were approved and verified remotely. Neither tag may be moved.

- Local branch: main; message: Fix Phase 3 byte-pinned fixture checkout.
  This state belongs to the correction commit; its exact ID is supplied in the
  approval request and available from git log -1 --format=%H at this checkpoint.
- Proposal: push only the reviewed correction to main and create/push annotated
  tag v0.4.2. No release, dataset, tokenizer, or weight upload.
- Correction approval: PENDING; nothing from this correction has been pushed.
- Preserve ignored checkpoints/phase-3-pilot-initial/, the accepted tokenizer under
  artifacts/tokenizers/english-bpe-v1/, and original raw/prepared corpora.

## Next action and external gate

Stop for approval. If approved, publish this exact commit/tag, verify the remote
references, then PAUSE for a fresh Windows/RTX 4070 SUPER retest and native Linux
results. Validation machines must not fix, commit, or push files. Start Phase 4
only when both platforms pass AND the owner explicitly authorizes it. A push
approval, CUDA availability, or CI success does not authorize Phase 4.

Local uv: .private/tools/bin/uv; .venv uses the runtime under .private/python.
