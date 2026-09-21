# Reproduce the LatoS v1.0.0 candidate

This guide distinguishes offline mechanism reproduction from the original full
English experiments. The release is prepared locally and awaits publication
approval; `v1.0.0` and its assets do not exist remotely yet. Until publication, use
the exact local candidate commit supplied in the approval checkpoint. After
publication, clone the existing repository and check out the approved tag:

```sh
git clone https://github.com/sebastienlato/LatoS.git
cd LatoS
git checkout v1.0.0
uv sync --locked
uv run --locked latos --version
uv run --locked latos doctor --device cpu --json
```

Repository access is required; visibility is unchanged. Use uv 0.12.15 and the
pinned Python 3.14.7. Initial setup may download the locked interpreter/packages
from their official registries; training itself needs no network or paid service.
The [setup guide](SETUP.md) describes platform prerequisites. macOS arm64, Linux
x86-64 CPU and Windows x86-64 CUDA are the lock's target platforms. An unavailable
explicit device fails; do not infer CUDA access from owner-reported results.
Run commands from the checkout root. Every output destination below must be new.

## Reproduce mechanisms with only tracked inputs

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv run --locked python experiments/phase-4/validate.py --output-dir outputs/release-tiny
uv run --locked latos chat --model-dir outputs/release-tiny/final/model --tokenizer-dir outputs/release-tiny/tokenizer --device cpu --context-limit 64 --prompt 'Hello.' --max-new-tokens 8 --json
```

The full test suite includes tiny pretraining/SFT runner and recovery tests,
assistant-only masks, shared chat prefixes, cache/streaming/stopping and platform-
appropriate cancellation checks. The standalone acceptance command acquires the
original local fixture, prepares it, fits a new tokenizer, trains 400 CPU updates,
and replays updates 98–400 from the update-97 checkpoint. It requires train loss
below 0.25 and below 10% of initial, and exact weights/optimizer/shuffle agreement
on the same tested CPU runtime. `acceptance.json` records actual values, hashes and
source/runtime identity. Its disposable prepared test file is removed before
training. Neither real reserved test payload is required or opened.

The chat command emits model/token/done JSON events. Success means the API works,
not that its response is useful. The tiny model has only 64 positions, so keep the
prompt and reply budget small. See [INFERENCE.md](INFERENCE.md) for history, reset,
EOS, cancellation and full-context rules. On an MPS host, `--device mps` requests
that backend explicitly. No bitwise equality across hosts or backends is promised.

## Install and inspect the downloadable package

Proposed assets: `latos-1.0.0.tar.gz` (source),
`latos-1.0.0-py3-none-any.whl` (LatoS code only),
`latos-1.0.0-tiny-fixture.zip`, and `SHA256SUMS`.
The source archive includes tests, configs, manifests, original fixture inputs,
guides and compact experiment evidence. Dependencies are not bundled. Check the
asset hashes against the approved publication record before use; checksums detect
corruption but are not an authenticated signature.

Start with the source archive or Git checkout to retain platform-specific lock
selection. From that source tree, create the locked environment, then replace
only LatoS with the wheel (put the downloaded wheel at the shown path):

```sh
uv sync --locked
uv pip install --python .venv/bin/python --no-deps --reinstall dist/latos-1.0.0-py3-none-any.whl
.venv/bin/python -m latos --version
```

On Windows, use `.venv/Scripts/python.exe` for the last two commands. Do not use
`uv run` after installing the wheel when verifying the non-editable installation:
it may synchronize the editable project again. Add the environment's `bin` (or
Windows `Scripts`) directory to PATH for subprocess CLI tests. A generic wheel
install by itself does not preserve the lock's platform-specific PyTorch selection.

Extract the tiny zip to a new directory with a normal archive tool. Its paths
are `model/`, `tokenizer/`, `MODEL_CARD.md`, `DATA_CARD.md`, `LICENSE` and
`MANIFEST.json`. Relative card links resolve in the matching source checkout;
the cards are also readable as standalone text. With the installed environment,
select the extracted `model/` and `tokenizer/` explicitly in the chat command above.
The zip includes no optimizer state; reproduce training to exercise recovery.

## Rebuild release assets locally

Build from a clean reviewed checkout, never from a copy of the entire working
folder. Build tools are already in the locked development group:

```sh
uv build --no-build-isolation
uv run --locked python experiments/phase-8/package.py --tiny-dir outputs/release-tiny --output dist/latos-1.0.0-tiny-fixture.zip
```

The packager accepts only the five explicitly pinned tiny artifact files and
three tracked notice/card files. It refuses symlinks, changed identities and an
existing destination. A new training run may differ across systems; the candidate
zip pins this release's exact CPU fixture artifacts. A differing run remains a
mechanism reproduction and must not silently replace the approved asset. Preserve
the release inventory and report the difference. The source/wheel member lists
must be reviewed against tracked content before any upload. The final assets and
SHA256SUMS are stored locally under ignored `dist/phase-8-release/` for approval.

## Full English experiments: additional inputs and original environments

A fresh clone has no acquired book corpus, English tokenizer, full learned weights,
optimizer states or raw logs. These are not downloaded by the CLI and are not in
this release's tiny zip. The [data card](DATA_CARD.md), [model card](MODEL_CARD.md),
Phase 5/6 inventories and original reports identify them. If the owner's retained
inputs are available, integrity-check their hashes first; never refit or overwrite
them for a release check. Both real test sets stay reserved.

For a new full run, acquisition/preparation and tokenizer fitting are documented
in [DATA.md](DATA.md) and [TOKENIZER.md](TOKENIZER.md). Preparation audits all splits
as a data-construction operation; it is not a model test evaluation. Phase 8 does
not rerun this operation on the reserved existing corpora. Follow local source
terms; stop on changed upstream hashes. Do not update a manifest merely to bypass
an integrity failure. Train-only and validation payloads are the experiment inputs.

Use a separate checkout/environment for each historical experiment:

- Base training/recovery: v0.6.0 (`fa0c3ac3b7f3890ffdcad411968a656da2f74b3b`),
  [Phase 5 commands](PRETRAINING.md), fixed 3,000-update configuration and
  500-update validation / 1,000-update checkpoints.
- SFT training/recovery: v0.7.0 (`f6636af34933b9678824cc8dd50f6ab6559a2de5`),
  [Phase 6 commands](INSTRUCTION_TUNING.md), exact retained base/tokenizer,
  generated train/validation conversations and fixed 200 updates.
- Full artifact inference: current release can load both original model-only
  snapshots. Use the explicit model/tokenizer paths in [INFERENCE.md](INFERENCE.md).

Those separate historical installations are required for old optimizer fingerprints;
v1.0.0 does not rewrite their metadata to bypass checks. The original source/runtime
snapshots remain authoritative. Rerunning a full experiment can yield different
weights; the SFT runner deliberately rejects an unrecognized base hash by default.
An intentionally different base requires an explicit new experiment identity, not
substitution into the recorded comparison. No full retraining or historical
optimizer replay was performed in Phase 8. The prior measured outcomes are retained,
not claimed as new fresh-checkout results.

## Publication boundary

Proposed destination: the existing GitHub repository, branch `main`, annotated
`v1.0.0`, and a release titled `LatoS v1.0.0 — reproducible educational release`
with the four named assets. No PyPI upload, dependency vendoring, corpus/full-Mac-
artifact upload, visibility change or paid service is proposed. All prior tags,
especially v0.8.0, remain fixed. [Release notes](../experiments/phase-8/RELEASE_NOTES.md)
and the exact commit/assets are reviewable locally. Nothing is published until
the owner explicitly approves the completed Phase 8 proposal.
