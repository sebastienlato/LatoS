# LatoS v1.0.0 — reproducible educational release

Local candidate; publication awaits explicit approval.

This milestone packages the independently implemented English-first small-model
workflow: documented data preparation, train-only BPE fitting, dense transformer,
training/recovery, assistant-only SFT and local streaming/KV-cache inference.
The release adds a tested reproduction guide, data/model cards, consolidated
experiment evidence and inspected artifacts. No model-quality improvement is
claimed and no core runtime or dependency version changes in this phase.

The full 17.3M base improved English validation loss from 9.089003 to 4.731898.
Experimental SFT lowered assistant loss from 8.354879 to 5.815233 but remained
0/32 exact replies and worsened English loss to 5.653422. Keep these models distinct.
The downloadable tiny fixture model is a separate memorization/mechanics example.

Assets: source archive, Python wheel, tiny fixture model/tokenizer zip, and
SHA256SUMS. Original code and the tiny fixture artifacts are MIT. Dependencies keep
their own licenses and are not bundled. Acquired books, prepared corpora, full Mac
learned artifacts, optimizer states, logs and private context are excluded. Their
identities and reproduction requirements are documented; this is not their backup.

Mac CPU/MPS evidence includes full retained-model checks. Historical Windows/CUDA
PASS is owner-reported for tiny 256-position artifacts plus a synthetic 512-position
model; the full Mac learned artifacts were not tested there, and Windows console
Ctrl-C remains unvalidated. Hosted Linux CPU CI passed separately within its tiny
CPU/workflow scope. Physical Linux remains deferred. Phase 8 adds no Windows/Linux
execution claim. Useful instruction following, general CUDA determinism, cross-device
equality, production latency, 512-token language quality, mixed precision and
distributed serving remain unestablished. Both real test sets remain reserved.

Use docs/RELEASE.md, docs/DATA_CARD.md, docs/MODEL_CARD.md and
experiments/phase-8/REPORT.md in the matching source archive or checkout.
All earlier tags, including v0.8.0, remain unchanged.
