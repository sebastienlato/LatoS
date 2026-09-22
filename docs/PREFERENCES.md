# Bounded preference learning

Package 1.2.0 adds a small float32 DPO runner with a fixed SFT reference. This is
an experiment interface for LatoS's synthetic exact-answer tasks, not a general
preference-dataset importer or a useful aligned assistant. See the fixed
[protocol](../experiments/phase-10/PLAN.md), [data notes](../experiments/phase-10/DATA.md),
[results](../experiments/phase-10/REPORT.md) and [review](../experiments/phase-10/REVIEW.md).

The independently implemented objective follows [Rafailov et al., equation 7](https://arxiv.org/html/2305.18290v3).
It averages negative log-sigmoid of the beta-scaled difference between policy and
reference chosen/rejected log ratios. Response probabilities are sums, including
EOS, with one next-token shift. Prompt, padding and earlier assistant turns are
masked. It uses the unchanged chat contract and rejects context overflow.
Variable response lengths can influence sequence probability; this is documented,
not corrected with an undisclosed length normalization. No extra KL term is added.

Both policy and immutable reference start from separate copies of the selected
SFT model. Only policy parameters enter fresh AdamW. Reference parameters remain
frozen and are checked against exact initial tensor values after training.
Outputs require a new directory, retain failure records, and include plan, source,
lock, config, generated preference records, all samples, per-update exposure,
metrics, artifact hashes and a separate model snapshot. These snapshots do not
contain resumable optimizer state. Existing dense recovery and unsupported LoRA
adapter optimizer resume are unchanged. DPO with LoRA, mixed precision, distributed
training, online preference collection and reward models are not implemented.

## Run from retained learned inputs

From the repository root after a locked install:

```sh
uv run --locked python experiments/phase-10/run.py --device mps --output-dir outputs/phase-10-reproduction
uv run --locked python experiments/phase-10/verify.py --device mps --run-dir outputs/phase-10-reproduction --output outputs/phase-10-reproduction-verification.json
```

The wrapper pins the original Phase 6 SFT weights and existing tokenizer/data
paths, and copies the fixed protocol into the run inventory before evaluation.
A source clone does not include those learned artifacts or acquired corpora.
Do not substitute the release's tiny model and call it a full learned reproduction.
The generic `latos preferences --help` exposes explicit input paths and required
weight hash for separately scoped tiny experiments. Its conversation input must
follow the existing verified Phase 6 manifest and exact-answer task convention;
changing domains requires a separate label-design review. `--device cpu` is
available; Windows/CUDA and Linux DPO have not been executed in this phase.

For an offline engineering exercise that builds its own tiny inputs:

```sh
uv run --locked pytest -q tests/test_preferences.py
```

Those tests delete both tiny reserved test files before the runner uses any data.
Production reserved test payloads are only hashed as opaque bytes for preservation.
Neither preference construction nor English/instruction evaluation reads them.
The 32 validation preference pairs reuse the 32 SFT validation prompts: they are
not a second independent benchmark. All reports call them validation accordingly.

Historical platform evidence is separately scoped in [Phase 9 closure](../experiments/phase-9/CLOSURE.md).
The amended merge contract remains atol=rtol=1e-4 and the original CPU 1e-5 failure
remains retained. Cache tolerances remain CPU/CUDA 1e-5 and MPS 1e-4. The owner-reported
Windows tiny CUDA experiment, separately inspected hosted Linux CPU evidence,
deferred physical Linux, unvalidated Windows console Ctrl-C, lack of useful chat,
language-quality limits beyond 256 tokens, general accelerator determinism,
cross-device equality, production performance, mixed precision and distributed
serving limits are unchanged. No new paid services are used.
