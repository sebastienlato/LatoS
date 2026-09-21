# Bounded LoRA adaptation

Phase 9 adds original float32 attention LoRA and a fixed-budget comparison with
full tuning. See [protocol](../experiments/phase-9/PLAN.md) and
[measured report](../experiments/phase-9/REPORT.md). Existing dependencies remain
locked. There is no adapter optimizer resume, quantization, dropout, adapter
composition, external framework compatibility or distributed training claim.

For a projection with weight W, the adapter computes
`x Wᵀ + (alpha/rank) (x Aᵀ) Bᵀ`. A starts as Gaussian noise (standard deviation
0.02); B starts at zero. Only A and B train. The targets are each block's fused QKV
and attention output projections. The fused QKV adapter shares A across Q, K and V;
it is not three independent adapters. Rank is 1–64 and at most model width.
The tokenizer, embeddings, tied language head, feed-forward layers and norms stay
frozen. Construction copies the base; the caller's model is not mutated.
This implements the published [LoRA method](https://arxiv.org/abs/2106.09685) with
LatoS-specific target choices and independently authored code.

## Run with retained local inputs

A clone does not contain the full learned base, tokenizer or prepared corpora.
Verify the [artifact identities](MODEL_CARD.md) and preserve them. Do not substitute
a randomly initialized model and report it as a learned-base experiment.
The following command runs both methods sequentially in separate processes, with
identical data, batches, configuration and fixed final-update selection:

```sh
uv sync --locked
uv run --locked python experiments/phase-9/run.py \
  --output-dir outputs/my-phase-9-comparison --device mps
```

Use CPU for the portable correctness path. Subsequent owner-reported Windows/CUDA
validation passed only a bounded tiny fixture; see
[closure scope](../experiments/phase-9/CLOSURE.md). Every output directory must be new.
Both methods use
assistant-only next-token labels from shared chat contract 1, the same 200-update
configuration, fresh AdamW and the same shuffle seed. No reserved test payload is
opened by either method. The 32 instruction validation cases and full English
validation are development evidence, not new held-out test results.

For a single run, `latos adapt train --help` lists required inputs. Select
`--method lora` (default) or `--method full`. Set `--base-sha256` to the expected
model weight-file hash. Configuration bounds come from `TrainingConfig`;
the fixed comparison uses [adaptation-pilot.json](../configs/training/adaptation-pilot.json).
Reports include exact runtime/source/config/data identities, exposure, synchronized
update timing, memory observations, all generated samples and artifact hashes.
Failures retain the new directory, plan, source and available metrics; restart in
a new directory. Adapter snapshots contain no optimizer/sampler recovery state.

## Save, load and merge

`LoRAModel`, `LoRAConfig`, `save_adapter`, `load_adapter` and `merge_adapter` are
available from `latos.lora`. The ordinary `Trainer` recognizes `LoRAModel`, checks
its freezing contract and puts only adapter parameters in AdamW. Dense training
retains its all-parameters-trainable contract. Ordinary dense model/checkpoint
writers reject unmerged adapters; never use them as adapter resume checkpoints.

The two-file adapter directory holds safetensors and JSON metadata. Its identity
binds the entire base tensor state and model configuration, including tokenizer
hash. Loading rejects another base even when dimensions match, invalid tensor
names/shapes/dtypes, nonfinite values, size/checksum mismatches and unknown schemas.
Hashes establish integrity against the recorded metadata, not authenticity of
untrusted metadata. Use trusted local artifacts; this is not a signed format.

```sh
uv run --locked latos adapt merge \
  --base outputs/phase-5-english-pilot/step-00003000/model \
  --adapter outputs/my-phase-9-comparison/lora/adapter \
  --output-dir outputs/my-phase-9-merged
uv run --locked latos chat \
  --model-dir outputs/my-phase-9-merged \
  --tokenizer-dir artifacts/tokenizers/english-bpe-v1 \
  --device cpu --prompt 'Copy this word: apple' --max-new-tokens 32
```

Merge creates a separate CPU float32 dense model using `W + (alpha/rank) B A`.
The source adapter and base stay unchanged. Merged models use the existing
model-only format and ordinary chat/cache commands; merging twice creates separate
equivalent outputs, not a second delta on the same model. Floating-point operation
order changes, so numerical agreement is tolerance-based rather than bitwise.
Unmerged adapters also use existing model/cache APIs, but the terminal CLI loads
dense snapshots; merge first for terminal use.

The full comparison is one rank/seed/rate configuration with equal exposure, not an
optimized method ranking. Fewer trainable parameters or lower validation loss do
not establish useful instruction following. Keep the negative Phase 6 SFT result
and all [quality/platform limits](MODEL_CARD.md): Windows historical evidence used
tiny 256-position artifacts plus synthetic 512, not full Mac models; Windows console
Ctrl-C is unvalidated, hosted Linux evidence has its own scope, and physical Linux
is deferred. Phase 9 Windows/CUDA and hosted Linux CPU evidence are recorded
separately in the closure; neither reproduces the full Mac learned experiment.
No full learned 512-position merge on Windows, general accelerator determinism,
cross-device equality, production latency, mixed precision or distributed-serving
claim is made. The tiny trained fraction (~2.44399%) differs from the full pilot
architecture's independently checked 0.851951279%. The 1e-4 merge amendment and
original CPU 1e-5 failure remain preserved.
