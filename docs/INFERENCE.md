# Local streaming inference

Phase 7 adds a streaming terminal interface and a session-owned KV cache. It uses
local verified model-only snapshots and the exact Phase 6 chat contract. No server,
network listener, browser, model API, paid service or new dependency is required.
It does not improve the model's language quality. The experimental SFT model scored
0/32 held-out exact replies and worsened English loss in Phase 6; the base remains
preserved separately. See [measured results](../experiments/phase-7/REPORT.md).

## Start a conversation

From the locked environment, select a checkpoint and matching tokenizer explicitly:

```sh
uv run --locked latos chat --model-dir outputs/phase-6-instruction-reviewed/step-00000200/model --tokenizer-dir artifacts/tokenizers/english-bpe-v1 --device mps
```

For the preserved base, use
`--model-dir outputs/phase-5-english-pilot/step-00003000/model`. Neither artifact is
bundled with a clone or wheel. Missing/corrupt inputs, tokenizer mismatch and an
explicit unavailable device fail without downloading or substituting anything.
CPU is the default; `--device auto` opts into selection from available backends.
The interface shows the checkpoint path and measured quality limits. JSON output
also includes verified weight/tokenizer hashes and the resolved device. The known
Phase 6 final SFT is identified by its weight hash, independent of its filename.

Enter one user message per line. `/reset` clears history, retaining the optional
`--system` text. `/exit` or EOF exits. Ctrl-C while replying cancels the reply;
Ctrl-C at the input prompt exits. The generator exposes cancellation between token
steps; one in-progress device operation can finish before cancellation is observed.
The terminal flushes each available text chunk. Control characters other than
newline/tab are shown as escapes so generated escape sequences cannot control the
terminal. JSON events preserve exact decoded text.

A nonempty EOS-completed response enters history. A cancelled, empty, token-capped,
or context-capped reply is shown but leaves history unchanged, including its user
message. This prevents a partial response from silently becoming a completed turn.
A result explicitly says whether history was saved. This conservative rule can make
conversation difficult with checkpoints that do not reliably generate EOS.
History is displayed text, re-encoded as separate header/body segments next turn;
it is not replay of potentially noncanonical sampled BPE token boundaries.

## Limits and sampling

`--max-new-tokens` defaults to 32 and counts EOS. `--context-limit` defaults to the
model's capacity (512 for the retained pilot); set 256 to stay within the prior
experiment's window size. Capacity checks do not establish quality at longer lengths.
Before a turn, the entire shared-format prompt **plus the requested reply budget**
must fit. Otherwise the interface rejects the turn and leaves history unchanged.
It never drops old messages, splits a message, slides positions or truncates text.
Use `/reset`, a shorter message, or a smaller reply budget after a rejection.

Greedy temperature 0 is the default. `--temperature`, `--top-k` and `--seed` use the
existing sampler, including PAD/BOS/UNK suppression and a private CPU RNG. A seed
restarts for each reply. Near ties and backend rounding can change sampled choices;
same-host fixed-case agreement is not general determinism or cross-device equality.
`--no-cache` selects the full-prefix reference path. CPU threads default to 1 and
can be set with `--threads` (1–64). The model is evaluation-only during inference.

For one reply, omit the interactive loop with `--prompt`. Add `--json` for newline-
delimited model, token, optional final text, and done events:

```sh
uv run --locked latos chat --model-dir outputs/phase-5-english-pilot/step-00003000/model --tokenizer-dir artifacts/tokenizers/english-bpe-v1 --prompt 'Describe a quiet garden.' --max-new-tokens 32 --json --device cpu
```

Token events can have empty text while incomplete UTF-8 bytes are buffered. The
final text event flushes incomplete/invalid bytes using the same replacement policy
as completed decoding. Concatenating token/text deltas equals the done event's text
and the tokenizer's completed decoding. Do not concatenate the done text again.
Errors go to stderr with a nonzero exit code; handled stop conditions return zero.
Timings exclude model loading and terminal consumer delay. `prefill_ms` is the first
forward including input transfer and synchronization; `first_token_ms` also includes
sampling and text decoding. Subsequent times have the same first-token scope.

## Cache contract

`latos.inference.KVCache(model).append(ids)` returns logits for the newly appended
positions. It stores rotated keys and unrotated values per layer on the model device.
Queries use absolute rotary offsets. After prefill, the explicit Boolean attention
mask permits key position `k` exactly when `k <= prefix_length + query_index`.
The non-square SDPA causal default alone does not implement this alignment.

The cache is separate from model parameters, buffers and snapshot metadata. It is
owned by one caller with a fixed batch/device/dtype/eval model. Normal parameter
updates and device/dtype moves invalidate it until reset; arbitrary `.data` mutation
is unsupported. Appends check shapes, IDs and total capacity before computation;
failed appends keep prior state. `reset()` drops all layers and resets position zero.
Each generated reply creates a new cache and prefills the full formatted history.
No cache is reused across replies or conversations. There is no shared server state.

Concatenation allocates new layer tensors per append. This is a bounded correctness
implementation, not preallocated/paged serving. `cache_bytes` is logical live K/V
storage, not peak allocator memory. At batch 1 and 512 positions the pilot uses
12,582,912 logical bytes. Temporary concatenation and attention storage add overhead.
No batching scheduler, concurrent model mutation, speculative decoding, compilation,
quantization, mixed precision, or longer-context extension is supported here.

The training forward and parameter names stay compatible. Old model-only snapshots
load unchanged. Optimizer recovery still requires its recorded source/runtime;
Phase 7 is not a migration of Phase 4–6 optimizer checkpoints.

## Reproduce the retained-model checks

Run each in a fresh process, with the preserved ignored inputs available and a new
output name. These checks never open either dataset's test payload:

```sh
uv run --locked python experiments/phase-7/validate.py --artifact base --device cpu --output outputs/phase-7-new/base-cpu.json
uv run --locked python experiments/phase-7/validate.py --artifact base --device mps --output outputs/phase-7-new/base-mps.json
uv run --locked python experiments/phase-7/validate.py --artifact sft --device cpu --output outputs/phase-7-new/sft-cpu.json
uv run --locked python experiments/phase-7/validate.py --artifact sft --device mps --output outputs/phase-7-new/sft-mps.json
```

The runner verifies fixed artifact hashes, compares incremental and chunked prefixes
through capacity, checks changed-future causality and reset, compares greedy outputs,
and measures warmed fixed chat and forced-token workloads. See the
[predeclared plan](../experiments/phase-7/PLAN.md) for tolerances and measurement scope.
Phase 7 CPU/MPS results apply to the measured Mac and its full learned artifacts.
Subsequent owner-reported Windows/RTX 4070 SUPER PASS covered retained tiny base/SFT
artifacts through capacity 256 and a separate synthetic 512-position model. It did
not validate the full Mac learned artifacts or Windows OS-level Ctrl-C/console-event
delivery. Same-CUDA logits met atol=1e-5, rtol=1e-5; fixed greedy IDs and callback
cancellation checks passed. Scoped latency was collected, with no speedup requirement
or direct Mac comparison. The CPU-default pytest suite alone is not CUDA evidence.

Separately inspected hosted Linux CPU CI passed 203 tests with seven MPS-only skips,
including tiny inference/CLI and POSIX SIGINT checks. It did not run the retained-
artifact benchmark matrix or a fresh non-editable wheel suite. Physical Linux remains
deferred. No useful instruction following, general CUDA determinism, cross-device
equality, production latency, mixed precision or distributed serving is established.
See [closure evidence](../experiments/phase-7/CLOSURE.md) for attribution and exact scope.
