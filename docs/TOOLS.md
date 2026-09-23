# Bounded structured tools

Phase 11 adds an experimental local protocol, two pure tools, a capped conversation
runner, and separate protocol and task evaluation. Package 1.3.0. This is an
engineering interface: unchanged base/SFT/DPO models did not produce valid calls
in the [fixed experiment](../experiments/phase-11/REPORT.md). It does not establish
learned tool use, useful assistant quality, safety alignment or production readiness.

## Commands

```sh
uv run --locked latos tools schema
uv run --locked latos tools execute --call '{"tool":"add","args":{"a":2,"b":3}}'
uv run --locked latos tools execute --call '{"tool":"lookup","args":{"key":"oak"}}'
uv run --locked latos tools run --model-dir outputs/phase-6-instruction-reviewed/step-00000200/model --tokenizer-dir artifacts/tokenizers/english-bpe-v1 --prompt 'Use add to sum 2 and 3.' --device mps
```

`execute` returns a JSON object such as `{"ok":true,"value":5}`. Missing catalog
keys return `{"ok":false,"error":"not_found"}` and exit 1. Invalid input exits 1.
`run` emits the complete bounded session record; exit 0 means a final envelope was
received, **not that its answer is correct**. Reported errors, exhausted budgets,
context limits, incomplete generation and generation errors exit 1; Ctrl-C exits
130. Windows console Ctrl-C remains unvalidated. The runner supports local dense
snapshots and never saves or trains a model. For tiny artifacts whose capacity is
less than 256, use the Python responder with an explicit smaller context budget;
the CLI defaults require capacity at least 256.

## Protocol version 1

Exactly one JSON object per assistant turn, with exactly one of these key sets:

- `{"tool":"add","args":{"a":2,"b":3}}`: two exact integers in [-1000,1000].
- `{"tool":"lookup","args":{"key":"oak"}}`: 1–16 lowercase ASCII letters.
- `{"final":"5"}` or `{"error":"not_found"}`: 1–128 printable characters.

The catalog is fixed: oak=7, elm=11, ash=13. Unknown keys return `not_found`;
unknown tool names and invalid arguments never execute. Keys are not filesystem
paths. There is no shell, expression evaluator, network, mutable registry or file
access in either tool. The dispatcher is independently written and uses the Python
standard library, with no added dependencies. `tools schema` exposes the versioned
protocol description; this is a LatoS contract, not an external API compatibility
claim or a general JSON Schema implementation.

Before decoding, reject responses over 1,024 UTF-8 bytes or nesting deeper than
two containers. Reject duplicate keys, nonfinite numbers, extra top-level values,
Markdown fences, and extra fields. Do not coerce bool, float or strings to integers.
Raw trace storage is also capped at 1,024 characters per response and marks truncation;
ordinary model outputs are retained in full. JSON parsing and envelope validation
are distinct from argument schema validation and task correctness.

At most four assistant generations and two actual tool executions occur. Invalid
requests consume a turn; a missing lookup consumes a call. Retrying receives a
structured error. Calls after the execution budget are not executed. Cancellation
before generation, after generation or before execution stops the session. A valid
JSON fragment at a token limit is still incomplete: only EOS-terminated responses
can execute or become final answers. These bounds apply to the built-in model
responder; arbitrary Python callbacks are trusted code, not sandboxed or timed out.

The original tokenizer and **chat contract 1 are unchanged**. Each assistant call
is followed by an ordinary user message containing `{"tool_result":...}`. The host
constructs this message; it is not a new privileged role or protection against prompt
injection. Every generation uses the full history and reserves its output budget.
Overflow returns `context_limit`, with no history truncation. Default generation
is greedy, cached, seed 0, 64 new tokens, 256 total tokens. A context failure during
retry is distinct from invalid JSON and from a model reporting a tool failure.

## Evaluation and reproduction

```sh
uv run --locked python experiments/phase-11/run.py --output-dir outputs/new-tool-evaluation --device mps
uv run --locked python experiments/phase-11/verify.py outputs/new-tool-evaluation
uv run --locked pytest tests/test_tools.py
```

The full experiment requires the three preserved local learned models and original
English tokenizer. It refuses an existing output directory and records model/source
identities, configuration, all responses and failures. It uses no corpus payload,
reserved test, API, paid service or newly trained weights. Unit tests instead use
original tiny fixtures, so they also work without those learned inputs. Full raw
runs and source snapshots are ignored local artifacts, not remote backups.

The [plan](../experiments/phase-11/PLAN.md) fixes 16 original development cases: eight
single-call successes, four lookup-then-add tasks and four missing lookups. They
are not unseen tests, training data, human judgments or evidence of broad coverage.
The exact canonical call sequence is required, including argument positions for
addition; mathematically equivalent alternate strategies may be scored incorrect.

Metrics include numerators and denominators. Parsing/envelope rates count emitted
assistant turns, including incomplete text; context failure before another reply
adds no turn. Argument schema validity divides by proposed call envelopes, while
correct tool/arguments divides by 20 required call positions, counting omissions
as misses. Execution success divides by executed calls, so expected missing keys
reduce that rate. Zero denominators are `null`, not a perfect score. Normal task
success requires correct calls, successful executions and exact final text on 12
cases. Failure handling separately requires the correct missing lookup and matching
final error on four cases. A rejected request alone does not count as model failure
handling. Scripted gold control is clearly labeled and excluded from model metrics.

No Phase 11 Windows/CUDA, hosted Linux or physical Linux validation is claimed.
Earlier platform evidence, negative DPO/SFT/LoRA findings, merge/cache tolerances
and unsupported adapter/DPO optimizer resume remain as recorded in the
[Phase 10 closure](../experiments/phase-10/CLOSURE.md).
