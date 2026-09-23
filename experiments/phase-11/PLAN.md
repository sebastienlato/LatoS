# Phase 11 fixed protocol

Fixed before model evaluation, 2026-09-23. Remote main and all eleven tag objects
and peeled targets match the verified Phase 10 closure publication at
87d744fb8cfd4228c78f35bbd220d2833462e037. The fresh-chat start is explicitly authorized.

Hypothesis: a strict bounded dispatcher can execute correct structured calls while
unchanged weak models may fail to generate them. Report engineering and learned
behavior independently. No training, checkpoint selection, prompting sweep, new
paid resource, dependency upgrade or external data. Preserve all prior artifacts.

Original protocol: one JSON object per assistant turn, either tool/args, final
(string), or error (string code). Two pure tools: add two integers in [-1000,1000],
and lookup an integer from a fixed three-entry local catalog. No shell, eval,
network, arbitrary files, plugins or model-chosen executable names. Exact keys,
strict integer types (no bool/float), bounded strings/bytes/nesting, no duplicate
keys or nonfinite JSON constants. Four assistant turns and two executions maximum;
malformed requests do not execute. Tool errors are machine-readable feedback.
Final answers never count as tool use by themselves.

Keep shared chat contract 1 and vocabulary unchanged. Tool replies are ordinary
user messages containing a JSON result, following the assistant call. The trusted
runner supplies those replies; this does not add a privileged tool role. Overflow,
cancellation and incomplete generation terminate without executing the pending call.
Callbacks are trusted local generation code, not a sandbox for arbitrary code.

Sixteen original development cases: eight single-call successes, four two-call
lookup/add tasks, four missing-key tasks. No training split or fresh test claim;
both existing reserved tests are integrity-hashed only. Freeze prompts and expected
call sequences before evaluating. Exact success requires all correct calls, successful
execution and exact final answer. Expected failure requires the prescribed missing
lookup and matching final error; runtime rejection alone is not model failure handling.

Evaluate preserved base, SFT and DPO separately, unchanged and identity-checked.
Float32 Mac MPS, one CPU thread, greedy cached generation, seed 0, at most 64 new
tokens per turn and 256 total context; no truncation, repair, constrained decoding,
or teacher-forced tool response. Same fixed protocol/prompt for each. Count valid
JSON, valid envelopes, schema-valid arguments, task-correct calls, execution success,
normal task success and missing-key failure handling with explicit denominators.
Retain every raw response and stop reason. Scripted oracle and adversarial checks
validate mechanics only, with independent exact expected outputs and metric tests.

Record source parent and exact file hashes, protocol/case/schema/model/tokenizer
identities, environment, durations, generation exposure and observed memory. Rehash
preserved inputs before/after; no optimizer state or weights are changed. CPU and
MPS tiny inference integration, full suite, fresh non-editable wheel, builds,
separate review/fixes and privacy/publication checks precede the local commit.
No Phase 11 Windows/CUDA, hosted Linux or physical Linux run is inferred from
historical evidence. Preserve all Phase 10 negative results and capability limits.
Stop at explicit publication approval: existing main only, no tag/release/assets.
