# Phase 7 separate review

A separate local review pass followed implementation and the initial 19-test
inference suite. The same Work agent reviewed the diff and runtime contracts;
this was not an independent external reviewer or a new platform validation.

## Actionable finding and fix

Cancellation could arrive during the final token or terminal output, after the
per-step check, allowing an EOS reply into history despite Ctrl-C. An asynchronous
KeyboardInterrupt could also interrupt byte-decoder/bookkeeping updates mid-step.

The terminal now installs a temporary SIGINT handler that sets a cancellation flag,
restores the previous handler in a finally block, and observes the flag at safe
token boundaries, including before the final event. The streaming API uses callback
cancellation; arbitrary exceptions propagate without committing history. Added a
real subprocess SIGINT test, deterministic EOS-boundary cancellation test, and
a final Unicode-flush cancellation test. Follow-up review moved the final flag
check after the residual text event so interruption of that last write also rolls
back the turn.
Generated final bytes still flush through the completed-decoding policy.

## Reviewed boundaries

- Absolute rotary positions, rectangular causal masks, layer K/V lifecycle, batch
  and capacity rejection, reset, transactional failure, model mutation detection.
- Checkpoint/state_dict compatibility and unchanged shared formatter; no training,
  corpus, optimizer or tokenizer-writing path enters inference.
- UTF-8 boundary buffering and invalid-byte replacement against completed decoding;
  control escapes in human terminal output, exact Unicode in JSON.
- Whole-turn budget reservation, formatted multi-turn history, rollback of empty,
  partial or cancelled replies, and reset of conversation/cache state.
- Fixed benchmark prompts and synthetic inputs, no test-set selection, synchronized
  timing scopes, early-EOS reporting, and logical-cache versus allocator memory.
- Explicit artifact loading and quality limits, missing/corrupt-input handling,
  package contents, locked dependencies, preservation hashes and publication privacy.

No remaining actionable finding in this bounded review. Limitations remain:
terminal-only UI, per-reply prefill, no paged cache, no concurrent model mutation,
no useful assistant claim, and no Phase 7 Windows/CUDA or Linux execution evidence.
Final validation and artifact checks are recorded in REPORT.md and PROJECT_STATE.md.
