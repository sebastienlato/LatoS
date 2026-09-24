# Phase 16 Windows/CUDA measurement handoff

This is an **external-validation checkpoint, not Phase 16 completion**. The owner
authorized a separate Windows Work session because there is no authenticated CUDA
connection from the authoritative Mac. Perform the bounded measurements below;
return the evidence for Mac review. Do not select the final configuration, start
Phase 17, push, tag, publish, rent compute, upgrade dependencies or change access.

## Transfer and identity

The local `outputs/phase16-windows-handoff.zip` contains only the reviewed source
commit under `source/`, the accepted Phase 15 training/development package under
`inputs/`, and `handoff.json` with every file's size/hash and exact source commit.
The source commit is local and has not been pushed; do not substitute remote main.
Transfer this archive directly to the owner's Windows machine, outside GitHub.
Extract into a **new directory**, preserving the existing checkout and all evidence.
It is a local transfer, not permission to redistribute acquired text/tokenizers.
Source notices and terms travel with the inputs. No private planning files,
credentials, environments, historical weights or acquired reserved/evaluation
payloads are included. Reviewed source includes its normal public unit fixtures.

Verify the archive SHA-256 against the Mac handoff message/record, then run the
standard-library `source/experiments/phase-16/verify_handoff.py` with the extracted
root as its argument. Stop on any mismatch. No model or data substitution is valid.
Read `AGENTS.md`, `PROJECT_STATE.md`, `ROADMAP.md`, `docs/PHASE16_INPUTS.md`, this
guide and `experiments/phase-16/PLAN.md`. The latest owner direction authorizes this
bounded external validation only; historical pending/publication snapshots do not
override it. Private context is intentionally not transferred in this archive.

## Environment and checks — Windows Work operates these

Use the existing Windows x86-64 CPython 3.14.7 and uv 0.12.15, with a new local
environment inside the extracted source. Follow `docs/SETUP.md` if project-local
tool setup is needed. Use the committed lock (`uv sync --locked --all-extras`).
It routes Torch to `2.14.0+cu130` on Windows. No global changes or invented versions.
Capture install output, dependency check, `latos doctor --device cuda --json`,
`nvidia-smi`, the exact interpreter and installed versions. Keep all logs in a new
ignored output directory. If hardware, driver, package or disk access blocks the
run, record the actual error and stop; do not ask the owner to debug routine issues.

Run lint/format checks and the full test suite in the checkout, then build a wheel
and test a fresh non-editable installation of that exact wheel with the same locked
dependencies. Verify import resolves into the new environment's site-packages,
not the checkout. Keep source/tests available as test inputs and ensure CLI PATH
points to the tested environment. Record every failure and exact skip reason.
Mac-only tests may skip; the new CUDA BF16 numerical test must execute. Skip counts
are evidence, not a target to force. Preserve the first failed attempt if fixes
become necessary; return any patch separately with tests and review.

## One bounded measurement suite

From the extracted `source` directory, use the checked local interpreter:

```powershell
uv run --locked --all-extras python experiments/phase-16/run.py --inputs ../inputs --output outputs/phase16-cuda-attempt1
```

This verifies pinned input bytes before training. Numerical controls run first.
Then six isolated candidate processes compare layers 8/14/20 at 34.09M/53.36M/72.63M
parameters, each in float32 (8 updates) and BF16 (64 updates). Actual native dense
models use the new tokenizer and real hash-selected training/development text.
This small feasibility exposure is distinct from serious Phase 17 pretraining.
Each candidate has a 30-minute process timeout; the control has ten minutes.
There are no automatic retries or parameter substitutions. Every process has new
output directories, logs, source snapshots and file inventories. Partial failures
and timeouts remain evidence, not completed measurements. A failed numerical
control blocks all candidate runs. Model/precision dimensions and thresholds may
not be relaxed after viewing results.

Check that the measured GPU is the actual RTX 4070 SUPER and that synchronized
timings, allocated/reserved peaks, host peak working set, validation, checkpoint
readback/reload, exposure and forced-length inference results were written.
The current probe keeps checkpoints at initial, midpoint and final states, resumes
from the midpoint, and tests synthetic nonfinite failure recovery separately.
The memory headroom limit includes validation and checkpoint duplication overhead.
Activation checkpointing, FP16, fused/foreach optimizers, compilation and TF32 are
not enabled. The common batch/context is a bounded first protocol, not a claim
that it maximizes throughput. Any needed revised experiment returns to Mac review.

Do not score frozen reserved tests, historical full-model benchmarks or final
instruction suites. Do not launch old experiment reproduction scripts. Full unit
tests are synthetic mechanics; short real probes are feasibility/early loss;
neither establishes a learned capability gate.

## Return evidence and stop

Retain the entire original output directory and checkpoints locally. Return a
compact result archive containing the handoff identity, install/test/build logs,
wheel identity, environment/driver reports, all probe JSON/JSONL, worker logs,
source snapshots and original inventories, including failures. Exclude model and
optimizer tensor payloads from the compact return archive but preserve their
paths, bytes and hashes in the original inventories; mark the archive explicitly
as an evidence subset, not a full checkpoint backup. Include any reviewed patch
and a fresh inventory if Windows code changes were needed. Never discard attempts.

Summarize actual executed checks and skipped/blocked paths, correctness results,
per-candidate losses/exposure/throughput/memory/disk/inference and the exact source
identity. Do not claim Phase 16 completion or nominate an accepted base. Stop for
Mac review and a final measured configuration/Phase 17 budget decision. Phase 17
and all GitHub writes remain prohibited in the Windows task.
