# Bounded Windows small-subset handoff

The owner explicitly authorized this diagnostic's implementation and bounded
Windows preparation. Mac stops at this checkpoint; the supplied kickoff is the
only Windows execution instruction. Historical Phase 17 remains permanently
failed/interrupted and Phase 17.1 development-failed. No accepted base, full Base
Model run, final acceptance, Phase 18, publication or paid compute is authorized.
The rejected 6.6-hour/four-pass plan must never be run.

## Placement and one launch

Use the preserved root:
`C:\LatoS-Validation\phase17\execution-20260924-verified`.
It already contains `source`, `inputs`, `evaluation`, the original checkout
`source\.venv` and installed `source\.wheel-env`. Reuse the original pinned
Windows/Python/Torch/driver combination. Do not fetch, pull, install, upgrade,
rerun old tools, repair inputs or replace historical evidence.

Copy the new `small-subset-windows-handoff.zip`, `small-subset-launch.py`, receipt
and Mac kickoff into that existing root. Transporting the supplied files is the
handoff; **all experiment preparation on Windows starts with the launch below**.
Do not pre-extract, pre-hash, tokenize, benchmark or run separate checks. Read the
kickoff, stop other GPU work, then run its exact single command from this root:

```powershell
$studyClock = ([Diagnostics.Stopwatch]::GetTimestamp() / [double][Diagnostics.Stopwatch]::Frequency).ToString("R", [Globalization.CultureInfo]::InvariantCulture)
& .\source\.venv\Scripts\python.exe .\small-subset-launch.py --transfer . --archive .\small-subset-windows-handoff.zip --sha256 <EXACT_MAC_ARCHIVE_SHA256> --origin $studyClock
```

The concrete kickoff contains the checksum, with no placeholder. Only the verified
standalone guardian may launch the study. It checks itself against the archive
manifest inside its guarded bootstrap. Do not invoke bootstrap/controller/workers
or packaging directly. The fixed new `small-subset-v1` directory is a permanent
attempt claim; never delete or rename it to rerun. Missing runtime/files, changed
source or any integrity failure means stop. There is no authorized retry.

## One wall clock, including packaging

The kickoff captures the Windows performance-counter clock before starting Python,
so interpreter startup is included. Python uses the same Windows monotonic clock;
the guardian requires that original timestamp before archive verification/extraction
or experiment preparation. It starts the controller suspended, assigns it to a Windows Job Object
with kill-on-close and no breakaway, then resumes it. All descendants inherit that
job. Controller interruption/guardian closure kills the contained processes.
A real zero-update parent/grandchild termination check runs within preparation.
If Job Object assignment or that check fails, no optimizer job may start.

Target about 33 minutes; preparation <=180 s, all serious arms combined <=1,440 s,
all development work <=480 s and verification/packaging <=300 s. These are cumulative
stage caps, not per-process allowances. Each deadline is also bounded by the original
clock. The controller stops by 2,690 s, and the independent guardian terminates the
whole process tree at 2,695 s, leaving five seconds before the owner's **2,700-second
absolute limit**. There is no timer reset, pause exclusion or later packaging run.
Emergency headroom is for evidence closure only, never extra training/scoring.

The actual Windows Job API and sustained timing are pending this execution. An OS
failure, suspended host, timeout, missing receipt or guardian elapsed time above
2,700 seconds is a failed/incomplete study, never evidence of budget compliance.
Do not restart or extend it. Preserve any partial ZIP as incomplete.

## Fixed sequence and evidence

1. Verify the reviewed tools, exact existing native/installed-wheel source, required
   development/reference/tokenizer inputs and fixed cache identities. Run zero-update
   accounting and real guardian checks. No final/reserved payload is required or read.
2. Run only the existing four tiny ordinary/partial-tail conformance pairs across
   checkout and installed wheel: 8-update reference plus 5-update separate-process
   replay per pair; **52 tiny updates total**, exact full-state/input match and the
   unchanged `1e-5` loss tolerance. Do not run the full historical test suite.
3. Three fresh seed-160 arms, with empty optimizers: A, 8 layers/34.09M, 382 updates;
   B, same model, 764 updates; C, 4 layers/21.24M, 764 updates. Exactly 1,910 serious
   updates and 9,715,075 target executions. Same existing 1,943,015-target subset,
   tokenizer, batch 2 × accumulation 8, 19-update warmup and fixed short/long cosine
   schedules. At each pass end, five microbatches with one window in the final batch.
   Preserve all eight initialization/pass-boundary checkpoints and monitoring results.
4. After all training completes, score only Phase 5 reference, A1, B1, B2 and C2 with
   the unchanged full development protocol. No intermediate-result-driven change,
   alternate checkpoint search or final access. Negative quality gates are retained
   diagnostic findings, not permission to tune or skip later prescribed observations.
5. Verify all 1,910 saved signatures against the fixed cache and two deterministic
   permutations, counters, full-state digests and checkpoint inventories. Reconstruct
   fixed gate comparisons and descriptive paired contrasts from saved rows. Package
   and read back a compact return within the same clock. No new model scoring here.

The native training/evaluation code is unchanged. A separate adapter flushes pass
accumulation and advances the persistent sampler. Serious checkpoints use a distinct
`small-subset-no-resume-v1` envelope because the historical native loader assumes
unflushed multi-pass accumulation. Tensor serialization/readback is exact; **there
is no serious checkpoint restore API or resumption authorization**. Original native
single-pass save/load is used only for the already specified tiny replay fixtures.
The bundled historical plan supplies immutable numerical controls and those fixtures;
its old full-training budgets/commands are not callable study instructions.

## Return and stop

The new cache is the existing audited Phase 16 selection, rebuilt and identity-checked
on Mac; it is local dataset material, not a Git/release artifact. Preserve original
input notices/provenance. Maximum new storage is 8 GiB, with >=12 GiB free at start;
worker host RSS <=8 GiB and reserved GPU memory <=85%. All writes stay under the new
study directory. All old source, data, logs, checkpoints and failures stay intact.

On completion, return these already produced files without another packaging job:

- `small-subset-v1\run\small-subset-return.zip`
- `small-subset-v1\run\return.receipt.json`
- `small-subset-v1\run\completed.json`
- `small-subset-v1\guardian.json`

The return contains saved logs/scores, source snapshots, checkpoint metadata and four
scored model tensor files when present. Other model/optimizer tensors remain on
Windows with inventory hashes; the return is not their complete backup. Never delete
the originals. `completed.json` and the guardian receipt bind end-to-end timing.
An exit zero is insufficient without complete receipts, inventories and timing.

If failure prevents a verified ZIP, return only existing guardian/outcome/log files
and report that packaging did not finish. Do not create a replacement archive, run
hashing/scoring after the deadline or conceal incomplete evidence. The Mac review
uses its own reviewed verifier and known handoff, not executable code from a return.

Use one concise final Windows summary: completion/failure, total wall time, counts,
file paths and material limitations. Keep detailed logs local, do not paste traces
into Work context or open parallel tasks. **Stop after evidence return for owner/Mac
review, whatever the quality results. No publication or Phase 18.**
