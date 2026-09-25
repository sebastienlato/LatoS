# Final subset-confirmation Windows handoff

Owner-authorized diagnostic, fresh seed 161. **Absolute complete-study cap: 30 minutes.**
This is the last authorized experiment for the current fixed subset. E4 failing any
fixed development gate ends that direction and returns to owner/master planning of
data/training design. An improving trajectory does not authorize more passes. Even
an E4 development pass does not accept Base Model 2.0 or authorize final scoring/Phase 18.

## One invocation

Reuse `C:\LatoS-Validation\phase17\execution-20260924-verified` with its original
`source`, `inputs`, `evaluation`, checkout `.venv` and installed `.wheel-env`.
No package/driver changes, downloads, Git writes, paid services or old launcher runs.
Preserve every old source, model, run directory and failure, including the completed
small-subset study and its packaging correction. Do not initialize from their weights.

Copy `subset-confirmation-windows-handoff.zip`, `subset-confirmation-launch.py`,
its receipt and the prepared Mac kickoff into that root. Transporting supplied
files is the handoff; all experiment preparation begins in the timed invocation.
Do not pre-extract, pre-hash, reconstruct data, benchmark or run separate tests.
If `subset-confirmation-v1` already exists, stop; never delete it to obtain a retry.

Run the exact concrete block from the Mac kickoff once. Its structure is:

```powershell
$studyClock = ([Diagnostics.Stopwatch]::GetTimestamp() / [double][Diagnostics.Stopwatch]::Frequency).ToString("R", [Globalization.CultureInfo]::InvariantCulture)
Set-Location -LiteralPath 'C:\LatoS-Validation\phase17\execution-20260924-verified'
& .\source\.venv\Scripts\python.exe .\subset-confirmation-launch.py --transfer . --archive .\subset-confirmation-windows-handoff.zip --sha256 <EXACT_MAC_SHA256> --origin $studyClock
```

The kickoff supplies the actual hash; do not use a placeholder. No direct bootstrap,
controller, worker, evaluation or packaging calls are authorized. The guardian starts
the complete clock before Python startup, verifies/extracts the archive, and contains
the controller plus all descendants in a Windows Job Object. The real nested-job
termination check must pass during preparation, before any optimizer work.

## Fixed work and limits

D: fresh 21.24M, two passes / 764 updates. E: fresh same model, four passes / 1,528
updates. Both use seed 161 and the unchanged 1,943,015-target / 6,105-window cache.
Same 19-update warmup, fixed short/long cosine schedules, BF16/deterministic settings,
batch two × accumulation eight, and exact five-microbatch pass tails. The first two
permutations and initial weights match. Save/monitor only initialization and pass
boundaries: eight checkpoints total, with exact storage readback and no resume API.

Preparation includes only zero-update controls and the original capped four tiny
8+5-update CUDA conformance pairs across checkout and wheel: at most 52 updates.
Their old fixture seeds are controls, never serious initializers. Serious total is
**2,292 updates / 11,658,090 targets**, maximum physical work **2,344 updates**.
No extra optimizer fixtures, seed, retries, recovery, skipped tails or data changes.

Only after both arms finish, run four fixed development observations: Phase 5,
D2, E2 and E4. E2 is a diagnostic snapshot, not another arm or selectable candidate.
No E1/E3/full/final scoring or intermediate-driven changes. Primary gates always use
Phase 5; corrected descriptive pairing operates on verified saved rows directly.
Packaging includes the terminal stop disposition and all positive/negative results.
A failed quality gate is a result to preserve, not an execution exception or retry.

Target 20 minutes, projected about 17 minutes. Hard cumulative stage caps:
preparation/conformance 180 s, both training jobs 1,080 s, all scoring 300 s,
verification/packaging 180 s. The remaining 60 s is evidence-closure reserve only.
Controller cutoff 1,790 s; independent guardian terminates all descendants at
1,795 s. **The entire invocation must finish within 1,800 s.** No reset, excluded
startup time, per-arm allowance, extension or deferred packaging. Prior studies'
spare time does not count toward this independently authorized budget.

Require >=12 GiB free disk, <=8 GiB new artifacts (including return/reserve),
<=8 GiB worker host RSS and <=85% reserved VRAM. Use the existing RTX 4070 SUPER,
pinned Python/Torch/driver and original numerical policy; no paid compute. Identity,
control, nonfinite, OOM, storage/memory/time or packaging failures end this attempt.
No repairs or retry are automatically authorized. Preserve any partial evidence.
Actual new-seed/four-pass execution and timing are untested until this run.

## Return and mandatory stop

The return is built and verified inside the same invocation. Return these existing
files from `subset-confirmation-v1`:

- `run\subset-confirmation-return.zip`
- `run\return.receipt.json`
- `run\completed.json`
- `guardian.json`

The ZIP includes scored D2/E2/E4 model files and metadata/logs/scores. Other model,
optimizer and cache bytes remain on Windows with inventory identities; keep all
originals, since the compact return is not a complete backup. Do not publish files.
If no verified return exists, report the failure and existing logs/guardian/outcome
only; do not create a later manual archive or invoke old corrective tools.

Report one concise status with wall time, counts, terminal disposition and file
paths. Do not paste raw traces, open parallel tasks or propose more subset passes.
**Stop for Mac/owner review.** The Mac will independently check evidence and scientific
interpretation. Preserve the permanent Phase 17 failure, Phase 17.1 quality failure
and earlier subset result. No accepted base, Phase 18 or publication follows here.
