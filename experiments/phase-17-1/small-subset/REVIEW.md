# Separate small-subset implementation review

A separate Mac pass reviewed the final controller, adapter, guardian, packaging and
saved-evidence reader after implementation. This is engineering review, not learned
quality or Windows validation. No external reviewer or agent delegation is claimed.

## Preserved scope

The authorized proposal is byte-identical to commit
`150221d6de5e4ca3969905643a64b75111bd91d6`. The native implementation, old controllers,
configs, fixed gates/reference, roadmap, lock and historical negative evidence are
unchanged. No dependency or model family was added. The 6.6-hour study remains
rejected. The existing Phase 16 subset reconstructs to its exact original identities;
its encoded local cache is transferred outside Git, with all original provenance.

## Review findings and resolutions

- **Tail semantics:** the native multi-pass loader assumes unflushed accumulation.
  Altering its counters or pretending the new tails satisfy that format would be
  incorrect. A separate adapter changes only the actual tail accumulation count,
  restores the schedule config before evidence capture, advances the persistent
  sampler at each pass boundary, and checks exposure. Separate diagnostic checkpoint
  envelopes retain full optimizer/sampler tensors and exact storage readback; they
  have no resume API. Historical checkpoint semantics remain untouched.
- **Deadline scope:** include archive verification/extraction, preflight, every arm,
  all scoring, verification and packaging in one clock. Capture its Windows QPC
  origin in the kickoff before Python starts; do not exclude interpreter startup. Create the controller
  suspended and assign it to a kill-on-close Windows Job Object before resuming.
  The guardian kills all descendants on expiry/closure, independently of worker
  cooperation. Workers cannot reset the start or invoke a later packaging job.
  Any platform/assignment failure stops before optimization. Actual Windows nested
  job termination is a required zero-update preparation check, still unexecuted.
- **Evidence closure:** package within the same cap, stream tensor files and avoid
  compression of the large return. Keep failures and interrupted artifacts; a
  successful worker/ZIP does not imply successful execution or quality. The Mac
  verifier requires the separately returned completed/guardian records and rejects
  more than 2,700 seconds or packaging outside the original clock.
- **Scientific isolation:** fixed A/B/C job order, same subset/permutation seed,
  identical A/B initial tensors, declared depth-only C change, prescribed B1
  observation, no scoring until training ends, no final-entry path, and no gate-
  driven branching. The sole acceptance reference is the preserved Phase 5 model.
  Descriptive cross-arm contrasts are labeled separately, with no model promotion.
- **Integrity:** bind source, native/installed-wheel imports, numeric controls,
  cache identities, model/optimizer/data settings, full-state digests, every input
  signature, pass tails and actual checkpoint readback. Reconstruct all serious
  inputs from the fixed cache and seed, not just aggregate counts. Record every
  job's actual imported source. Return omitted tensors as inventory identities,
  never claim that missing bytes were rehashed on Mac.
- **Performance correction:** a first evidence-verifier draft shadowed the dataset
  variable; the corrected reader distinguishes trace rows from the fixed dataset.
  An expanded scripted test also exposed repeated whole-subset hashing per update.
  It was interrupted after 40.32 seconds with 17 checks complete; cache the verified
  dataset identity once. The final complete suite now finishes in about three seconds.
- **Test scope correction:** initial timeout tests accidentally scanned their parent
  test directory, encountering pytest symlinks. Pass the exact study root explicitly.
  Later added tests initially lacked two imports; corrected before the final run.
  All initial failures/interruption XML files remain local. None performed optimizer
  updates or represents a learned attempt/retry.

## Actual validation and limits

Twenty-three focused checks pass in development and the existing installed native
wheel, and again against an extracted allowlisted provisional handoff in each
runtime. Tests forbid native trainer/AdamW updates. They cover the real cache
reader/corruption rejection, accumulation and schedule logic, tiny CPU checkpoint
storage, scripted full 1,910-record evidence with tampering, fixed controller order,
process timeout cleanup, mocked Windows Job API lifecycle, expired/failure packaging,
Mac launch rejection and over-budget return rejection. The scripted records are not
training. Tiny CPU random models support storage checks only; no forward/scoring.

Lint, format, JSON, link, privacy and historical-preservation checks pass. Provisional
archive member/readback checks and source-file equality are recorded. Final handoff
Git/source/member binding and corruption rejection occur after the reviewed local
commit, with exact receipts retained outside Git; no remote publication follows.
No broad optimizer-bearing suite, GPU training, new model scoring, CUDA/physical
Linux run or CI was performed. Hardware/runtime throughput, actual 52-update CUDA
conformance, native Windows Job behavior and reduced-depth learned results remain
pending. Failure of any required Windows control ends the bounded attempt.

Win32 declarations and process-containment semantics were checked against primary
Microsoft documentation: [Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects),
[AssignProcessToJobObject](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject),
and [extended limits](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information).
Clock origin uses [Python monotonic time](https://docs.python.org/3/library/time.html#time.monotonic)
and the [Windows Stopwatch counter](https://learn.microsoft.com/en-us/dotnet/api/system.diagnostics.stopwatch.gettimestamp).
The implementation is original Python/ctypes; no external project code or recipe
was imported. Existing runtime versions remain pinned and unchanged.

**Disposition:** ready for the requested Windows execution checkpoint only. The
owner-authorized diagnostic is not a full Base Model attempt; no final scoring,
accepted base, Phase 18, paid compute or publication is authorized.
