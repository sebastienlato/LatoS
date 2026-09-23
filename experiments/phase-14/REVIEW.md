# Phase 14 — separate local review

A separate review followed implementation and the first historical panel. It was
performed locally within this task, without delegated or external reviewers.
Historical results and protocol thresholds were not changed to improve outcomes.

## Findings and fixes

- The initial comparison verified overall aggregates but trusted category,
  per-document and generation summaries. It now verifies ordered case identities,
  recomputes all those summaries, checks choice predictions against likelihoods,
  and checks instruction scoring/observations against raw text. A forged aggregate
  with an updated file hash is rejected in the CPU integration test. Inventories
  are integrity records, not signatures against replacing all evidence together.
- Added a captured input index and per-choice byte denominators so dropped/reordered
  cases and incorrect length-normalized predictions are reviewable. All original
  cases remain mandatory; missing/overflow cases block numeric acceptance.
- An overlong instruction could raise before a per-case overflow record. Formatting
  now determines the complete prompt before checking the fixed session budget,
  preserving the failed case, expected answer and prompt without truncation.
- Strengthened the weighted-aggregation test with deliberately unequal per-row
  means, rather than a fixture in which an incorrect mean-of-means also passed.
- Record numerical environment flags and package versions alongside exact source,
  lock, dataset/model/tokenizer/protocol identities. The compare tool rejects a
  changed backend/runtime/batch configuration for its paired primary reference.
- Tests cover synthetic successful final-access declarations and rejected failed
  gates without opening any real reserved payload. Final access remains an audited
  process rule, not security against a caller editing code or evidence.
- Added optional-extra execution to the existing CPU CI test command so Parquet
  integrity/schema tests run there after approved publication. No hosted CI run is
  claimed. Lock tests check Arrow wheel availability separately from execution.

## Verification boundaries

Next-token scoring is checked against independent scalar logsumexp arithmetic;
masking, padding, continuation boundaries, summed versus length-normalized choice
ranking, ties, deterministic generation, read-only weights/RNG and CPU/MPS batching
have focused tests. Gold fixture controls are not learned-model evidence. All five
historical artifacts have 0/96 new instruction successes; no model is promoted.

The first panel remains at `outputs/phase-14-evaluation`. The reviewed panel is
retained separately with the unchanged data, prompt, scoring objective, decoding,
threshold and artifact selections. Raw data and learned artifacts remain ignored.
Source snapshots honestly record the Phase 13 parent plus local Phase 14 changes.
Original historical source-fingerprint contracts are not modified.

Limitations remain explicit: narrow external science domain, two literary LM books,
small synthetic instruction fixtures, partial lexical contamination checks, public
fixtures, single-host resource observations and no Phase 14 CUDA/Linux execution.
No architecture, tokenizer, training mechanism, old evidence or tag changes.

A final comparison-only review also fixed two identity gaps: Phase 17 comparisons
now require the pinned Phase 5 weight hash, and the historical instruction reference
must belong to the frozen five-artifact panel and match the candidate's numerical
environment. A focused test rejects both baseline substitution and historical
backend mismatch. This postprocessing fix followed the reviewed inference panel;
it changes no forward scoring, raw output, threshold or aggregate. The panel's
captured package source and the final verifier/comparison hashes identify both
states explicitly. No inference rerun is claimed for that comparison-only fix.
