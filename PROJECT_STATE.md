# Project state

Updated: 2026-09-24.

## Current owner authorization and execution checkpoint

**Phase 17.1 implementation and one prespecified fresh Windows attempt are explicitly
owner-authorized.** Mac implementation, focused validation and separate review are
complete. Stop at the Windows execution handoff; the owner manually transfers the
checked package to a fresh Windows Work session and returns evidence here.
**No full training runs on Mac. No Phase 18 or publication is authorized.**

The approved [proposal](experiments/phase-17-1/PROPOSAL.md) and
[contract](experiments/phase-17-1/plan.json) remain byte-identical to reviewed commit
`1b57e0d317a57d3bd2cde6a28cce6c8184022b31`. Their historical pending-authorization
wording is superseded by the latest owner instruction and the hash-bound
[authorization record](experiments/phase-17-1/authorization.json).

## Permanent historical outcome and diagnostic support

Published Phase 17 remains permanently interrupted/failed at
`09c1935b085218fbbc9ea88703d8ba12e63c516e`: recovery equivalence failed, recovered
model unaccepted, fixed development/final acceptance absent, exit gate unmet.
No recovered, probe or diagnostic model initializes or replaces the fresh attempt.

The independently reviewed synthetic recovery diagnostic remains positive within
its frozen scope: 2,964 updates, legacy drift reproduced, exact deterministic state
and loss agreement across 485 replay updates. It does not identify the historical
kernel cause, repair Phase 17 or establish learned quality. See
[review](experiments/recovery-diagnostic/RETURN_REVIEW.md) and
[verification](experiments/recovery-diagnostic/RETURN_VERIFICATION.json).

## Completed local implementation

New standalone tools implement deterministic CUDA policy, source/input integrity,
append-only session journals, exclusive launch/OS locks, ticketed/hash-bound child
requests, cumulative time/artifact limits, immutable checkpoint policy records and
exact full-state/schedule recovery checks. Native `src/latos`, dependencies/lock,
model/data/training configurations, prior evidence and learned-quality gates are
unchanged. The initial supervisor permits only prescribed preflight, one fresh
training run and paired development; final acceptance needs independent Mac review.

[Windows execution guide](experiments/phase-17-1/WINDOWS.md),
[implementation review](experiments/phase-17-1/IMPLEMENTATION_REVIEW.md) and
[validation](experiments/phase-17-1/implementation-validation.json).

**35 focused CPU checks passed, zero skips/failures, in both checkout and the
preserved installed-wheel environment; zero optimizer updates.** These include
real tiny zero-update checkpoint roundtrips, failure/corruption/policy rejection,
real watchdog/locks, return packaging, and separately labeled scripted supervisor
controls. Existing train/development cache verification passed: 119,748/2,552 windows
and 37,811,418/804,337 targets, without rebuilding or creating a full model.
Formatting/lint pass. The lock is unchanged; `uv lock --check` could not run because
`uv` was unavailable on PATH. No new full-suite, CUDA, hosted CI or physical Linux
result is claimed. Actual Windows locking/import/CUDA conformance remains pending.

## Frozen execution scope and next action

Run the checked allowlisted tools beside the preserved original transfer at
`C:\LatoS-Validation\phase17\execution-20260924-verified`, with the existing
checkout and installed-wheel runtimes. The tools do not include corpus, tokenizer,
weights, held-out payloads or private context. Exact source commit, ZIP SHA-256 and
post-commit verification are recorded in the supplied kickoff/receipt and local
continuity. Do not pull Git or replace the historical source/environment.

Windows first verifies source/input integrity, then runs the same zero-update CPU
checks and exactly the approved tiny adapter conformance design: at most 52 CUDA
updates across four reference/replay pairs, with no retries. Only successful
preflight permits the fresh seed-160 34,087,424-parameter model and unchanged one-pass
recipe: 7,485 logical updates / 37,811,418 targets, BF16, batch two × accumulation
eight, final accumulation two. Two-hour preparation/preflight/training/recovery cap,
one-hour cumulative fixed evaluation reserve, three-hour active total, 20 GiB new
artifacts, 8 GiB host / 85% reserved GPU caps and zero new paid resources remain.

Return complete or failed evidence for authoritative Mac review. No automatic
restart or final scoring. Conditional external-interruption recovery needs a
separate Mac record and stays within two resumptions and all cumulative bounds.
Passing development/resource gates plus the reviewed candidate/contamination
selection declaration are required before the single final comparison. Phase 18
remains blocked. Native implementation hash stays
`76cf4a198ee6acde566fdcc70c9420c0cae8ab6cc0d6b4d7e4c9c0cba5c8f09c`.

Current work is local on `main`. Existing remote is
`https://github.com/sebastienlato/LatoS.git`; eventual destination `main`, no tag
proposed. No push, tag, release, asset upload or other remote write is authorized
or performed. This implementation checkpoint is not an executed/accepted base model.
