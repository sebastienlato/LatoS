# Phase 15 separate local review

A distinct review pass followed implementation and the first complete data pass.
It was performed in this task, without delegated or external reviewers. Source
quality review continued on the fixed samples and every significant observed
defect was quarantined or addressed by a general rule before tokenizer fitting.
The remaining uncertainty is documented rather than converted into a quality claim.

## Findings and fixes

- Raw samples exposed quotations, unframed personal anecdotes, unreliable factual
  and code answers, and missing numeric-template values. Add conservative filters
  and four input-level quarantines. Post-filter samples identified four further
  defective records; a monotonic final curation removes them without changing any
  surviving content, IDs or split assignment. Minor grammar errors and unverified
  claims remain explicit limitations; no full factual/safety certification.
- The original after-sample construction included text later removed as paragraphs.
  It now reflects actual retained train/development text. Pre-split construction
  samples can precede new reservations; do not call these wholly unseen final tests.
- Tokenizer fitting initially trusted a row's split label without checking its
  deterministic group assignment. The verified reader now checks both; a fixture
  with a forged train label and reserved group is rejected even after updating
  the outer file hash. Fitting works with held-out payloads absent.
- Dot/dot-dot filenames were not explicitly rejected by the initial acquisition
  validator. Reject both; cached corruption remains an error without replacement.
  One initial wheel lacked that later guard and failed its new regression. The
  rebuilt final wheel is tested separately; all attempts/logs remain.
- The Wikipedia card's license wording was stale. Retain it verbatim in the raw
  notice cache, distinguish controlling Wikipedia 4.0 terms from Dolly's 3.0
  snapshot/contexts, and keep use and redistribution decisions separate.
- Initial 60k-page retention missed the declared construction floor. Preserve that
  completed result and refine resource bounds using its actual runtime/RSS. The
  final 600 MB selection remains inside the original 12 GB artifact / 24 GiB RSS
  budgets; no acceptance minimum, evaluation score or quality gate was lowered.
- Resource scripts initially hard-coded host labels/imported a Unix-only module.
  Future runs report actual platform and optional RSS; unsupported RSS is unavailable,
  not zero. No Windows/CUDA or Linux execution is claimed.
- Preparation source identity now refers to the preserved pre-run snapshot. Later
  finalization, acquisition guards, resource wrappers and tokenizer code have
  separate source/artifact identities. Documentation-only collision wording was
  corrected: full-index coverage is over fingerprints, not collision-free strings.
- The complete output audit exposed three same-split pretraining/Dolly overlaps.
  Initial finalization compared only pretraining records and retained those pairs.
  Finalization now indexes instructions first and removes matching pretraining
  chunks. It also checks short exact chunks that have no five-shingles. Both cases
  have regression coverage. The final audit observes zero duplicate pairs.
- A runner named `tokenizers.py` shadowed the external Tokenizers package and failed
  during import before fitting. Rename it `compare_tokenizers.py`; standalone
  entrypoint help tests now exercise all relevant scripts from another directory.

## Actual verification

The final full corpus is audited, not sampled, for output identity/counts, group
membership, source-bound text identity, protected overlap and full-record lexical
pairs. The input joins also cover complete documents and >=40-word fragments.
A 4,000-real-candidate two-pass raw preparation replay is byte-identical. Repeating
full-scale final curation also produces identical data, ledgers and report. This
is not a claim of a second complete raw-to-final full-corpus run.

[Validation](validation.json) records final test/build/install results. New tests
compare the prefix join to an independent brute-force reference; cover malformed
records, integrity failures, role order, metadata/path quality, source/template
isolation, protected detection, deterministic serialization, train-only fitting,
lossless codec behavior and fixed BPB/chat compatibility. Standard repository tests
retain the old numerical/mechanical coverage and disposable training fixtures.
They are not newly learned capability measurements.

[Quality review](quality-review.json) records fixed sample identities and inspection
limits. [Artifacts](artifacts.json), [replay](replay.json),
[finalization replay](finalization-replay.json), [resources](resources.json) and
[construction gates](acceptance.json) bind the actual local result. Acquired text,
reserved payloads, learned tokenizers, full logs and local package remain ignored.
All historical evidence and frozen evaluations are preserved; numeric gates and
negative model findings remain unchanged. Publication approval is still required.
