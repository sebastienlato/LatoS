# After Phase 17.1 — owner/master-planning decision only

**Stop training and keep Phase 18 blocked.** Phase 17.1 produced a mechanically
complete model that failed the fixed development quality gates. The remaining
time, interruption allowances and final-evaluation reserve are not retry budgets.
No new experiment, data acquisition, final scoring or publication is authorized
by this document.

## What changes in the Roadmap 2.0 assessment

The immediate bottleneck is now demonstrated base-model quality, rather than an
incomplete execution. Phase 15's data audit and Phase 16's resource/configuration
selection remain valid within their scopes, but did not guarantee learned success.
Do not use SFT, interoperability, advanced post-training or a release to bypass
the missing Phase 17 quality gate. The destination and numeric acceptance criteria
remain unchanged; altering them would require a separate roadmap amendment.

The new-corpus monitoring loss improves substantially, but matched-book BPB is
9.54% worse than the historical base, ARC-Easy gains only eight cases, and repetition
is higher. This supports a **generalization/quality gap despite in-domain learning**.
It does not show that the deterministic correction, architecture, tokenizer or any
single training setting caused the gap. Serious-run restart fidelity also remains
untested because this run had no interruption.

## Recommended next authorization: bounded design review, not an immediate retry

If the owner wants to continue, authorize a separate design-only review using the
existing curves, source composition, exposure accounting and fixed development
records. Set **zero optimizer updates, zero GPU training, zero new paid resources,
no new held-out/final scoring and no new data acquisition** for that review. Its
deliverable should be one prespecified, costed corrective-study proposal, or a
documented stop decision—not an open-ended search.

Examine two hypotheses without treating either as proven:

- **Exposure:** the one-pass budget provides about 1.109 target positions per
  parameter; monitoring loss still decreases. That does not establish a sufficient
  budget, saturation point or benefit from more repetitions, especially because
  the learning rate was decaying. Any proposed exposure study must distinguish
  unique data from repeated executions and retain an unchanged-data control.
- **Domain coverage/data mix:** encyclopedia-heavy training did not transfer well
  enough to both fixed book evaluations or suppress repetition. Any proposed data
  revision must justify general-English coverage independently of individual test
  examples, preserve provenance/rights, split/dedup and contamination protections,
  and retain an unchanged-exposure comparison. Do not acquire protected evaluation
  sources or near-duplicates to make the metric easier.

Prefer a design that can distinguish these hypotheses while initially holding the
accepted architecture/tokenizer and deterministic mechanics fixed. Simultaneously
changing model size, tokenizer, corpus and budget would make another outcome hard
to interpret. The current result does not justify increasing model size simply
because VRAM is available. A different choice needs its own evidence and rationale.

Before any subsequent training authorization, freeze the actual accepted inputs,
fresh initialization, control/candidate comparison, schedule and exposure, finite
attempt count, measured-resource projection and hard limits, stopping/recovery
rules, checkpoint selection, contamination review and unchanged paired acceptance
process. The Phase 5 model remains the fixed primary acceptance reference; Phase
17.1 can only be an additional failed experimental control. Preserve this candidate;
never silently continue it, call it an accepted base, or promote an earlier checkpoint. Any newly
proposed initialization policy must be explicit rather than assumed.

The old final acceptance set remains **unscored**, not a source of tuning feedback.
Development results are now observed; repeated development-driven selection must
be acknowledged in any next protocol. A passing future development comparison
still needs separate Mac selection/contamination review before the single permitted
final comparison. No automatic final access or Phase 18 follows.

The alternative is to stop with the preserved engineering platform and negative
learned results. That is an honest bounded outcome; it is not completion of the
LatoS 2.0 capability roadmap. The owner/master-planning process chooses whether to
authorize the design review, a later concrete experiment, a roadmap amendment or
no further work. Publishing this closure, if separately approved, lifts none of
these transition gates.
