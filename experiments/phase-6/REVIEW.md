# Separate Phase 6 review

Performed after the initial implementation, complete 187-test pass and first
measured pilot, then rechecked after fixes. Review was a separate local pass by
Work; no independent human or external reviewer is claimed.

## Findings fixed

1. The optional masked dataset inherited pretraining's overlap-window boundary
   label. Assigned it a distinct explicit-target window policy, while preserving
   the original unmasked identity. The loss and sampling objective were unchanged.
2. The first verifier rehashed samples and recomputed scalar losses but did not
   independently replay generated responses or audit score aggregates. It now
   rebuilds prefixes with the shared formatter, regenerates every base/SFT reply,
   and verifies text, token IDs, EOS reasons, exact matches and family totals.
   Inventory paths are also constrained to the run directory.
3. Git provenance during uncommitted development identifies the parent plus dirty
   status, not the exact new source tree. The final runner now copies package
   Python source and the lock into its immutable hashed artifacts. The previous
   source was preserved before editing. This strengthens reproduction evidence.

## Contracts examined

- BOS once, explicit turn EOS, ordinary role header segments, no tokenizer changes,
  exact generation-prefix equality and assistant first-token prediction.
- Assistant-only eligibility remains unshifted through collation; the model shifts
  exactly once. Mask hashes enter identity, target counts enter accumulation and
  recovery, and all-target pretraining behavior is unchanged.
- Every conversation is an isolated row; right padding cannot influence earlier
  valid tokens. Empty targets and overflows fail explicitly instead of partial
  answer supervision. No reserved test payload is loaded by execution.
- Split word groups precede rendering, full-conversation duplicates are rejected,
  and template similarity is openly declared. This is a tiny synthetic lexical
  exercise; 0/32 completion is not useful instruction following.
- Fresh optimizer from the selected model, not a pretraining resume. Existing
  strict runtime checkpoint contracts remain enforced. CPU/MPS evidence is scoped.
- Final checkpoint chosen in advance, English validation regression reported, all
  samples retained, failures and the pre-review successful run preserved.
- No new dependency versions, acquired dataset, paid service or Phase 7 interface.
  Explicit package archive includes Phase 6 scripts/tests; prior artifacts remain
  ignored and unchanged. Publication stays local until explicit owner approval.

After fixes, the full development suite passed (187 tests), the fixed experiment
was repeated with the final implementation, and a fresh-process verifier replayed
all samples, losses and updates 100→200 within the declared MPS tolerance.
Fresh wheel and final packaging/privacy evidence appears in validation.json.
No remaining actionable finding was identified in this bounded review. This is
not a claim that the model has passed a quality, security or safety benchmark.
