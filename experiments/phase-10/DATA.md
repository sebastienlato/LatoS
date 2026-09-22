# Preference data provenance and limits

Original LatoS synthetic exact-answer preferences, prepared 2026-09-22 by
independently written deterministic code in `latos.preferences.preference_records`.
Source: the preserved Phase 6 original conversation templates and word groups,
under the repository MIT license; no external dataset, API or human labels.
Source generation and provenance remain in [Phase 6](../phase-6/data.py) and
[instruction documentation](../../docs/INSTRUCTION_TUNING.md).

The verified source manifest defines 48 training word groups / 192 conversations
and eight validation word groups / 32 conversations. Each group supplies copy,
extract, first-letter and multi-turn recall. All related prompts stay within their
original split. No new test split is generated. Existing reserved instruction and
English tests are never parsed or evaluated.

For each conversation, retain all messages except the last assistant turn as the
prompt; choose that final exact answer. Scan forward cyclically within the same
split for the next same-family answer whose text differs, and use it as rejected.
Store its source ID for audit. Example: copy `apple` prefers `apple` over `bridge`.
First-letter pairs skip answers with the same initial. The rule creates errors
relative to these exact tasks; it does not encode subjective helpfulness or safety.
The runner rejects equal tokenized answers, verifies shared prompt prefixes and
checks actual source IDs/groups and exact prompts for overlap across train/validation.

Prior recall assistant turns are context only for DPO. SFT validation still scores
all assistant turns, exactly as before. EOS is a response target. No truncation,
packing or train/validation mixing is used. Generated records, source/tokenizer,
window/mask hashes and counts are recorded in the run inventory and results.

These prompts intentionally share templates across splits; the vocabulary groups
are disjoint. Near similarity is therefore structural and expected, not evidence
of broad generalization. The original SFT has already trained on the training
conversations. Its strong training ranking is not an independent success. The
validation pairs and SFT exact-generation checks share prompts, so their outcomes
are correlated. Development validation has been used in earlier phases; no fresh
held-out test result is claimed. Sequence-summed likelihood is length-sensitive;
report chosen/rejected log probabilities and raw as well as reference-relative
rankings, without calling relative baseline ties wins.

The small synthetic preferences and weak SFT baseline support an engineering
experiment only. Improved optimization loss cannot establish alignment, factual
reliability, safety, open-ended chat quality or superiority to other methods.
