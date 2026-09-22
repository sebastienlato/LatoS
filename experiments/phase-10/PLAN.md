# Phase 10 fixed protocol

Fixed before the learned run, 2026-09-22. Start only after exact Phase 9 closure
and all eleven tag refs are verified. No new paid resource or dependency upgrade.

Hypothesis: a bounded DPO run may improve relative preference margins on familiar
synthetic English task templates without improving exact generation or English
language modeling. Negative outcomes are valid; do not promote the model.

Policy initialization and frozen reference are separate copies of the original
Phase 6 fixed-update-200 SFT model, identified in the existing model card. Never
substitute the Phase 9 LoRA/merged/full control. Use original tokenizer, chat
contract 1, float32, Mac MPS, one CPU thread; existing CPU path for tiny checks.
All previous models, evidence, failures and reserved tests remain unchanged.

One run: 100 updates, batch four pairs, seed-101 epoch shuffle, fresh AdamW,
constant learning rate 1e-5, beta 0.1, weight decay 0, clip norm 1, length 256.
Select final update 100 before training; no sweep, early stopping or validation
selection. No accumulation, scheduler, optimizer resume or adapter DPO support.
Model-only output is separate; retain any failed attempts and their logs.

Preference data derives deterministically from existing Phase 6 train/validation
conversations (192/32), separately within each split. Chosen is the exact task
answer. Rejected is the next distinct answer in the same task family, wrapping in
source order. Keep prompts and earlier assistant turns identical. Source word
groups remain disjoint; templates intentionally overlap. Synthetic exact-answer
labels are not human preferences, safety judgments or sampled model feedback.
Validation words and answers never enter training. Never load either reserved
test payload. Hashing reserved bytes for preservation is allowed.

Objective: mean softplus(-beta * ((log p(chosen)-log p(rejected)) -
(log ref(chosen)-log ref(rejected)))). Each response log probability is a sum over
final assistant body and EOS, with one causal shift. Mask prompt, earlier replies
and padding. Reject overflow rather than truncate. No length normalization or
explicit additional KL term. Frozen reference is evaluated without gradients.
Derived independently from Rafailov et al., DPO, equation 7:
https://arxiv.org/html/2305.18290v3 . No external implementation copied.

Before/after: preference loss, relative margins (baseline ties must not be called
wins), raw chosen-likelihood wins, response log probabilities; all 32 held-out SFT
conversations' assistant loss, greedy exact replies and EOS stops; all 111,523
English validation targets at length 256. Preserve all generation samples.
These are development validation, not fresh tests or broad generalization.
Pair likelihood success does not establish useful generation or alignment.

Check formula and gradients independently, shift/masking/EOS/padding, fixed
reference/no optimizer membership, round trip, exposure accounting, split identity,
wrong inputs and failure retention. Run a tiny CPU integration with test files
absent, full suite, installed wheel, and a separate review with fixes. Record
runtime, source/config/data hashes, synchronized update timing and observed memory.
Preserve all Phase 9 limits, including amended merge 1e-4 and original 1e-5 failure,
unsupported adapter optimizer resume, tiny Windows/CUDA and separate hosted Linux
scope, deferred physical Linux, no useful assistant and no general device equality.

Stop after reviewed local commit; propose main only, no new tag/release/assets.
