"""Independent next-token scoring with explicit targets and deterministic aggregation."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Sequence:
    ids: tuple[int, ...]
    mask: tuple[bool, ...]

    def validate(self, vocab_size, context):
        if not 2 <= len(self.ids) <= context or len(self.mask) != len(self.ids):
            raise ValueError("Invalid scoring sequence length or mask")
        if any(type(i) is not int or not 0 <= i < vocab_size for i in self.ids):
            raise ValueError("Invalid scoring token")
        if any(type(x) is not bool for x in self.mask) or self.mask[0] or not any(self.mask):
            raise ValueError("Invalid scoring mask or no targets")


def score_sequences(model, sequences, *, batch_size=8, context=256):
    """NLL per row. Logit t predicts token t+1; prefix and right padding never score.

    Float32 logits move to CPU before float64 logsumexp and summation. No model
    loss/training evaluator is reused. A final unused logit row is never scored.
    """
    if any(m.training for m in model.modules()):
        raise ValueError("Evaluation requires all modules in eval mode")
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError("Invalid evaluation batch size")
    if type(context) is not int or not 2 <= context <= model.config.context_length:
        raise ValueError("Scoring context exceeds model capacity")
    if not sequences:
        raise ValueError("No sequences to score")
    for sequence in sequences:
        sequence.validate(model.config.vocab_size, context)
    results = []
    device = model.embedding.weight.device
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            rows = sequences[start : start + batch_size]
            length = max(len(row.ids) for row in rows)
            ids = torch.zeros((len(rows), length), dtype=torch.long, device=device)
            for i, row in enumerate(rows):
                ids[i, : len(row.ids)] = torch.tensor(row.ids, device=device)
            logits = model(ids).logits.detach().cpu().double()
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite evaluation logits")
            for i, row in enumerate(rows):
                positions = [j for j in range(1, len(row.ids)) if row.mask[j]]
                selected = logits[i, [j - 1 for j in positions]]
                targets = torch.tensor([row.ids[j] for j in positions])
                nlls = torch.logsumexp(selected, dim=-1) - selected.gather(
                    1, targets[:, None]
                ).squeeze(1)
                results.append({"nll": nlls.sum().item(), "targets": len(positions)})
    return results


def legacy_windows(codec, text, context=256):
    ids = codec.encode(text, add_bos=True, add_eos=True)
    return [
        Sequence(tuple(ids[i : i + context]), (False,) + (True,) * (len(ids[i : i + context]) - 1))
        for i in range(0, len(ids) - 1, context - 1)
    ]


def byte_spans(text, budget=192):
    """Lossless, tokenizer-independent UTF-8 spans; never split a Unicode scalar."""
    if type(budget) is not int or budget < 4 or not isinstance(text, str) or not text:
        raise ValueError("Invalid text or byte span budget")
    span, size = "", 0
    for char in text:
        width = len(char.encode("utf-8"))
        if size + width > budget:
            yield span
            span, size = "", 0
        span += char
        size += width
    if span:
        yield span


def matched_windows(codec, text):
    result = []
    for span in byte_spans(text):
        ids = codec.encode(span, add_bos=True)
        if codec.decode(ids) != span:
            raise ValueError("Byte-normalized evaluation requires lossless tokenization")
        result.append(Sequence(tuple(ids), (False,) + (True,) * (len(ids) - 1)))
    return result


def continuation(codec, prefix, answer):
    """Score a space-prefixed answer, excluding question/prefix and EOS.

    Require a stable joint-encoding boundary instead of guessing which crossing
    BPE token belongs to the answer. Unsupported boundaries fail explicitly.
    """
    if not prefix or prefix[-1].isspace() or not answer.strip():
        raise ValueError("Continuation needs a non-whitespace prefix end and an answer")
    before = codec.encode(prefix, add_bos=True)
    ids = codec.encode(prefix + " " + answer, add_bos=True)
    if ids[: len(before)] != before or len(ids) <= len(before):
        raise ValueError("Tokenizer merges across the continuation boundary")
    return Sequence(tuple(ids), (False,) * len(before) + (True,) * (len(ids) - len(before)))


def lm_summary(rows):
    if not rows or any(
        type(r["targets"]) is not int
        or r["targets"] <= 0
        or not math.isfinite(r["nll"])
        or r["nll"] < 0
        for r in rows
    ):
        raise ValueError("Invalid language-model results")
    nll = math.fsum(r["nll"] for r in rows)
    targets = sum(r["targets"] for r in rows)
    loss = nll / targets
    result = {
        "nll": nll,
        "targets": targets,
        "nats_per_token": loss,
        "perplexity": math.exp(loss) if loss < 700 else None,
    }
    if all("bytes" in r for r in rows):
        size = sum(r["bytes"] for r in rows)
        if size <= 0:
            raise ValueError("No text bytes")
        result.update(bytes=size, bits_per_byte=nll / (size * math.log(2)))
    return result


def wilson(successes, total):
    """Descriptive 95% binomial interval, not an IID/population guarantee."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    p, z = successes / total, 1.959963984540054
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def accuracy(rows, key="correct"):
    if not rows or any(type(r[key]) is not bool for r in rows):
        raise ValueError("Expected nonempty Boolean scores")
    count = sum(r[key] for r in rows)
    return {
        "correct": count,
        "total": len(rows),
        "accuracy": count / len(rows),
        "wilson95": wilson(count, len(rows)),
    }
