"""Objective fixture scoring and descriptive generation observations."""

import json
import re

from latos.chat import format_chat
from latos.inference import stream_generate


def instruction_correct(text, case):
    if case["scorer"] == "exact":
        return text.strip() == case["expected"]
    if case["scorer"] != "json":
        raise ValueError("Unknown instruction scorer")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Repeated JSON key")
            result[key] = value
        return result

    def bad(value):
        raise ValueError(value)

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=bad)
    except ValueError, RecursionError:
        return False
    # Canonical JSON preserves numeric/Boolean types (Python True == 1 does not).
    return json.dumps(value, sort_keys=True, ensure_ascii=False) == json.dumps(
        case["expected"], sort_keys=True, ensure_ascii=False
    )


def observations(text, stop_reason):
    words = re.findall(r"\w+", text.casefold())
    trigrams = list(zip(words, words[1:], words[2:], strict=False))
    return {
        "empty": not text.strip(),
        "replacement_character": "\ufffd" in text,
        "eos": stop_reason == "eos",
        "word_count": len(words),
        "repeated_trigram_fraction": (1 - len(set(trigrams)) / len(trigrams)) if trigrams else 0.0,
    }


def generate_case(model, codec, case, *, instruction, context=256, budget=64):
    if instruction:
        ids, _ = format_chat(
            codec,
            [{"role": "user", "content": case["prompt"]}],
            max_length=4096,
            generation_prompt=True,
        )
    else:
        ids = codec.encode(case["prompt"], add_bos=True)
    if len(ids) + budget > context:
        return {
            "id": case["id"],
            "category": case["category"],
            "status": "context_overflow",
            "correct": False,
            "prompt_tokens": len(ids),
            "prompt": case["prompt"],
            **({"family": case["family"], "expected": case["expected"]} if instruction else {}),
        }
    done = None
    for event in stream_generate(
        model,
        codec,
        list(ids),
        max_new_tokens=budget,
        temperature=0.0,
        top_k=0,
        seed=0,
        cached=True,
    ):
        if event["event"] == "done":
            done = event
    if done is None:
        raise RuntimeError("Generation returned no completion record")
    return {
        "id": case["id"],
        "category": case["category"],
        "status": "ok",
        "prompt": case["prompt"],
        **done,
        "observations": observations(done["text"], done["stop_reason"]),
        **(
            {
                "correct": instruction_correct(done["text"], case),
                "family": case["family"],
                "expected": case["expected"],
            }
            if instruction
            else {}
        ),
    }
