"""Conservative, auditable quality heuristics; not a safety or correctness proof."""

import re
import unicodedata
from collections import Counter

from latos.data2.lexical import words

# Entire documents/conversations are rejected, never repaired into invented answers.
SOURCE_BLOCK = re.compile(
    r"alice.?s adventures in wonderland|the war of the worlds|a tale of two cities|"
    r"the wonderful wizard of oz|lewis carroll|h\.?\s*g\.?\s*wells|charles dickens|"
    r"l\.?\s*frank baum|ai2[_ -]?arc|arc[_ -]?(easy|challenge)|"
    r"openbookqa|winogrande|hellaswag|truthfulqa|gsm8k|mmlu|big.?bench|"
    r"stanford alpaca|flan collection|benchmark question",
    re.I,
)
CONTACT = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b(?:\+?\d[ ()-]*){10,15}\b")
SECRET = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:sk-|ghp_)[A-Za-z0-9]{20,}")
UNSUITABLE = re.compile(
    r"\b(?:child pornography|sexual assault instructions|kill yourself|"
    r"how to (?:make a bomb|commit suicide|steal passwords)|nigger|faggot)\b",
    re.I,
)
BOILERPLATE = re.compile(
    r"^(?:References|External links|Bibliography|Further reading|Notes and references)\s*$", re.I
)


def clean(text: str, *, encyclopedia: bool = False) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    if encyclopedia:
        lines = text.splitlines()
        end = next(
            (i for i, line in enumerate(lines) if BOILERPLATE.fullmatch(line.strip())), len(lines)
        )
        text = "\n".join(lines[:end])
    return text.strip()


def validate_messages(messages: list[dict]) -> None:
    if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
        raise ValueError("Conversation requires complete user/assistant pairs")
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Malformed message")
        if message["role"] != ("user" if i % 2 == 0 else "assistant"):
            raise ValueError("Invalid conversation order")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError("Empty message")
        message["content"].encode("utf-8", errors="strict")


def quality_reason(text: str, *, instruction: bool = False) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return "empty"
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeError:
        return "encoding"
    if "\ufffd" in text or any(unicodedata.category(c) == "Cc" and c not in "\n\t" for c in text):
        return "corrupt_or_control"
    if len(text.encode()) > 100_000:
        return "oversized"
    if SOURCE_BLOCK.search(text):
        return "excluded_source_family"
    if re.search(r'["“][^"”\n]{120,}["”]', text) or re.search(
        r"\b(?:song lyrics|full lyrics|copyright reserved|all rights reserved)\b", text, re.I
    ):
        return "quotation_or_rights_uncertainty"
    if re.search(r"\b(?:about|approximately) {2,}(?:north|south|east|west)", text, re.I):
        return "missing_template_value"
    if re.search(r"\b(?:about|approximately|roughly) +[.,;]", text, re.I) or re.search(
        r"\b(?:of|about|approximately|is|was|lies) {2,}"
        r"(?:of|and|in length|in height|above|below|north|south|east|west)\b",
        text,
        re.I,
    ):
        return "missing_template_value"
    if CONTACT.search(text) or SECRET.search(text):
        return "potential_contact_or_secret"
    if UNSUITABLE.search(text):
        return "unsuitable_lexical"
    tokens = words(text)
    if len(tokens) < (5 if instruction else 80):
        return "too_short"
    if sum(c.isalpha() for c in text) / len(text) < (0.35 if instruction else 0.55):
        return "low_prose_fraction"
    triples = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    if triples and 1 - len(set(triples)) / len(triples) > 0.35:
        return "excessive_repetition"
    if tokens and Counter(tokens).most_common(1)[0][1] / len(tokens) > 0.2:
        return "dominant_word"
    return None


def make_detector():
    from lingua import LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()


def is_english(detector, text: str) -> bool:
    from lingua import Language

    # Inspect beginning/middle/end, bounded to 1,500 characters in total.
    if len(text) > 1500:
        middle = len(text) // 2
        text = text[:500] + "\n" + text[middle : middle + 500] + "\n" + text[-500:]
    scores = detector.compute_language_confidence_values(text)
    return bool(scores and scores[0].language == Language.ENGLISH and scores[0].value >= 0.6)


def response_reason(record: dict) -> str | None:
    """Reject recognizable response defects, preserving unknown factual uncertainty."""
    prompt = record["messages"][0]["content"]
    creative = re.search(r"write|story|pretend|imagine|role.?play|fiction|poem", prompt, re.I)
    for message in record["messages"][1::2]:
        response = message["content"]
        if response.strip().casefold() == prompt.strip().casefold():
            return "response_echo"
        if not creative and re.search(
            r"\bmy (?:mom|mother|father|wife|husband|childhood|family)\b|\bI was born\b",
            response,
            re.I,
        ):
            return "unframed_personal_claim"
    return None
