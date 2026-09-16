"""Exact and exhaustive lexical near-duplicate checks for a bounded corpus."""

import re
from collections import Counter, defaultdict

WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
SHINGLE_WORDS = 5
THRESHOLD_PERCENT = 80


def shingles(text: str) -> frozenset[tuple[str, ...]]:
    words = WORD.findall(text.casefold())
    return frozenset(
        tuple(words[i : i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)
    )


class DuplicateIndex:
    """Index every shingle, without sampling; verify full-set Jaccard similarity."""

    def __init__(self):
        self.exact: dict[str, int] = {}
        self.postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
        self.sizes: list[int] = []

    def match(self, text: str, grams: frozenset) -> tuple[int, str] | None:
        if text.casefold() in self.exact:
            return self.exact[text.casefold()], "exact"
        intersections: Counter[int] = Counter()
        for gram in grams:
            for other in self.postings.get(gram, ()):
                size = self.sizes[other]
                if min(size, len(grams)) * 100 >= THRESHOLD_PERCENT * max(size, len(grams)):
                    intersections[other] += 1
        for other, intersection in sorted(intersections.items()):
            union = len(grams) + self.sizes[other] - intersection
            if intersection * 100 >= THRESHOLD_PERCENT * union:
                return other, "near"
        return None

    def add(self, text: str, grams: frozenset) -> None:
        index = len(self.sizes)
        self.exact[text.casefold()] = index
        self.sizes.append(len(grams))
        for gram in grams:
            self.postings[gram].append(index)


def deduplicate(records: list[dict]) -> tuple[list[dict], list[dict]]:
    # Reserve held-out examples first. Training never displaces a test example.
    priority = {"test": 0, "validation": 1, "train": 2}
    ordered = sorted(records, key=lambda r: (priority[r["split"]], r["source_id"], r["paragraph"]))
    index = DuplicateIndex()
    kept, removed = [], []
    for record in ordered:
        grams = shingles(record["text"])
        match = index.match(record["text"], grams)
        if match is None:
            index.add(record["text"], grams)
            kept.append(record)
        else:
            other, kind = match
            removed.append(
                {
                    "removed_id": record["id"],
                    "kept_id": kept[other]["id"],
                    "kind": kind,
                    "removed_split": record["split"],
                    "kept_split": kept[other]["split"],
                }
            )
    return sorted(kept, key=lambda r: (r["source_id"], r["paragraph"])), removed
