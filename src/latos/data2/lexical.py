"""Deterministic lexical fingerprints and exhaustive prefix-filtered Jaccard joins."""

import hashlib
import math
import re
import unicodedata
from collections import defaultdict

WORDS = re.compile(r"\w+", re.UNICODE)


def words(text: str) -> list[str]:
    return WORDS.findall(unicodedata.normalize("NFKC", text).casefold())


def normalized(text: str) -> str:
    return " ".join(words(text))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def grams(text: str, width: int = 5) -> frozenset[int]:
    tokens = words(text)
    return frozenset(
        int.from_bytes(
            hashlib.blake2b(" ".join(tokens[i : i + width]).encode(), digest_size=8).digest(), "big"
        )
        for i in range(len(tokens) - width + 1)
    )


class LexicalIndex:
    """Full-set verification; prefix filter has no sketch/sampling recall loss.

    For Jaccard >= t the overlap is >= ceil(t * max(n,m)). Thus the prefixes
    n-ceil(t*n)+1 and m-ceil(t*m)+1 under one common order must intersect.
    64-bit fingerprints have a small, unmeasured collision risk; guarantees concern
    fingerprint sets, not collision-free original shingle strings.
    Empty shingle sets never count as near matches; exact text is checked separately.
    """

    def __init__(self, threshold: float = 0.8):
        if not 0 < threshold <= 1:
            raise ValueError("Invalid Jaccard threshold")
        self.threshold = threshold
        self.postings = defaultdict(list)
        self.sets: list[frozenset[int]] = []
        self.comparisons = 0

    def prefix(self, values):
        return sorted(values)[: len(values) - math.ceil(self.threshold * len(values)) + 1]

    def matches(self, values: frozenset[int]) -> list[int]:
        if not values:
            return []
        candidates = set()
        for value in self.prefix(values):
            candidates.update(self.postings.get(value, ()))
        result = []
        for index in sorted(candidates):
            other = self.sets[index]
            if min(len(values), len(other)) < self.threshold * max(len(values), len(other)):
                continue
            self.comparisons += 1
            overlap = len(values & other)
            if overlap >= self.threshold * (len(values) + len(other) - overlap):
                result.append(index)
        return result

    def add(self, values: frozenset[int]) -> int:
        index = len(self.sets)
        self.sets.append(values)
        if values:
            for value in self.prefix(values):
                self.postings[value].append(index)
        return index


class Groups:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)
