"""Materialize original, deterministic English teaching cases into ignored storage."""

import argparse
from pathlib import Path

from latos.data.manifest import canonical_json, sha256

# Independent source groups are assigned before rendering any conversations.
# Related tasks for one word stay together. Shared task templates are intentional.
GROUPS = {
    "train": "apple bridge candle garden window river basket cloud meadow mirror pencil orange "
    "pillow silver summer winter forest button pocket rocket carpet castle feather island "
    "kitten ladder marble needle ocean paper planet purple rabbit ribbon saddle sailor "
    "shadow shovel spider spring stone street table ticket timber valley velvet village".split(),
    "validation": "anchor blossom copper drawer engine fountain glacier harbor".split(),
    "test": "acorn beacon pebble thimble walnut willow compass lantern".split(),
}
SYSTEM = "Answer the user's request briefly and exactly."


def conversation(group, family, pairs):
    messages = [{"role": "system", "content": SYSTEM}]
    for user, assistant in pairs:
        messages.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        )
    return {
        "id": f"{group}-{family}",
        "source_group": group,
        "family": family,
        "messages": messages,
    }


def records_for(split):
    records = []
    for word in GROUPS[split]:
        records.extend(
            [
                conversation(word, "copy", [(f"Copy exactly this word: {word}", word)]),
                conversation(
                    word,
                    "extract",
                    [
                        (
                            f"Read this note: The label is {word}. "
                            "What is the label? Reply with only the label.",
                            word,
                        )
                    ],
                ),
                conversation(
                    word, "first", [(f"Give only the first letter of this word: {word}", word[0])]
                ),
                conversation(
                    word,
                    "recall",
                    [
                        (f"Remember the word {word}. Reply only with OK.", "OK"),
                        ("Which word did I ask you to remember? Reply with only that word.", word),
                    ],
                ),
            ]
        )
    return records


def materialize(directory):
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "name": "latos-english-instructions-v1",
        "provenance": (
            "Original Codex-authored templates and word groups, 2026-09-20; "
            "deterministic expansion; no external data or model API"
        ),
        "terms": (
            "Original project material under repository MIT license; "
            "synthetic educational data, not human demonstrations"
        ),
        "split_policy": (
            "word source groups assigned before rendering; "
            "four related conversations remain together"
        ),
        "splits": {},
    }
    fingerprints = set()
    for split, groups in GROUPS.items():
        records = records_for(split)
        for record in records:
            digest = sha256(canonical_json(record["messages"]))
            if digest in fingerprints:
                raise ValueError("Duplicate conversation")
            fingerprints.add(digest)
        payload = b"".join(canonical_json(record) for record in records)
        (directory / f"{split}.jsonl").write_bytes(payload)
        manifest["splits"][split] = {
            "source_groups": groups,
            "ids": [record["id"] for record in records],
            "bytes": len(payload),
            "sha256": sha256(payload),
            "conversations": len(records),
        }
    (directory / "manifest.json").write_bytes(canonical_json(manifest))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    materialize(parser.parse_args().output_dir)
