"""Author the fixed original MIT instruction/generation fixtures; never training data."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build():
    cases = []

    def add(category, family, prompt, expected, scorer="exact"):
        cases.append(
            {
                "id": f"{category}-{len(cases):03}",
                "category": category,
                "family": family,
                "prompt": prompt,
                "expected": expected,
                "scorer": scorer,
            }
        )

    for name, number in [("Mira", 3), ("Jules", 8), ("Oren", 2), ("Tess", 5)]:
        add(
            "structured",
            "json-object",
            f'Return only a JSON object with "name" set to "{name}" and '
            f'"count" set to the integer {number}.',
            {"name": name, "count": number},
            "json",
        )
        add(
            "structured",
            "json-array",
            f'Return only a JSON array containing "{name}" followed by the integer {number}.',
            [name, number],
            "json",
        )
        add(
            "structured",
            "csv",
            f"Write one CSV row, without a header: name {name}, count {number}.",
            f"{name},{number}",
        )
        add(
            "structured",
            "key-value",
            f"Format the name {name} and count {number} as "
            f"name=VALUE;count=VALUE. Output only that line.",
            f"name={name};count={number}",
        )
    for word, upper, reversed_word, letters in [
        ("fern", "FERN", "nref", "f-e-r-n"),
        ("kite", "KITE", "etik", "k-i-t-e"),
        ("plum", "PLUM", "mulp", "p-l-u-m"),
        ("dusk", "DUSK", "ksud", "d-u-s-k"),
    ]:
        add(
            "transformation",
            "upper",
            f"Convert {word} to uppercase. Return only the converted word.",
            upper,
        )
        add(
            "transformation",
            "lower",
            f"Convert {upper} to lowercase. Return only the converted word.",
            word,
        )
        add(
            "transformation",
            "reverse",
            f"Reverse the order of the letters in {word}. Return only the result.",
            reversed_word,
        )
        add(
            "transformation",
            "join",
            f"Separate the letters of {word} with hyphens. Return only the result.",
            letters,
        )
    for animal, place, number, color in [
        ("otter", "pond", "14", "blue"),
        ("finch", "barn", "23", "red"),
        ("badger", "hill", "31", "green"),
        ("seal", "bay", "42", "gold"),
    ]:
        add(
            "extraction",
            "field",
            f"Record: animal={animal}; place={place}; count={number}. Return only the place.",
            place,
        )
        add(
            "extraction",
            "brackets",
            f"Return only the text inside square brackets: {color} [{animal}] {place}.",
            animal,
        )
        add(
            "extraction",
            "sentence",
            f"The {animal} saw {number} boats near the {place}. How many "
            f"boats? Reply with digits only.",
            number,
        )
        add(
            "extraction",
            "list-position",
            f"List: {animal}, {place}, {color}. Return only the third item.",
            color,
        )
    for word, count in [("moss", "4"), ("pebble", "6"), ("reed", "4"), ("lagoon", "6")]:
        add(
            "constrained", "literal", f"Reply with exactly this word and nothing else: {word}", word
        )
        add(
            "constrained",
            "two-lines",
            f"Write {word} on the first line and DONE on the second line. Write no other text.",
            f"{word}\nDONE",
        )
        add(
            "constrained",
            "count",
            f"Count the letters in {word}. Reply using one digit and nothing else.",
            count,
        )
        add(
            "constrained",
            "choice",
            f"Choose the shorter word: {word} or extraordinary. Output only your choice.",
            word,
        )
    qa = [
        ("How many days are in a week? Reply with one digit.", "7"),
        ("How many months are in a year? Reply with digits only.", "12"),
        ("What number comes immediately after nine? Reply with digits only.", "10"),
        ("How many sides does a triangle have? Reply with one digit.", "3"),
        ("Does ice normally melt when heated? Reply yes or no.", "yes"),
        ("Can a stone breathe? Reply yes or no.", "no"),
        ("Is a whale a mammal? Reply yes or no.", "yes"),
        ("Is the Sun a planet? Reply yes or no.", "no"),
        ("What is 2 plus 3? Reply with digits only.", "5"),
        ("What is 9 minus 4? Reply with digits only.", "5"),
        ("What is 3 times 2? Reply with digits only.", "6"),
        ("What is 8 divided by 2? Reply with digits only.", "4"),
        ("Nia is taller than Sol. Who is shorter? Reply with the name only.", "Sol"),
        (
            "A cup is inside a box. The box is in a room. Is the cup in the room? Reply yes or no.",
            "yes",
        ),
        ("All daxes are blue. Pip is a dax. What color is Pip? Reply with the color only.", "blue"),
        (
            "The red bag weighs 2 kilograms and the green bag weighs 5 "
            "kilograms. Which bag is heavier? Reply red or green.",
            "green",
        ),
    ]
    for i, (prompt, answer) in enumerate(qa):
        add("qa", ["factual", "yes-no", "arithmetic", "given-facts"][i // 4], prompt, answer)
    for a, b, sorted_text, joined in [
        ("pear", "apple", "apple,pear", "APPLE-PEAR"),
        ("wolf", "bear", "bear,wolf", "BEAR-WOLF"),
        ("red", "blue", "blue,red", "BLUE-RED"),
        ("sun", "moon", "moon,sun", "MOON-SUN"),
    ]:
        add(
            "multistep",
            "sort-join",
            f"Sort these words alphabetically: {a}, {b}. Join them with a "
            f"comma and no spaces. Output only the result.",
            sorted_text,
        )
        add(
            "multistep",
            "sort-upper-join",
            f"Sort {a} and {b} alphabetically, convert both to uppercase, and "
            f"join with a hyphen. Output only the result.",
            joined,
        )
        add(
            "multistep",
            "select-transform",
            f"From this record select the second word, then uppercase it: "
            f"{a};{b}. Output only the result.",
            b.upper(),
        )
        add(
            "multistep",
            "filter-format",
            f"Remove the word {a} from this list: {a}, {b}, {a}. Put the "
            f"remaining word in parentheses. Output nothing else.",
            f"({b})",
        )
    prompts = [
        ("narrative", "The gate opened just before dawn, and"),
        ("narrative", "Mara found a folded note beneath the empty cup. It said"),
        ("description", "The small garden after the rain"),
        ("description", "Inside the quiet railway station,"),
        ("explanation", "A seed needs water because"),
        ("explanation", "To keep a room warm in winter,"),
        ("dialogue", '"Where did you leave the key?" asked Mina.\n'),
        ("dialogue", '"I am sorry I arrived late," said Tomas.\n'),
        ("procedure", "To make a cup of tea, first"),
        ("procedure", "Before crossing a busy road,"),
        ("factual", "Water changes from liquid to ice when"),
        ("factual", "The Earth travels around"),
        ("modern", "The laptop could not connect to the network, so"),
        ("modern", "The bus timetable showed that"),
        ("reasoning", "There were three apples. One was eaten, leaving"),
        ("reasoning", "If the blue box is larger than the red box, the red box is"),
        ("uncertainty", "I do not know the answer, but"),
        ("uncertainty", "The evidence is incomplete, so we should"),
        ("unicode", "The café sign read “Welcome”, and"),
        ("unicode", "The temperature fell to −2°C overnight. In the morning,"),
        ("consistency", "Lena put the letter in a drawer and closed it. The letter was now"),
        ("consistency", "The boat was painted green yesterday. Its new color was"),
        ("list", "Items to bring for a rainy walk:\n1."),
        ("list", "A short shopping list:\n-"),
    ]
    return {
        "schema_version": 1,
        "version": "latos-eval-1",
        "license": "MIT",
        "provenance": "Original AI-assisted project fixtures authored "
        "2026-09-23; not human labels or training data.",
        "split": "development; fixed before historical evaluation; exclude from all training",
        "instructions": cases,
        "generation": [
            {"id": f"generation-{i:02}", "category": category, "prompt": prompt}
            for i, (category, prompt) in enumerate(prompts)
        ],
    }


if __name__ == "__main__":
    path = ROOT / "configs/evaluation/suite-v1.json"
    if path.exists():
        raise SystemExit("Suite already exists; do not overwrite the frozen suite")
    path.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
