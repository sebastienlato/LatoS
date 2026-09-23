"""Data 2.0 mechanics use original disposable fixtures, never real reservations."""

import json
import random
from pathlib import Path

import pytest

from latos.data2.acquire import acquire, file_hash, read_manifest
from latos.data2.filtering import clean, quality_reason, validate_messages
from latos.data2.lexical import Groups, LexicalIndex, digest
from latos.data2.prepare import family_keys, prepare, split_for
from latos.data2.protected import Guard
from latos.data2.sources import best_path


def test_prefix_join_matches_exhaustive_unordered_sets():
    rng = random.Random(1515)
    values = [frozenset(rng.sample(range(250), rng.randint(5, 80))) for _ in range(160)]
    for base in values[:80]:
        values.append(frozenset(sorted(base)[2:] + [300, 301]))
    rng.shuffle(values)
    for threshold in (0.6, 0.8, 1.0):
        index = LexicalIndex(threshold)
        for i, row in enumerate(values):
            expected = [
                j
                for j, other in enumerate(values[:i])
                if len(row & other) >= threshold * len(row | other)
            ]
            assert index.matches(row) == expected
            assert index.add(row) == i
    assert LexicalIndex().matches(frozenset()) == []


def test_jaccard_boundary_and_transitive_groups():
    index = LexicalIndex(0.8)
    index.add(frozenset(range(8)))
    assert index.matches(frozenset(range(10))) == [0]
    assert index.matches(frozenset(range(11))) == []
    groups = Groups(4)
    groups.union(3, 2)
    groups.union(1, 2)
    assert groups.find(3) == groups.find(1)
    assert groups.find(0) != groups.find(3)


@pytest.mark.parametrize(
    "text,reason",
    [
        ("", "empty"),
        ("abc\ufffd", "corrupt_or_control"),
        ("a\ud800b", "encoding"),
        ("abc\x01", "corrupt_or_control"),
        ("A Tale of Two Cities is a novel", "excluded_source_family"),
        ("Please contact alice@example.org about this", "potential_contact_or_secret"),
        ("how to make a bomb using household objects", "unsuitable_lexical"),
        ("word " * 100, "excessive_repetition"),
        ("The animal weighs about . Its tail is long.", "missing_template_value"),
        ("This railway is approximately  in length.", "missing_template_value"),
    ],
)
def test_rejection_reasons(text, reason):
    assert quality_reason(text) == reason


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": "hi"}],
        [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "hello"}],
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}],
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "extra": True},
        ],
    ],
)
def test_bad_conversations(messages):
    with pytest.raises(ValueError):
        validate_messages(messages)


def test_contamination_exact_containment_and_near():
    g = Guard()
    text = " ".join("item" + str(i) for i in range(40))
    g.add(text, "synthetic_protected")
    assert g.match(text.upper()) == "protected_exact"
    assert g.match("prefix " + text + " suffix") == "protected_13word"
    # Disrupt every 12-word run, but keep enough common five-shingles for a near case.
    short = "a b c d e f g h i j k l"
    g.add(short, "synthetic_protected")
    assert g.match("a b c d e f g h i j k l z") == "protected_near"
    assert g.match("a distinct invented example") is None
    assert "texts_by_kind" in g.summary()
    assert text not in json.dumps(g.summary())


def test_whitespace_cleaning_and_template_group():
    assert clean("Cafe\u0301\r\n  text\t ") == "Café\n  text"
    assert clean("body\nReferences\nnot body", encyclopedia=True) == "body"
    a = {"kind": "instruction", "prompt": 'Write a story about "A" for 3 people.'}
    b = {"kind": "instruction", "prompt": 'Write a story about "B" for 9 people.'}
    assert set(family_keys(a)) & set(family_keys(b))
    assert split_for("fixed") == split_for("fixed")


def test_best_tree_path_quality_provenance_and_parent():
    def node(identity, role, replies=(), **changes):
        return {
            "message_id": identity,
            "role": role,
            "lang": "en",
            "review_result": True,
            "synthetic": False,
            "text": "original fixture",
            "labels": {"quality": {"value": 0.9}},
            "replies": list(replies),
            **changes,
        }

    unsafe = node("bad", "assistant", synthetic=True, rank=0)
    good = node("good", "assistant", rank=1)
    tree = {"tree_state": "ready_for_export", "prompt": node("root", "prompter", [unsafe, good])}
    assert [n["message_id"] for n in best_path(tree)] == ["root", "good"]
    good["parent_id"] = "wrong"
    with pytest.raises(ValueError, match="parent"):
        best_path(tree)


def fixture_manifest(tmp_path):
    raw = tmp_path / "input.txt"
    raw.write_text("original fixture\n")
    item = {
        "file": raw.name,
        "bytes": raw.stat().st_size,
        "sha256": file_hash(raw),
        "url": "https://example.org/input.txt",
    }
    manifest = {
        "schema_version": 1,
        "sources": [
            {
                "id": "fixture",
                "revision": "1",
                "license": "MIT",
                "rights": "original test",
                "origin": "fixture",
                "files": [item],
            }
        ],
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    return p, raw, manifest


def test_acquisition_verified_cache_corruption_and_no_adoption(tmp_path):
    p, raw, manifest = fixture_manifest(tmp_path)
    assert acquire(p, tmp_path)["files"][raw.name] == manifest["sources"][0]["files"][0]["sha256"]
    raw.write_text("corrupt")
    with pytest.raises(ValueError, match="integrity"):
        acquire(p, tmp_path)
    assert raw.read_text() == "corrupt"


@pytest.mark.parametrize(
    "field,value",
    [
        ("file", "../escape"),
        ("file", ".."),
        ("sha256", "bad"),
        ("bytes", 3_000_000_000),
        ("url", "http://example.org"),
    ],
)
def test_manifest_bounds(tmp_path, field, value):
    p, _, manifest = fixture_manifest(tmp_path)
    manifest["sources"][0]["files"][0][field] = value
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_manifest(p)


def test_preparation_determinism_groups_and_reserved_no_sample(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("latos.data2.prepare")
    records = []
    for i in range(100):
        # Unique five-grams; enough prose to exercise actual paragraph/chunk logic.
        text = " ".join(f"term{i}value{j}" for j in range(100))
        records.append(
            {
                "id": f"wiki:{i}",
                "source": "wiki",
                "kind": "pretrain",
                "category": "encyclopedia",
                "title": f"Title {i}",
                "text": text,
            }
        )
    records.append({**records[0], "id": "wiki:duplicate"})
    monkeypatch.setattr(
        module, "load_sources", lambda *_: (json.loads(json.dumps(records)), {"fixture": 101}, [])
    )
    monkeypatch.setattr(module, "is_english", lambda *_: True)
    monkeypatch.setattr(module, "quality_reason", lambda *_, **__: None)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    a, b = tmp_path / "a", tmp_path / "b"
    ra = prepare(manifest, tmp_path, a, Guard(), detector=object())
    rb = prepare(manifest, tmp_path, b, Guard(), detector=object())
    assert ra == rb
    assert ra["duplicates"]["document_exact_or_jaccard80"] == 1
    memberships = {}
    for path in a.glob("pretrain-*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            memberships.setdefault(row["group_id"], set()).add(row["split"])
            assert row["content_sha256"] == digest(row["text"])
    assert all(len(x) == 1 for x in memberships.values())
    assert all(file_hash(a / name) == file_hash(b / name) for name in ra["files"])
    from latos.data2.audit import audit

    assert audit(a, Guard())["passed"]
    damaged = a / "documents.jsonl"
    damaged.write_text(damaged.read_text() + "{}\n")
    with pytest.raises(ValueError, match="integrity"):
        audit(a, Guard())
    with pytest.raises(FileExistsError):
        prepare(manifest, tmp_path, a, Guard(), detector=object())


def test_language_identification_real_optional_detector():
    pytest.importorskip("lingua")
    from latos.data2.filtering import is_english, make_detector

    detector = make_detector()
    assert is_english(
        detector, "This English explanation describes how a library lends books to its readers."
    )
    assert not is_english(
        detector,
        "Esta explicación está escrita en español y describe cómo funciona una biblioteca pública.",
    )


def test_training_only_tokenizer_fitting_and_chat_contract(tmp_path):
    from latos.data2.acquire import write_json
    from latos.data2.tokenizer import codec_checks, fit, read_records, training_sample
    from latos.tokenization import LatoTokenizer

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    group = next(digest(str(i)) for i in range(100) if split_for(digest(str(i))) == "train")
    row = {
        "id": "fixture",
        "kind": "pretrain",
        "source": "fixture",
        "split": "train",
        "group_id": group,
        "text": (
            "This original fixture explains how a library "
            "organizes its books and welcomes readers. "
        )
        * 4,
    }
    row["content_sha256"] = digest(row["text"])
    (corpus / "pretrain-train.jsonl").write_text(json.dumps(row) + "\n")
    (corpus / "instruction-train.jsonl").write_text("")
    report = {
        "files": {
            n: {"sha256": file_hash(corpus / n)}
            for n in ("pretrain-train.jsonl", "instruction-train.jsonl")
        }
    }
    write_json(corpus / "report.json", report)
    # Development/reserved files deliberately do not exist.
    assert training_sample(corpus)[1][0]["id"] == "fixture"
    a, b = tmp_path / "tokenizer-a", tmp_path / "tokenizer-b"
    fit(corpus, a, 512)
    fit(corpus, b, 512)
    assert file_hash(a / "tokenizer.json") == file_hash(b / "tokenizer.json")
    assert file_hash(a / "metadata.json") == file_hash(b / "metadata.json")
    assert codec_checks(LatoTokenizer.load(a))["passed"]
    with pytest.raises(ValueError, match="limited"):
        list(read_records(corpus, "pretrain", "reserved"))
    row["group_id"] = next(
        digest(str(i)) for i in range(1000) if split_for(digest(str(i))) == "reserved"
    )
    (corpus / "pretrain-train.jsonl").write_text(json.dumps(row) + "\n")
    report["files"]["pretrain-train.jsonl"]["sha256"] = file_hash(corpus / "pretrain-train.jsonl")
    write_json(corpus / "report.json", report)
    with pytest.raises(ValueError, match="identity"):
        training_sample(corpus)


def test_response_and_rights_filters():
    from latos.data2.filtering import response_reason

    record = {
        "messages": [
            {"role": "user", "content": "Who inspires you?"},
            {"role": "assistant", "content": "My mother inspires me."},
        ]
    }
    assert response_reason(record) == "unframed_personal_claim"
    record["messages"][0]["content"] = "Write a story in the first person."
    assert response_reason(record) is None
    assert (
        quality_reason('A review said "' + "quoted material " * 20 + '"')
        == "quotation_or_rights_uncertainty"
    )
    assert (
        quality_reason("This place is about  southeast of the village.") == "missing_template_value"
    )


def test_data2_locked_binary_platforms():
    import tomllib

    lock = tomllib.loads((Path(__file__).resolve().parents[1] / "uv.lock").read_text())
    package = next(p for p in lock["package"] if p["name"] == "lingua-language-detector")
    assert package["version"] == "2.2.0"
    for target in ("macosx_11_0_arm64", "manylinux_2_17_x86_64", "win_amd64"):
        assert any(
            "cp314-cp314-" + target in w["url"] and w["hash"].startswith("sha256:")
            for w in package["wheels"]
        )


@pytest.mark.parametrize(
    "script", ["run.py", "compare_tokenizers.py", "finalize.py", "repeat.py", "package_inputs.py"]
)
def test_experiment_entrypoints_do_not_shadow_dependencies(script, tmp_path):
    import subprocess
    import sys

    path = Path(__file__).resolve().parents[1] / "experiments/phase-15" / script
    result = subprocess.run(
        [sys.executable, str(path), "--help"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_final_curation_removes_short_exact_chunks_without_resplitting(tmp_path):
    from latos.data2.acquire import write_json
    from latos.data2.curation import finalize

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    group = next(digest(str(i)) for i in range(100) if split_for(digest(str(i))) == "train")
    docs = [
        {
            "id": f"wiki:{i}",
            "kind": "pretrain",
            "source": "wiki",
            "category": "encyclopedia",
            "group_id": group,
            "split": "train",
            "rows": 1,
        }
        for i in range(2)
    ]
    for kind in ("pretrain", "instruction"):
        for split in ("train", "development", "reserved"):
            rows = (
                [
                    {
                        **d,
                        "document_id": d["id"],
                        "text": "Short tail.",
                        "content_sha256": digest("Short tail."),
                    }
                    for d in docs
                ]
                if kind == "pretrain" and split == "train"
                else []
            )
            (source / f"{kind}-{split}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
    (source / "documents.jsonl").write_text("".join(json.dumps(r) + "\n" for r in docs))
    for name in ("duplicates.jsonl", "rejections.jsonl"):
        (source / name).write_text("")
    write_json(source / "quality-before.json", {})
    write_json(
        source / "report.json",
        {
            "files": {p.name: {"sha256": file_hash(p)} for p in source.iterdir()},
            "rejections": {},
            "duplicates": {},
        },
    )
    policy = tmp_path / "policy.json"
    write_json(policy, {"schema_version": 1, "records": {}})
    report = finalize(source, policy, output)
    assert report["finalization"]["removed_records"] == 1
    row = json.loads((output / "pretrain-train.jsonl").read_text())
    assert row["text"] == "Short tail." and row["group_id"] == group
    with pytest.raises(FileExistsError):
        finalize(source, policy, output)
    # A pretraining chunk can also repeat a complete instruction record. Keep the
    # instruction intact and remove both pretraining copies, including short text.
    extra = {
        **docs[0],
        "id": "oasst:fixture",
        "kind": "instruction",
        "source": "oasst",
        "category": "single_turn",
    }
    messages = [{"role": "user", "content": "Short"}, {"role": "assistant", "content": "tail."}]
    (source / "instruction-train.jsonl").write_text(
        json.dumps({**extra, "messages": messages, "content_sha256": digest("Short\ntail.")}) + "\n"
    )
    with (source / "documents.jsonl").open("a") as f:
        f.write(json.dumps(extra) + "\n")
    write_json(
        source / "report.json",
        {
            "files": {
                p.name: {"sha256": file_hash(p)}
                for p in source.iterdir()
                if p.name != "report.json"
            },
            "rejections": {},
            "duplicates": {},
        },
    )
    second = finalize(source, policy, tmp_path / "cross-kind")
    assert second["finalization"]["removed_records"] == 2
    assert (tmp_path / "cross-kind/pretrain-train.jsonl").read_text() == ""
    assert (
        json.loads((tmp_path / "cross-kind/instruction-train.jsonl").read_text())["messages"]
        == messages
    )
