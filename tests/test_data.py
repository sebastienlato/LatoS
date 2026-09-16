"""Offline acquisition, leakage prevention, corruption, and reproducibility checks."""

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from latos.data.acquire import HTTPSRedirects, acquire
from latos.data.dedup import DuplicateIndex, deduplicate, shingles
from latos.data.manifest import canonical_json, load_manifest, sha256
from latos.data.prepare import audit, clean_paragraph, extract_body, prepare

FIXTURE = Path(__file__).resolve().parents[1] / "data/fixtures/tiny"


@pytest.fixture
def corpus(tmp_path):
    fixture = tmp_path / "fixture"
    shutil.copytree(FIXTURE, fixture)
    manifest = fixture / "manifest.json"
    raw = tmp_path / "raw"
    acquire(manifest, raw)
    return manifest, raw, tmp_path / "processed"


def test_repeatable_preparation_and_no_split_overlap(corpus):
    manifest, raw, output = corpus
    report = prepare(manifest, raw, output)
    second = output.with_name("second")
    assert prepare(manifest, raw, second) == report
    for path in output.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    assert {k: v["records"] for k, v in report["splits"].items()} == {
        "train": 3,
        "validation": 3,
        "test": 4,
    }
    assert report["duplicates"] == {"exact": 1, "near": 1}
    assert report["cross_split_duplicates_removed"] == 2
    assert audit(manifest, output) == {
        "status": "ok",
        "records": 10,
        "exact_duplicates": 0,
        "near_duplicates": 0,
    }


def test_acquisition_uses_verified_cache(corpus, monkeypatch):
    manifest, raw, _ = corpus
    monkeypatch.setattr("latos.data.acquire.download", lambda _: pytest.fail("Unexpected download"))
    assert acquire(manifest, raw)["cached"] == 3


def test_corrupt_cache_is_not_overwritten(corpus):
    manifest, raw, output = corpus
    bad = raw / "fixture-train.txt"
    bad.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        acquire(manifest, raw)
    with pytest.raises(ValueError, match="checksum"):
        prepare(manifest, raw, output)
    assert bad.read_bytes() == b"corrupt"
    assert not output.exists()


def test_existing_output_is_preserved(corpus):
    manifest, raw, output = corpus
    output.mkdir()
    sentinel = output / "important.txt"
    sentinel.write_text("keep me")
    with pytest.raises(ValueError, match="already exists"):
        prepare(manifest, raw, output)
    assert sentinel.read_text() == "keep me"


@pytest.mark.parametrize("identity", ["document_id", "group_id"])
def test_manifest_rejects_identity_leakage(corpus, identity):
    manifest, _, _ = corpus
    data = json.loads(manifest.read_text())
    data["sources"][1][identity] = data["sources"][0][identity]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="crosses splits"):
        load_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "../escape"),
        ("sha256", "bad"),
        ("bytes", True),
        ("bytes", 4_000_001),
        ("language", "fr"),
        ("license", ""),
        ("path", "../../outside.txt"),
    ],
)
def test_manifest_rejects_invalid_source(corpus, field, value):
    manifest, _, _ = corpus
    data = json.loads(manifest.read_text())
    data["sources"][0][field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_manifest(manifest)


def test_changed_download_never_becomes_cache(corpus, monkeypatch):
    manifest, raw, _ = corpus
    data = json.loads(manifest.read_text())
    source = data["sources"][0]
    del source["path"]
    source["url"] = "https://example.org/book.txt"
    manifest.write_text(json.dumps(data))
    target = raw / f"{source['id']}.txt"
    target.unlink()
    monkeypatch.setattr("latos.data.acquire.download", lambda _: b"changed")
    monkeypatch.setattr("latos.data.acquire.time.sleep", lambda _: None)
    with pytest.raises(ValueError, match="source changed"):
        acquire(manifest, raw)
    assert not target.exists()
    assert len(list(raw.iterdir())) == 2


def test_https_redirect_cannot_downgrade():
    with pytest.raises(ValueError, match="remain HTTPS"):
        HTTPSRedirects().redirect_request(None, None, 302, "", {}, "http://example.org/book")


def test_markers_are_required_and_excluded():
    text = (
        b"Language: English\r\nLicense notice\r\n"
        b"*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\r\n"
        b"The original body.\r\n"
        b"*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\r\nLicense footer"
    )
    assert extract_body(text, "gutenberg").strip() == "The original body."
    with pytest.raises(ValueError, match="boundary markers"):
        extract_body(text.replace(b"END OF", b"FINISH OF"), "gutenberg")
    with pytest.raises(ValueError, match="declare English"):
        extract_body(text.replace(b"Language: English", b"Language: French"), "gutenberg")
    with pytest.raises(UnicodeDecodeError):
        extract_body(b"\xff", "text")


def test_unicode_and_whitespace_cleaning():
    text = (FIXTURE / "train.txt").read_text().split("\n\n")[0]
    assert clean_paragraph(text.replace(" ", " \n\t ")) == (text, None)
    assert clean_paragraph(text + "\x00")[1] == "invalid_character"
    assert clean_paragraph(text + "\ufffd")[1] == "invalid_character"
    assert clean_paragraph("brief")[1] == "length"
    assert clean_paragraph("日本語の文章です。" * 30)[1] in ("few_words", "script_ratio")
    decomposed = text.replace("Mara", "Ma\u0301ra")
    cleaned, reason = clean_paragraph(decomposed)
    assert reason is None and cleaned.startswith("Mára")


def test_prose_about_production_is_not_a_credit_notice():
    text = (
        "The marks were produced by water moving across the stone for many years. "
        "A visitor can see how the shallow channels follow the slope toward the river."
    )
    assert clean_paragraph(text) == (text, None)
    assert clean_paragraph("Produced by the volunteer team. " + text)[1] == "boilerplate"
    assert clean_paragraph("word " + "123456 " * 25)[1] == "few_words"


def test_index_matches_brute_force_jaccard():
    rng = random.Random(17)
    base = [f"word{chr(97 + i)}" for i in range(26)]
    texts = [" ".join(base)]
    for _ in range(80):
        words = base.copy()
        for _ in range(rng.randrange(1, 8)):
            words[rng.randrange(len(words))] = rng.choice(["river", "lake", "garden", "road"])
        texts.append(" ".join(words))
    index = DuplicateIndex()
    previous = []
    for text in texts:
        grams = shingles(text)
        expected = any(
            text.casefold() == other.casefold() or len(grams & old) * 100 >= 80 * len(grams | old)
            for other, old in previous
        )
        assert (index.match(text, grams) is not None) == expected
        index.add(text, grams)
        previous.append((text, grams))


def test_shingle_threshold_boundary():
    index = DuplicateIndex()
    grams = frozenset((str(i),) for i in range(10))
    index.add("one", grams)
    assert index.match("two", frozenset((str(i),) for i in range(8))) == (0, "near")
    assert index.match("three", frozenset((str(i),) for i in range(7))) is None


def test_duplicate_selection_is_order_independent():
    text = (FIXTURE / "train.txt").read_text().split("\n\n")[0]
    records = [
        {"id": split, "source_id": split, "paragraph": 0, "split": split, "text": text}
        for split in ("train", "validation", "test")
    ]
    kept, removed = deduplicate(records)
    assert kept[0]["split"] == "test" and len(removed) == 2
    assert deduplicate(list(reversed(records))) == (kept, removed)


def test_audit_detects_modified_output(corpus):
    manifest, raw, output = corpus
    prepare(manifest, raw, output)
    with (output / "train.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="checksum"):
        audit(manifest, output)


def test_audit_checks_identity_even_if_report_hash_is_updated(corpus):
    manifest, raw, output = corpus
    prepare(manifest, raw, output)
    path = output / "train.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["split"] = "test"
    data = b"".join(canonical_json(r) for r in records)
    path.write_bytes(data)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    report["splits"]["train"].update(sha256=sha256(data), bytes=len(data))
    report_path.write_bytes(canonical_json(report))
    with pytest.raises(ValueError, match="crosses splits"):
        audit(manifest, output)


def test_no_split_may_be_empty_after_cleaning(corpus):
    manifest, raw, output = corpus
    data = json.loads(manifest.read_text())
    source = data["sources"][1]
    tiny = b"too short"
    (raw / f"{source['id']}.txt").write_bytes(tiny)
    source.update(bytes=len(tiny), sha256=sha256(tiny))
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="No records remain"):
        prepare(manifest, raw, output)
    assert not output.exists()


def test_manifest_order_does_not_change_split_bytes(corpus):
    manifest, raw, output = corpus
    prepare(manifest, raw, output)
    data = json.loads(manifest.read_text())
    data["sources"].reverse()
    manifest.write_text(json.dumps(data))
    second = output.with_name("reordered")
    prepare(manifest, raw, second)
    for split in ("train", "validation", "test"):
        assert (output / f"{split}.jsonl").read_bytes() == (second / f"{split}.jsonl").read_bytes()


def test_cli_offline_pipeline(corpus):
    manifest, raw, output = corpus
    result = subprocess.run(
        [
            "latos",
            "data",
            "prepare",
            "--manifest",
            str(manifest),
            "--raw-dir",
            str(raw),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["manifest_id"] == "tiny-english-v1"
    result = subprocess.run(
        ["latos", "data", "audit", "--manifest", str(manifest), "--output-dir", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and json.loads(result.stdout)["status"] == "ok"


def test_all_public_manifests_validate():
    real = FIXTURE.parents[1] / "manifests/english-books-v1.json"
    assert len(load_manifest(real)["sources"]) == 12
    assert len(load_manifest(FIXTURE / "manifest.json")["sources"]) == 3
