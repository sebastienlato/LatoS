"""Prepare and audit source-separated paragraph records without network access."""

import json
import platform
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

from latos import __version__
from latos.data.dedup import WORD, DuplicateIndex, deduplicate, shingles
from latos.data.manifest import SPLITS, canonical_json, load_manifest, sha256, verified_bytes

POLICY = {
    "id": "english-paragraphs-v1",
    "min_chars": 120,
    "max_chars": 8000,
    "min_words": 20,
    "min_ascii_letter_percent": 90,
    "max_records": 50_000,
    "normalization": "NFC; collapse whitespace within each paragraph",
    "near_duplicate": "casefolded word 5-shingles; set Jaccard >= 80%; exhaustive index",
    "duplicate_priority": ["test", "validation", "train"],
}


def extract_body(data: bytes, format_name: str) -> str:
    text = data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    if format_name == "text":
        return text
    if format_name != "gutenberg":
        raise ValueError("Unsupported text format")
    starts = list(
        re.finditer(
            r"^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .+?\*\*\*\s*$", text, re.M
        )
    )
    ends = list(
        re.finditer(
            r"^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .+?\*\*\*\s*$", text, re.M
        )
    )
    if len(starts) != 1 or len(ends) != 1 or starts[0].end() >= ends[0].start():
        raise ValueError("Expected one ordered pair of Gutenberg boundary markers")
    if not re.search(r"^Language: English\s*$", text[: starts[0].start()], re.M):
        raise ValueError("Gutenberg source does not declare English")
    return text[starts[0].end() : ends[0].start()]


def clean_paragraph(text: str) -> tuple[str, str | None]:
    if "\ufffd" in text or any(
        unicodedata.category(char).startswith("C") and char not in "\n\t" for char in text
    ):
        return "", "invalid_character"
    cleaned = " ".join(unicodedata.normalize("NFC", text).split())
    if not POLICY["min_chars"] <= len(cleaned) <= POLICY["max_chars"]:
        return cleaned, "length"
    if len(WORD.findall(cleaned)) < POLICY["min_words"]:
        return cleaned, "few_words"
    if re.search(r"project gutenberg|gutenberg\.org|\[illustration", cleaned, re.I) or re.match(
        r"Produced by\b", cleaned, re.I
    ):
        return cleaned, "boilerplate"
    letters = [char for char in cleaned if char.isalpha()]
    ascii_letters = sum(char.isascii() for char in letters)
    if not letters or ascii_letters * 100 < POLICY["min_ascii_letter_percent"] * len(letters):
        return cleaned, "script_ratio"
    return cleaned, None


def implementation_hash() -> str:
    root = Path(__file__).parent
    return sha256(
        b"".join((root / name).read_bytes() for name in ("manifest.py", "dedup.py", "prepare.py"))
    )


def prepare(manifest_path: Path, raw_dir: Path, output_dir: Path) -> dict:
    manifest = load_manifest(manifest_path)
    if output_dir.exists():
        raise ValueError("Output directory already exists; choose a new destination")
    candidates, source_reports = [], []
    for source in sorted(manifest["sources"], key=lambda s: s["id"]):
        data = verified_bytes(raw_dir / f"{source['id']}.txt", source)
        body = extract_body(data, source["format"])
        if source["format"] == "gutenberg":
            marker = source["body_start"]
            if body.count(marker) != 1:
                raise ValueError(f"Body start marker is not unique: {source['id']}")
            body = body[body.index(marker) :]
        rejected: Counter[str] = Counter()
        count = 0
        # The book/document split was fixed in the manifest BEFORE paragraph extraction.
        for number, paragraph in enumerate(re.split(r"\n\s*\n", body)):
            if not paragraph.strip():
                continue
            count += 1
            text, reason = clean_paragraph(paragraph)
            if reason:
                rejected[reason] += 1
                continue
            digest = sha256(text.encode())
            candidates.append(
                {
                    "id": f"{source['id']}:{number}:{digest}",
                    "source_id": source["id"],
                    "document_id": source["document_id"],
                    "group_id": source["group_id"],
                    "split": source["split"],
                    "paragraph": number,
                    "text_sha256": digest,
                    "text": text,
                }
            )
            if len(candidates) > POLICY["max_records"]:
                raise ValueError("Corpus exceeds the in-memory record budget")
        source_reports.append(
            {
                "source_id": source["id"],
                "raw_sha256": source["sha256"],
                "raw_bytes": len(data),
                "paragraphs": count,
                "filtered": dict(rejected),
            }
        )
    kept, removed = deduplicate(candidates)
    report = {
        "schema_version": 1,
        "manifest_id": manifest["id"],
        "manifest_sha256": sha256(manifest_path.read_bytes()),
        "policy": POLICY,
        "implementation_sha256": implementation_hash(),
        "latos_version": __version__,
        "python_version": platform.python_version(),
        "unicode_version": unicodedata.unidata_version,
        "sources": source_reports,
        "candidate_records": len(candidates),
        "duplicates": dict(Counter(r["kind"] for r in removed)),
        "cross_split_duplicates_removed": sum(
            r["removed_split"] != r["kept_split"] for r in removed
        ),
        "splits": {},
    }
    payloads = {"duplicates.jsonl": b"".join(canonical_json(r) for r in removed)}
    for split in SPLITS:
        records = [r for r in kept if r["split"] == split]
        if not records:
            raise ValueError(f"No records remain in {split}; review sources before proceeding")
        data = b"".join(canonical_json(r) for r in records)
        payloads[f"{split}.jsonl"] = data
        report["splits"][split] = {
            "records": len(records),
            "documents": len({r["document_id"] for r in records}),
            "characters": sum(len(r["text"]) for r in records),
            "words": sum(len(r["text"].split()) for r in records),
            "bytes": len(data),
            "sha256": sha256(data),
        }
    report["duplicates_sha256"] = sha256(payloads["duplicates.jsonl"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".latos-prepare-", dir=output_dir.parent))
    try:
        for name, data in payloads.items():
            (staging / name).write_bytes(data)
        (staging / "report.json").write_bytes(canonical_json(report))
        audit(manifest_path, staging)
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def audit(manifest_path: Path, output_dir: Path) -> dict:
    """Re-read output, validate hashes/identities/quality, and check every duplicate candidate."""
    manifest = load_manifest(manifest_path)
    sources = {source["id"]: source for source in manifest["sources"]}
    report = json.loads((output_dir / "report.json").read_text())
    if (
        report["manifest_sha256"] != sha256(manifest_path.read_bytes())
        or report["policy"] != POLICY
    ):
        raise ValueError("Report manifest or cleaning policy mismatch")
    if report["implementation_sha256"] != implementation_hash():
        raise ValueError("Report belongs to a different pipeline implementation")
    index = DuplicateIndex()
    identities = set()
    total = 0
    for split in SPLITS:
        data = (output_dir / f"{split}.jsonl").read_bytes()
        summary = report["splits"][split]
        if sha256(data) != summary["sha256"] or len(data) != summary["bytes"]:
            raise ValueError(f"Output checksum mismatch: {split}")
        records = [json.loads(line) for line in data.splitlines()]
        if not records or len(records) != summary["records"]:
            raise ValueError(f"Output count mismatch: {split}")
        for record in records:
            source = sources.get(record["source_id"])
            if (
                source is None
                or any(record[key] != source[key] for key in ("document_id", "group_id", "split"))
                or record["split"] != split
            ):
                raise ValueError("Output source identity crosses splits or is unknown")
            text = record["text"]
            digest = sha256(text.encode())
            if (
                record["text_sha256"] != digest
                or record["id"] != f"{source['id']}:{record['paragraph']}:{digest}"
            ):
                raise ValueError("Invalid record identity or text checksum")
            if record["id"] in identities:
                raise ValueError("Repeated record identity")
            identities.add(record["id"])
            if clean_paragraph(text) != (text, None):
                raise ValueError("Output violates the cleaning policy")
            grams = shingles(text)
            if index.match(text, grams) is not None:
                raise ValueError("Duplicate output text under the declared metric")
            index.add(text, grams)
        expected = {
            "documents": len({r["document_id"] for r in records}),
            "characters": sum(len(r["text"]) for r in records),
            "words": sum(len(r["text"].split()) for r in records),
        }
        if any(summary[key] != value for key, value in expected.items()):
            raise ValueError("Reported corpus statistics do not match output")
        total += len(records)
    if sha256((output_dir / "duplicates.jsonl").read_bytes()) != report["duplicates_sha256"]:
        raise ValueError("Duplicate ledger checksum mismatch")
    return {"status": "ok", "records": total, "exact_duplicates": 0, "near_duplicates": 0}
