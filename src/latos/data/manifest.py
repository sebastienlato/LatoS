"""Validate source identities, provenance, and fixed document-level splits."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

SPLITS = ("train", "validation", "test")
MAX_SOURCE_BYTES = 4_000_000
MAX_TOTAL_BYTES = 20_000_000


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def local_source(source: dict, manifest_path: Path) -> Path:
    root = manifest_path.resolve().parent
    path = (root / source["path"]).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Local source must be inside the manifest directory")
    return path


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Expected manifest schema_version 1")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", str(manifest.get("id", ""))):
        raise ValueError("Manifest id must be a safe lowercase identifier")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Manifest must contain sources")
    seen = set()
    assignments: dict[tuple[str, str], str] = {}
    total = 0
    required = (
        "id",
        "document_id",
        "group_id",
        "title",
        "creator",
        "language",
        "split",
        "format",
        "license",
        "license_url",
        "metadata_url",
        "revision",
        "retrieved_at",
        "sha256",
    )
    for source in sources:
        if not isinstance(source, dict) or any(
            not isinstance(source.get(key), str) or not source[key].strip() for key in required
        ):
            raise ValueError("Each source needs string identity, provenance, and checksum fields")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", source["id"]):
            raise ValueError("Invalid source id")
        if source["id"] in seen:
            raise ValueError("Duplicate source id")
        seen.add(source["id"])
        if source["split"] not in SPLITS or source["language"] != "en":
            raise ValueError("Sources need an English language declaration and a valid split")
        if source["format"] not in ("gutenberg", "text"):
            raise ValueError("Unsupported source format")
        if source["format"] == "gutenberg" and not isinstance(source.get("body_start"), str):
            raise ValueError("Gutenberg sources require a reviewed body_start marker")
        if source["format"] == "gutenberg" and not source["body_start"].strip():
            raise ValueError("Empty body_start marker")
        if not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
            raise ValueError("Invalid SHA-256")
        size = source.get("bytes")
        if type(size) is not int or not 0 < size <= MAX_SOURCE_BYTES:
            raise ValueError("Source size is outside the bounded corpus budget")
        total += size
        if ("url" in source) == ("path" in source):
            raise ValueError("Specify exactly one source URL or local path")
        if "url" in source:
            url = urlsplit(source["url"])
            if url.scheme != "https" or not url.hostname or url.username or url.password:
                raise ValueError("Remote sources must use HTTPS without credentials")
        else:
            local_source(source, path)
        for key in ("document_id", "group_id"):
            identity = (key, source[key])
            if assignments.setdefault(identity, source["split"]) != source["split"]:
                raise ValueError(f"{key} crosses splits: {source[key]}")
    if total > MAX_TOTAL_BYTES:
        raise ValueError("Manifest exceeds the total corpus byte budget")
    if {source["split"] for source in sources} != set(SPLITS):
        raise ValueError("Manifest must reserve train, validation, and test sources")
    return manifest


def verified_bytes(path: Path, source: dict) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(source["bytes"] + 1)
    if len(data) != source["bytes"] or sha256(data) != source["sha256"]:
        raise ValueError(f"Size or checksum mismatch for {source['id']}; cached bytes preserved")
    return data
