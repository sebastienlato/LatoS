"""Acquire only reviewed, hash-pinned HTTPS inputs without adopting changed bytes."""

import datetime
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported acquisition schema")
    names = set()
    total = 0
    for source in manifest["sources"]:
        if not all(source.get(k) for k in ("id", "revision", "license", "rights", "origin")):
            raise ValueError("Missing provenance or rights")
        for item in source["files"]:
            name = item["file"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in names or name in (".", ".."):
                raise ValueError("Unsafe or duplicate filename")
            if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                raise ValueError("Invalid SHA-256")
            if type(item["bytes"]) is not int or not 0 < item["bytes"] <= 700_000_000:
                raise ValueError("Input size budget exceeded")
            if not item["url"].startswith("https://"):
                raise ValueError("Acquisition requires HTTPS")
            total += item["bytes"]
            names.add(name)
    if total > 2_000_000_000:
        raise ValueError("Total acquisition budget exceeded")
    return manifest


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            raise ValueError("Refusing insecure redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def verify(path: Path, item: dict) -> None:
    if (
        path.is_symlink()
        or path.stat().st_size != item["bytes"]
        or file_hash(path) != item["sha256"]
    ):
        raise ValueError(f"Input integrity failure: {path.name}")


def acquire(manifest_path: Path, destination: Path) -> dict:
    manifest = read_manifest(manifest_path)
    destination.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(HTTPSRedirect())
    report = {"manifest_sha256": file_hash(manifest_path), "files": {}}
    for source in manifest["sources"]:
        for item in source["files"]:
            path = destination / item["file"]
            if path.exists():
                verify(path, item)
            else:
                partial = path.with_suffix(path.suffix + ".partial")
                # Failed partials are evidence: do not silently overwrite them.
                with partial.open("xb") as out:
                    request = urllib.request.Request(
                        item["url"], headers={"User-Agent": "LatoS-Data2/1"}
                    )
                    with opener.open(request, timeout=90) as response:
                        size = 0
                        while chunk := response.read(1024 * 1024):
                            size += len(chunk)
                            if size > item["bytes"]:
                                raise ValueError("Download exceeds pinned size")
                            out.write(chunk)
                verify(partial, item)
                os.rename(partial, path)
            report["files"][item["file"]] = item["sha256"]
            print(f"Verified {item['file']} ({item['bytes']} bytes)", flush=True)
    receipt = destination / ("acquisition-" + report["manifest_sha256"][:16] + ".json")
    if not receipt.exists():
        write_json(
            receipt,
            {
                **report,
                "verified_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "note": (
                    "First completed acquisition/cache verification for this manifest; "
                    "timestamp is not part of deterministic corpus identity."
                ),
            },
        )
    return report
