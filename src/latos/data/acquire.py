"""Acquire bounded, checksum-pinned files without replacing an existing raw source."""

import os
import tempfile
import time
import urllib.request
from pathlib import Path

from latos.data.manifest import load_manifest, local_source, sha256, verified_bytes


class HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            raise ValueError("Source redirect must remain HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(source: dict) -> bytes:
    request = urllib.request.Request(source["url"], headers={"User-Agent": "LatoS-data/0.2"})
    opener = urllib.request.build_opener(HTTPSRedirects())
    with opener.open(request, timeout=30) as response:
        return response.read(source["bytes"] + 1)


def acquire(manifest_path: Path, raw_dir: Path) -> dict:
    manifest = load_manifest(manifest_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "copied": 0, "cached": 0}
    for source in sorted(manifest["sources"], key=lambda s: s["id"]):
        target = raw_dir / f"{source['id']}.txt"
        if target.exists():
            verified_bytes(target, source)
            counts["cached"] += 1
            continue
        if "path" in source:
            data = verified_bytes(local_source(source, manifest_path), source)
            kind = "copied"
        else:
            data = download(source)
            kind = "downloaded"
            time.sleep(2)  # Respect the source mirror with serial, paced downloads.
        if len(data) != source["bytes"] or sha256(data) != source["sha256"]:
            raise ValueError(f"Downloaded source changed: {source['id']}; no cache file written")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=raw_dir, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            # Hard-linking publishes complete bytes atomically and refuses to overwrite.
            os.link(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        counts[kind] += 1
    return {"manifest_id": manifest["id"], **counts}
