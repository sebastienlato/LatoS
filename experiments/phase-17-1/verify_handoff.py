"""Verify the supplied archive before extraction or execution; standard library only."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from common import hash_file, require, safe_path, verify_bundle


def verify(archive, expected, destination):
    require(hash_file(archive) == expected, "Supplied ZIP digest mismatch")
    require(not destination.exists(), "Extract only into a new tools directory")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        manifest = json.loads(z.read("bundle.json"))
        require(
            len(names) == len(set(names))
            and set(names) == set(manifest["files"]) | {"bundle.json"},
            "ZIP members differ",
        )
        require(
            sum(i.file_size for i in z.infolist()) < 2 * 1024**2,
            "Unexpected controller bundle size",
        )
        destination.mkdir()
        for name in names:
            path = safe_path(destination, name)
            raw = z.read(name)
            if name != "bundle.json":
                item = manifest["files"][name]
                require(
                    len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"],
                    "ZIP member corrupt",
                )
            path.write_bytes(raw)
    verify_bundle(destination)
    return {
        "verified_members": len(names),
        "source_commit": manifest["source_commit"],
        "optimizer_updates": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.sha256, args.destination), indent=2))
