"""Exercise Git checkout conversion against the byte-pinned fixture contract."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from latos.data.acquire import acquire
from latos.data.prepare import audit, prepare

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path("data/fixtures/tiny")


@pytest.mark.parametrize(
    ("autocrlf", "eol", "control_crlf"),
    [
        ("true", "crlf", True),
        ("input", "crlf", False),
        ("false", "crlf", False),
        ("false", "lf", False),
    ],
)
def test_git_checkout_preserves_fixture_bytes(tmp_path, autocrlf, eol, control_crlf):
    repository = tmp_path / "repository"
    repository.mkdir()
    # Do not read or modify the user's system/global Git configuration or index.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repository), *args],
            env=env,
            check=True,
            capture_output=True,
        ).stdout

    git("init", "--quiet")
    manifest_bytes = (ROOT / FIXTURE / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    originals = {FIXTURE / "manifest.json": manifest_bytes}
    for source in manifest["sources"]:
        originals[FIXTURE / source["path"]] = (ROOT / FIXTURE / source["path"]).read_bytes()
    for path, data in originals.items():
        destination = repository / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        assert b"\r" not in data, "Fixture contract requires committed LF bytes"
    (repository / ".gitattributes").write_bytes((ROOT / ".gitattributes").read_bytes())
    (repository / "unrelated.txt").write_bytes(b"unrelated\ncontrol\n")
    git(
        "-c",
        "core.autocrlf=false",
        "add",
        ".gitattributes",
        "unrelated.txt",
        *(p.as_posix() for p in originals),
    )

    # Empty destination: exercise Git's real index-to-working-tree conversion.
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git(
        "-c",
        f"core.autocrlf={autocrlf}",
        "-c",
        f"core.eol={eol}",
        "checkout-index",
        "--all",
        f"--prefix={checkout.as_posix()}/",
    )
    assert (b"\r\n" in (checkout / "unrelated.txt").read_bytes()) == control_crlf
    for path, original in originals.items():
        assert git("show", f":{path.as_posix()}") == original
        assert (checkout / path).read_bytes() == original
    for source in manifest["sources"]:
        data = (checkout / FIXTURE / source["path"]).read_bytes()
        assert len(data) == source["bytes"]
        assert hashlib.sha256(data).hexdigest() == source["sha256"]

    manifest_path = checkout / FIXTURE / "manifest.json"
    assert acquire(manifest_path, tmp_path / "raw")["copied"] == 3
    report = prepare(manifest_path, tmp_path / "raw", tmp_path / "processed")
    assert {name: split["records"] for name, split in report["splits"].items()} == {
        "train": 3,
        "validation": 3,
        "test": 4,
    }
    assert audit(manifest_path, tmp_path / "processed")["records"] == 10

    # Deliberately change checkout bytes: verification must still reject CRLF.
    test_path = checkout / FIXTURE / "test.txt"
    test_path.write_bytes(test_path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="Size or checksum mismatch"):
        acquire(manifest_path, tmp_path / "corrupt-raw")
    assert manifest_path.read_bytes() == manifest_bytes
