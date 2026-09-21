"""Release archive boundary: allowlisted bytes, corruption, links and no overwrite."""

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "release_package", ROOT / "experiments/phase-8/package.py"
)
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)


@pytest.fixture
def inputs(tmp_path):
    root, tiny = tmp_path / "source", tmp_path / "tiny"
    inventory = {}
    for name, source in PACK.ARTIFACTS.items():
        path = tiny / source
        path.parent.mkdir(parents=True, exist_ok=True)
        data = source.encode()
        path.write_bytes(data)
        inventory[name] = PACK.identity(data)
    for source in PACK.NOTICES.values():
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    policy = root / "experiments/phase-8/tiny-artifacts.json"
    policy.parent.mkdir(parents=True)
    policy.write_text(json.dumps(inventory))
    # Neither extra artifacts nor sensitive sibling material may enter the archive.
    (tiny / "training.safetensors").write_bytes(b"not permitted")
    (root / ".private").mkdir()
    (root / ".private/context.md").write_text("private sentinel")
    return root, tiny, tmp_path / "bundle.zip"


def test_package_is_exact_deterministic_and_refuses_overwrite(inputs):
    root, tiny, output = inputs
    result = PACK.package(tiny, output, root=root)
    second = output.with_name("second.zip")
    assert PACK.package(tiny, second, root=root) == result
    assert second.read_bytes() == output.read_bytes()
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {*PACK.ARTIFACTS, *PACK.NOTICES, "MANIFEST.json"}
        manifest = json.loads(archive.read("MANIFEST.json"))
        for name, expected in manifest["members"].items():
            assert PACK.identity(archive.read(name)) == expected
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        PACK.package(tiny, output, root=root)
    assert output.read_bytes() == before


def test_corrupt_or_expanded_inventory_fails_before_output(inputs):
    root, tiny, output = inputs
    path = tiny / "final/model/config.json"
    original = path.read_bytes()
    path.write_bytes(original + b"corrupt")
    with pytest.raises(ValueError, match="identity mismatch"):
        PACK.package(tiny, output, root=root)
    assert not output.exists()
    path.write_bytes(original)
    policy = root / "experiments/phase-8/tiny-artifacts.json"
    inventory = json.loads(policy.read_text())
    inventory["unexpected"] = PACK.identity(b"extra")
    policy.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="allowlist"):
        PACK.package(tiny, output, root=root)
    assert not output.exists()


def test_symlinked_artifact_directory_is_rejected(inputs):
    root, tiny, output = inputs
    folder = tiny / "final/model"
    actual = tiny / "original"
    folder.rename(actual)
    try:
        folder.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("Host does not permit symlink creation")
    with pytest.raises(ValueError, match="Symlink"):
        PACK.package(tiny, output, root=root)
    assert not output.exists()
