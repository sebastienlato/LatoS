"""Check platform selection and complete wheel availability without foreign execution."""

import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
from packaging.markers import Marker, default_environment
from packaging.tags import compatible_tags, cpython_tags, mac_platforms
from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
LOCK = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))


def environment(system, machine):
    return {
        **default_environment(),
        "sys_platform": system,
        "platform_machine": machine,
        "platform_system": {"darwin": "Darwin", "linux": "Linux", "win32": "Windows"}[system],
        "os_name": "nt" if system == "win32" else "posix",
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
        "implementation_version": "3.14.7",
        "python_full_version": "3.14.7",
        "python_version": "3.14",
    }


def matches(marker, env):
    return Marker(marker).evaluate(env)


def selected_packages(lock, env):
    """Follow runtime and development dependency edges in uv's resolved graph."""
    root = next(p for p in lock["package"] if p["name"] == "latos")
    pending = [*root["dependencies"], *root["dev-dependencies"]["dev"]]
    selected = {}
    while pending:
        dependency = pending.pop()
        if "marker" in dependency and not matches(dependency["marker"], env):
            continue
        candidates = [
            p
            for p in lock["package"]
            if p["name"] == dependency["name"]
            and all(p[key] == dependency[key] for key in ("version", "source") if key in dependency)
            and (
                not p.get("resolution-markers")
                or any(matches(marker, env) for marker in p["resolution-markers"])
            )
        ]
        assert len(candidates) == 1, dependency
        package = candidates[0]
        name = package["name"]
        if name in selected:
            assert selected[name] == package
            continue
        selected[name] = package
        pending.extend(package.get("dependencies", []))
    return selected


@pytest.mark.parametrize(
    ("system", "machine", "torch_version", "registry"),
    [
        ("darwin", "arm64", "2.14.0", "https://pypi.org/simple"),
        ("linux", "x86_64", "2.14.0+cpu", "https://download.pytorch.org/whl/cpu"),
        ("win32", "AMD64", "2.14.0+cu130", "https://download.pytorch.org/whl/cu130"),
        ("win32", "x86_64", "2.14.0+cu130", "https://download.pytorch.org/whl/cu130"),
    ],
)
def test_locked_platform_has_correct_backend_and_binary_dependencies(
    system, machine, torch_version, registry
):
    env = environment(system, machine)
    assert any(matches(marker, env) for marker in PROJECT["tool"]["uv"]["environments"])
    assert any(matches(marker, env) for marker in LOCK["supported-markers"])
    packages = selected_packages(LOCK, env)
    assert packages["torch"]["version"] == torch_version
    assert packages["torch"]["source"]["registry"] == registry
    assert ("colorama" in packages) == (system == "win32")
    if system == "darwin":
        platforms = list(mac_platforms(version=(14, 0), arch="arm64"))
    elif system == "linux":
        platforms = [
            "manylinux_2_28_x86_64",
            "manylinux_2_27_x86_64",
            "manylinux_2_17_x86_64",
            "manylinux2014_x86_64",
        ]
    else:
        platforms = ["win_amd64"]
    tags = set(cpython_tags((3, 14), abis=["cp314"], platforms=platforms))
    tags.update(compatible_tags((3, 14), interpreter="cp314", platforms=platforms))
    for name, package in packages.items():
        wheels = [
            wheel
            for wheel in package.get("wheels", [])
            if tags.intersection(
                parse_wheel_filename(unquote(urlsplit(wheel["url"]).path.rsplit("/", 1)[-1]))[3]
            )
        ]
        assert wheels, f"No CPython 3.14 wheel for {name} on {system}/{machine}"
        assert all(wheel.get("hash", "").startswith("sha256:") for wheel in wheels)


@pytest.mark.parametrize(
    ("system", "machine"),
    [
        ("darwin", "x86_64"),
        ("linux", "aarch64"),
        ("win32", "ARM64"),
        ("win32", "x86"),
    ],
)
def test_unvalidated_architectures_are_not_admitted(system, machine):
    env = environment(system, machine)
    for markers in (PROJECT["tool"]["uv"]["environments"], LOCK["supported-markers"]):
        assert not any(matches(marker, env) for marker in markers)


def test_accelerator_indexes_are_explicit_and_only_route_torch():
    settings = PROJECT["tool"]["uv"]
    assert set(settings["sources"]) == {"torch"}
    assert all(index["explicit"] is True for index in settings["index"])
    for system, machine in [("darwin", "arm64"), ("linux", "x86_64"), ("win32", "AMD64")]:
        routed = [
            s["index"]
            for s in settings["sources"]["torch"]
            if matches(s["marker"], environment(system, machine))
        ]
        assert (
            routed == {"darwin": [], "linux": ["pytorch-cpu"], "win32": ["pytorch-cu130"]}[system]
        )
