"""Exercise the installed package and real process exit codes."""

import json
import subprocess
import sys

import pytest

from latos import __version__, cli, doctor


@pytest.mark.parametrize("args", [[], ["--help"], ["doctor", "--help"]])
def test_help(args):
    result = subprocess.run([sys.executable, "-m", "latos", *args], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_version():
    result = subprocess.run(["latos", "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == f"LatoS {__version__}"


def test_json_doctor():
    result = subprocess.run(
        ["latos", "doctor", "--device", "cpu", "--json"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema_version"] == 1
    assert report["status"] == "ok"
    assert report["tested_device"] == "cpu"


def test_invalid_command():
    result = subprocess.run(["latos", "train"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_doctor_failure_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor, "available_backends", lambda: {"cpu": True, "mps": False, "cuda": False}
    )
    assert cli.main(["doctor", "--device", "cuda", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_help_does_not_import_torch():
    result = subprocess.run(
        [sys.executable, "-c", "import latos.cli, sys; assert 'torch' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_missing_runtime_returns_json_error():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torch'] = None; "
            "from latos.cli import main; raise SystemExit(main(['doctor', '--json']))",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "error"
