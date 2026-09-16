"""Device policy and actual CPU arithmetic checks."""

import pytest

from latos import doctor


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        ({"cpu": True, "mps": False, "cuda": False}, "cpu"),
        ({"cpu": True, "mps": True, "cuda": False}, "mps"),
        ({"cpu": True, "mps": True, "cuda": True}, "cuda"),
    ],
)
def test_auto_selection(available, expected):
    assert doctor.select_device("auto", available) == expected


def test_explicit_cpu_overrides_accelerator():
    assert doctor.select_device("cpu", {"cpu": True, "mps": True, "cuda": True}) == "cpu"


@pytest.mark.parametrize("device", ["cuda", "mps", "unknown"])
def test_unavailable_or_unknown_device_is_not_replaced(device):
    with pytest.raises(ValueError):
        doctor.select_device(device, {"cpu": True, "mps": False, "cuda": False})


def test_cpu_diagnostic():
    report = doctor.diagnose("cpu")
    assert report["status"] == "ok"
    assert report["tested_device"] == "cpu"
    assert report["precision"] == "float32"
    assert report["python_version"]
    assert report["torch_version"]
    assert report["disk_free_bytes"] > 0


def test_unavailable_device_is_reported(monkeypatch):
    monkeypatch.setattr(
        doctor, "available_backends", lambda: {"cpu": True, "mps": False, "cuda": False}
    )
    report = doctor.diagnose("mps")
    assert report["status"] == "error"
    assert report["tested_device"] is None
    assert "unavailable" in report["error"]


def test_advertised_device_can_fail_execution(monkeypatch):
    monkeypatch.setattr(
        doctor, "available_backends", lambda: {"cpu": True, "mps": True, "cuda": False}
    )

    def broken_device(device):
        raise RuntimeError("Device allocation failed")

    monkeypatch.setattr(doctor, "smoke_check", broken_device)
    report = doctor.diagnose("auto")
    assert report["status"] == "error"
    assert report["tested_device"] == "mps"
    assert "Device allocation failed" in report["error"]


def test_missing_memory_api_is_unknown(monkeypatch):
    def unavailable(name):
        raise ValueError("Unsupported")

    monkeypatch.setattr(doctor.os, "sysconf", unavailable)
    assert doctor.physical_memory_bytes() is None


def test_numpy_tensor_interoperation():
    import numpy as np
    import torch

    original = np.array([1.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal((torch.from_numpy(original) * 2).numpy(), [2.0, 4.0])


def test_smoke_check_uses_float32_even_if_default_changes(monkeypatch):
    import torch

    observed = []
    original_tensor = torch.tensor

    def record_tensor(*args, **kwargs):
        tensor = original_tensor(*args, **kwargs)
        observed.append(tensor.dtype)
        return tensor

    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        monkeypatch.setattr(torch, "tensor", record_tensor)
        doctor.smoke_check("cpu")
    finally:
        torch.set_default_dtype(previous)
    assert observed and all(dtype == torch.float32 for dtype in observed)
