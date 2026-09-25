"""Zero-update accounting, integrity and deadline tests; no CUDA/model training."""

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DIR = Path(
    os.environ.get("LATOS_SUBSET_TEST_TOOLS", ROOT / "experiments/phase-17-1/subset-confirmation")
)
sys.path.insert(0, str(DIR))
import cf_adapter as a  # noqa: E402
import cf_common as c  # noqa: E402
import cf_controller as controller  # noqa: E402
import cf_evidence as evidence  # noqa: E402
import cf_worker as worker  # noqa: E402


@pytest.fixture(autouse=True)
def no_optimization(monkeypatch):
    from latos.evaluation import runner
    from latos.model.network import LatoModel
    from latos.training.engine import Trainer

    def forbidden(*args, **kwargs):
        raise AssertionError("Zero-update checks must not optimize")

    monkeypatch.setattr(Trainer, "update", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(LatoModel, "forward", forbidden)
    monkeypatch.setattr(runner, "run", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def test_fixed_accounting():
    assert c.selfcheck()["optimizer_updates"] == 0
    assert sum(j["updates_reserved"] for j in controller.fixtures()) == 52
    assert list(c.ARMS) == ["D", "E"]
    assert set(c.ENDPOINTS) == {"D2", "E2", "E4"}


def test_two_pass_target_conservation():
    from latos.training.data import ShuffleStream

    for size in (30, 32, 6105):
        stream = ShuffleStream(size, 161)
        for epoch in range(4):
            if epoch:
                stream.epoch += 1
                stream.order = torch.randperm(size, generator=stream.generator)
                stream.cursor = 0
            visited = []
            while stream.cursor < size:
                count = c.accumulation(size, stream.cursor, 2, 8)
                batches = [stream.take(2) for _ in range(count)]
                visited.extend(i for row in batches for i in row)
            assert sorted(visited) == list(range(size))
            assert stream.epoch == epoch
        if size == 6105:
            assert count == 5 and len(batches[-1]) == 1


def test_adapter_restores_config_and_tail():
    from latos.training.config import TrainingConfig

    config = TrainingConfig(batch_size=2, accumulation_steps=8, max_steps=764, warmup_steps=19)
    t = SimpleNamespace(config=config, stream=SimpleNamespace(size=6105, cursor=6096))

    def scripted():
        assert t.config.accumulation_steps == 5
        assert t.config.max_steps == 764 and t.config.warmup_steps == 19
        return {"microbatches": 5}

    t.update = scripted
    assert a.update(t)["microbatches"] == 5 and t.config is config

    def fail():
        raise ValueError("scripted failure")

    t.update = fail
    with pytest.raises(ValueError):
        a.update(t)
    assert t.config is config


def test_seed_order_and_pass_switch():
    from latos.training.data import ShuffleStream

    one, two = ShuffleStream(6105, 161), ShuffleStream(6105, 161)
    assert torch.equal(one.order, two.order)
    one.cursor = two.cursor = 6105
    a.advance_pass(SimpleNamespace(stream=one, step=382))
    a.advance_pass(SimpleNamespace(stream=two, step=382))
    assert one.epoch == 1 and one.cursor == 0 and torch.equal(one.order, two.order)
    two.cursor = 6105
    with pytest.raises(ValueError):
        a.advance_pass(SimpleNamespace(stream=two, step=383))


def test_schedule_contrast_and_dimensions():
    from latos.model import ModelConfig
    from latos.training.config import TrainingConfig

    short = TrainingConfig(max_steps=764, warmup_steps=19, learning_rate=0.0003, min_lr_ratio=0.1)
    long = dataclasses.replace(short, max_steps=1528)
    assert short.learning_rate_at(19) == long.learning_rate_at(19)
    assert short.learning_rate_at(764) < long.learning_rate_at(764)
    assert short.learning_rate_at(764) == long.learning_rate_at(1528)
    model = ModelConfig.load(ROOT / "configs/training2/selected-model.json")
    assert model.parameter_count == 34087424
    assert dataclasses.replace(model, n_layers=4).parameter_count == 21238272


def test_deadline_includes_preparation():
    with pytest.raises(TimeoutError):
        c.budget(180, now=180)
    assert c.budget(180, now=179) == 1
    assert sum(c.STAGES.values()) == 1740 and c.TOTAL == 1800


def test_child_is_killed_on_deadline(tmp_path):
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        c.child(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            dict(os.environ),
            tmp_path / "child.log",
            started + 0.2,
            tmp_path,
            lambda x: None,
        )
    assert time.monotonic() - started < 3


def test_completed_child_cannot_override_failure(tmp_path):
    with pytest.raises(ValueError, match="Child failed"):
        c.child(
            [sys.executable, "-c", "raise SystemExit(9)"],
            dict(os.environ),
            tmp_path / "child.log",
            time.monotonic() + 3,
            tmp_path,
            lambda x: None,
        )


def test_unsafe_members_and_immutable_records(tmp_path):
    for name in ("../escape", ".private/notes", "C:/outside", "a\\b", "/outside"):
        with pytest.raises(ValueError):
            c.safe(tmp_path, name)
    c.write(tmp_path / "once.json", {"ok": True})
    with pytest.raises(FileExistsError):
        c.write(tmp_path / "once.json", {})
    expected = {"once.json": c.record(tmp_path / "once.json")}
    c.verify(tmp_path, expected)
    (tmp_path / "once.json").write_text("changed")
    with pytest.raises(ValueError):
        c.verify(tmp_path, expected)


def test_torn_trace_rejected(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_bytes(b'{"step":1}')
    with pytest.raises(ValueError, match="Torn"):
        evidence.rows(path)


def test_cache_identities():
    cache = ROOT / "outputs/small-subset-cache"
    assert cache.is_dir(), "Prepared fixed cache required for this integration validation"
    manifest = c.read(DIR / "cache-identities.json")
    c.verify(cache, manifest["files"])
    # Match the bundle layout without copying corpus bytes.
    for split in ("train", "development"):
        assert c.read(cache / f"{split}.json")["identity"] == manifest["identities"][split]
    assert manifest["identities"]["train"]["targets"] == 1943015


def test_actual_cache_reader(tmp_path):
    import shutil

    shutil.copytree(ROOT / "outputs/small-subset-cache", tmp_path / "cache")
    train, dev = worker.datasets(tmp_path)
    assert train.identity == c.read(DIR / "cache-identities.json")["identities"]["train"]
    assert dev.identity["targets"] == 57036
    # Corrupt an ID without invoking a model.
    with (tmp_path / "cache/train.u32").open("r+b") as f:
        f.write(b"\x00\x00\x00\x00")
    with pytest.raises(ValueError):
        worker.datasets(tmp_path)


def test_extraction_rejects_wrong_hash(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../outside", "bad")
    with pytest.raises(ValueError):
        controller.extract(archive, tmp_path / "tools", "a" * 64)
    with pytest.raises(ValueError):
        controller.extract(archive, tmp_path / "tools", c.sha(archive))


def test_no_final_or_old_training_entrypoint():
    assert "final" not in c.ENDPOINTS
    assert all(s[1] <= 4 for s in c.ARMS.values())
    spec = importlib.util.spec_from_file_location("subset_guardian", DIR / "launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.guarded)
    assert "torch" not in (DIR / "launch.py").read_text().split("import argparse")[1]


def test_diagnostic_checkpoint_roundtrip_no_optimizer(tmp_path):
    from latos.model import ModelConfig, create_model
    from latos.training.config import TrainingConfig
    from latos.training.data import TokenDataset
    from latos.training.engine import Trainer

    cfg = ModelConfig(1, 260, 512, 8, 2, 1, 16, 10000.0, 1e-5, "a" * 64)
    train = TokenDataset(((1, 5, 2),) * 6105, 512, 260, "a" * 64, "train", "b" * 64)
    dev = dataclasses.replace(train, split="validation")
    trainer = Trainer(
        create_model(cfg, 161),
        TrainingConfig(
            sequence_length=512, batch_size=2, accumulation_steps=8, max_steps=382, warmup_steps=19
        ),
        train,
        dev,
    )
    spec = importlib.util.spec_from_file_location(
        "subset_fp", ROOT / "experiments/recovery-diagnostic/mechanics.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    state = m.snapshot(trainer)
    a.save_checkpoint(trainer, tmp_path / "checkpoint", {"scripted_CPU": True}, m)
    assert m.snapshot(trainer) == state and not trainer.optimizer.state
    saved = c.read(tmp_path / "checkpoint/diagnostic.json")
    assert saved["readback_exact"] and saved["kind"] == "subset-confirmation-no-resume-v1"
    c.verify(tmp_path / "checkpoint", saved["files"])
    with pytest.raises(ValueError):
        a.save_checkpoint(trainer, tmp_path / "checkpoint", {}, m)


def test_failure_fixture_mismatch_is_terminal(tmp_path):
    for name in ("ref", "resumed"):
        (tmp_path / "jobs" / name).mkdir(parents=True)
    fake = [
        {"input": {"hash": str(i)}, "state": {"v": i}, "metric": {"microbatches": 2}}
        for i in range(8)
    ]
    (tmp_path / "jobs/ref/updates.jsonl").write_bytes(b"".join(c.canonical(r) for r in fake))
    replay = fake[3:].copy()
    replay[0] = {**replay[0], "state": {"v": "mismatch"}}
    (tmp_path / "jobs/resumed/updates.jsonl").write_bytes(b"".join(c.canonical(r) for r in replay))
    with pytest.raises(ValueError, match="Fixture replay"):
        controller.check_fixture(
            tmp_path, {"reference": "ref", "id": "resumed", "fixture": "ordinary"}
        )


def test_guardian_suspends_assigns_and_kills_entire_job(monkeypatch):
    import ctypes

    spec = importlib.util.spec_from_file_location("guard_mock", DIR / "launch.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    calls = []

    class Function:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            calls.append((self.name, args))
            if self.name == "CreateJobObjectW":
                return 10
            if self.name == "CreateProcessW":
                assert args[5] == 4  # CREATE_SUSPENDED before job assignment.
                args[-1]._obj.process = 20
                args[-1]._obj.thread = 21
                return 1
            if self.name == "SetInformationJobObject":
                assert args[2]._obj.Basic.Flags == 0x2000
            if self.name == "WaitForSingleObject":
                return 258
            return 1

    api = SimpleNamespace(
        **{
            name: Function(name)
            for name in (
                "CreateJobObjectW",
                "SetInformationJobObject",
                "CreateProcessW",
                "AssignProcessToJobObject",
                "ResumeThread",
                "WaitForSingleObject",
                "TerminateJobObject",
                "TerminateProcess",
                "GetExitCodeProcess",
                "CloseHandle",
            )
        }
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **kw: api, raising=False)
    result = guard.guarded(["python.exe", "fixture.py"], time.monotonic() + 1)
    assert result["deadline_terminated"]
    names = [n for n, _ in calls]
    assert names.index("AssignProcessToJobObject") < names.index("ResumeThread")
    assert "TerminateJobObject" in names
    assert {a[0] for n, a in calls if n == "CloseHandle"} == {10, 20, 21}


def test_complete_scripted_evidence_and_tamper_rejection(tmp_path, monkeypatch):
    """2,292 fabricated records exercise the verifier, never an optimizer/model."""
    import hashlib
    import shutil

    from latos.data.manifest import canonical_json
    from latos.training.config import TrainingConfig

    shutil.copytree(ROOT / "outputs/small-subset-cache", tmp_path / "tools/cache")
    train = worker.datasets(tmp_path / "tools")[0]
    train_identity = train.identity
    spec = importlib.util.spec_from_file_location(
        "verify_fp", ROOT / "experiments/recovery-diagnostic/mechanics.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setitem(sys.modules, "runtime", SimpleNamespace(fingerprints=lambda: m))
    for arm, (_, passes, steps) in c.ARMS.items():
        job = tmp_path / "jobs" / arm
        job.mkdir(parents=True)
        config = TrainingConfig(
            sequence_length=512,
            batch_size=2,
            accumulation_steps=8,
            max_steps=steps,
            warmup_steps=19,
            learning_rate=0.0003,
            min_lr_ratio=0.1,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            max_grad_norm=1.0,
            seed=161,
        )
        rates = [config.learning_rate_at(i) for i in range(1, steps + 1)]
        c.write(job / "schedule.json", rates)
        model = c.read(ROOT / "configs/training2/selected-model.json")
        model["n_layers"] = 4
        initial = {
            "scripted": True,
            "model_config": model,
            "training_config": config.to_dict(),
            "optimizer": {},
            "model": {"scripted": "initial"},
            "global_rng": "scripted",
            "sampler": {"step": 0},
        }
        c.write(job / "initial.json", {"state": initial})
        cp = job / "step-00000000"
        cp.mkdir()
        c.write(cp / "diagnostic.json", {"state": initial, "files": {}, "readback_exact": True})
        gen = torch.Generator().manual_seed(161)
        total = 0
        records = []
        for epoch in range(passes):
            order = torch.randperm(6105, generator=gen)
            for cursor in range(0, 6105, 16):
                sig = m.input_signature(
                    SimpleNamespace(
                        stream=SimpleNamespace(cursor=cursor, order=order),
                        train_data=train,
                        config=SimpleNamespace(batch_size=2, accumulation_steps=8),
                    )
                )
                step = len(records) + 1
                total += sig["targets"]
                observed_cursor = min(cursor + 16, 6105)
                windows = epoch * 6105 + observed_cursor
                sampler = {
                    "epoch": epoch,
                    "cursor": observed_cursor,
                    "targets": total,
                    "step": step,
                    "order": m.tensor_digest(order),
                    "rng": m.tensor_digest(gen.get_state()),
                }
                state = {
                    "sampler": sampler,
                    "datasets": {"train": train_identity},
                    "scripted": True,
                    "model_config": model,
                    "training_config": config.to_dict(),
                    "model": {"scripted": str(step)},
                    "optimizer": {"scripted": str(step)},
                    "optimizer_groups": "scripted",
                    "global_rng": "scripted",
                }
                state["state_sha256"] = hashlib.sha256(canonical_json(state)).hexdigest()
                row = {
                    "input": sig,
                    "state": state,
                    "metric": {
                        "step": step,
                        "tokens_seen": total,
                        "windows_seen": windows,
                        "targets": sig["targets"],
                        "learning_rate": rates[step - 1],
                        "loss": 7.0,
                        "microbatches": 5 if observed_cursor == 6105 else 8,
                        "epoch": epoch,
                    },
                }
                records.append(row)
                if observed_cursor == 6105:
                    cp = job / f"step-{step:08d}"
                    cp.mkdir()
                    c.write(
                        cp / "diagnostic.json",
                        {"state": state, "files": {}, "readback_exact": True},
                    )
        (job / "updates.jsonl").write_bytes(b"".join(c.canonical(r) for r in records))
    assert evidence.verify_training(tmp_path)["serious_updates"] == 2292
    trace = tmp_path / "jobs/D/updates.jsonl"
    saved = evidence.rows(trace)
    saved[-1]["metric"]["microbatches"] = 8
    trace.write_bytes(b"".join(c.canonical(r) for r in saved))
    with pytest.raises(ValueError, match="Tail accounting"):
        evidence.verify_training(tmp_path)


def test_controller_fixed_sequence_and_failure_stop(tmp_path, monkeypatch):
    """Scripted subprocess receipts, explicitly not CUDA execution."""
    monkeypatch.setattr(
        controller,
        "os",
        SimpleNamespace(name="nt", environ={"LATOS_SUBSET_JOB_ACTIVE": "1"}, fsync=os.fsync),
    )
    manifest = {"files": {"cf_controller.py": {"sha256": "fixed"}}}

    def extract(archive, tools, digest):
        tools.mkdir()
        c.write(
            tools / "authorization.json",
            {
                "authorized": True,
                "attempt": "subset-confirmation-v1",
                "proposal_sha256": "fixed",
                "hard_wall_seconds": 1800,
                "new_paid_compute": 0,
                "phase18_authorized": False,
                "publication_authorized": False,
            },
        )
        return manifest

    monkeypatch.setattr(controller, "extract", extract)
    monkeypatch.setattr(controller, "sha", lambda p: "fixed")
    monkeypatch.setattr(
        controller.shutil, "disk_usage", lambda p: SimpleNamespace(free=20 * 1024**3)
    )
    monkeypatch.setattr(controller, "check_fixture", lambda *args: None)
    calls = []

    def fake_child(command, env, log, deadline, root, journal):
        if command[-1].endswith(".json"):
            req = c.read(Path(command[-1]))
            calls.append(req["id"])
            assert deadline <= begun + 1790
            job = Path(req["output"]) / "jobs" / req["id"]
            job.mkdir()
            c.write(job / "result.json", {"optimizer_updates": req.get("updates_reserved", 0)})
        else:
            c.write(root / "run/return.receipt.json", {"scripted": True})

    monkeypatch.setattr(controller, "child", fake_child)
    begun = time.monotonic()
    args = SimpleNamespace(
        started=begun,
        transfer=tmp_path,
        output=tmp_path / "run",
        archive=tmp_path / "archive.zip",
        sha256="fixed",
    )
    controller.run(args)
    assert calls == [
        "prepare",
        *[r["id"] for r in controller.fixtures()],
        "D",
        "E",
        "eval-reference",
        "eval-D2",
        "eval-E2",
        "eval-E4",
    ]
    assert c.read(tmp_path / "run/completed.json")["accepted_base"] is False


def test_failed_study_packages_existing_evidence_only(tmp_path, monkeypatch):
    study = tmp_path / "study"
    root = study / "run"
    root.mkdir(parents=True)
    c.write(study / "launch.json", {"started_monotonic": 1})
    c.write(root / "outcome.json", {"execution_complete": False, "accepted_base": False})
    c.write(root / "failure.json", {"scripted_failure": True})
    monkeypatch.setenv("LATOS_SUBSET_PACKAGE", "1")
    evidence.package(root, time.monotonic() + 5)
    receipt = c.read(root / "return.receipt.json")
    assert receipt["verified"]
    with zipfile.ZipFile(root / "subset-confirmation-return.zip") as z:
        returned = json.loads(z.read("return.json"))
        assert returned["outcome"]["execution_complete"] is False
        assert "guardian/launch.json" in returned["files"]
        assert "failure.json" in returned["files"]
    assert not (root / "comparisons.json").exists()


def test_expired_packaging_never_starts(tmp_path, monkeypatch):
    monkeypatch.setenv("LATOS_SUBSET_PACKAGE", "1")
    with pytest.raises(TimeoutError):
        evidence.package(tmp_path, time.monotonic() - 1)
    assert not list(tmp_path.iterdir())


def test_windows_guardian_rejects_mac_execution(tmp_path):
    if os.name == "nt":
        return  # Actual Job Object conformance runs inside the bounded Windows preparation.
    done = subprocess.run(
        [
            sys.executable,
            str(DIR / "launch.py"),
            "--transfer",
            str(tmp_path),
            "--archive",
            "unused.zip",
            "--sha256",
            "a" * 64,
        ],
        capture_output=True,
    )
    assert done.returncode != 0 and not list(tmp_path.iterdir())


def test_return_rejects_over_budget_before_reading_payloads(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "subset_review", ROOT / "experiments/phase-17-1/subset-confirmation/verify_return.py"
    )
    reviewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reviewer)
    archive = tmp_path / "return.zip"
    archive.write_bytes(b"not even parsed before the time check")
    receipt = tmp_path / "receipt.json"
    c.write(receipt, c.record(archive))
    completed = tmp_path / "completed.json"
    c.write(
        completed,
        {
            "return_receipt_sha256": c.sha(receipt),
            "total_elapsed_seconds": 1790,
            "execution_complete": True,
        },
    )
    guardian = tmp_path / "guardian.json"
    c.write(guardian, {"total_wall_seconds": 1801, "exit_code": 0})
    with pytest.raises(ValueError, match="wall limit"):
        reviewer.review(
            archive, receipt, tmp_path / "handoff.zip", tmp_path / "out", completed, guardian
        )
    assert not (tmp_path / "out").exists()


def test_terminal_policy_stops_on_failure_and_never_accepts_a_base():
    failed = c.terminal_disposition(False)
    assert failed["fixed_subset_direction"] == "stop/rethink-data-training"
    assert failed["terminal_endpoint"] == "E4"
    for result in (failed, c.terminal_disposition(True)):
        assert result["further_fixed_subset_exposure_authorized"] is False
        assert result["base_model_acceptance_run"] is False
        assert result["accepted_base"] is False and result["phase18_authorized"] is False
    with pytest.raises(ValueError):
        c.terminal_disposition("pass")


@pytest.fixture(scope="module")
def historical_saved_panel(tmp_path_factory):
    import shutil

    source = ROOT / "outputs/small-subset-corrected-reviewed/return/original/run"
    panel = tmp_path_factory.mktemp("saved_C2_fixture_not_seed161_results")
    shutil.copytree(source / "evaluation-reference", panel / "evaluation-reference")
    for endpoint in c.ENDPOINTS:
        shutil.copytree(source / "evaluation-C2", panel / f"evaluation-{endpoint}")
    return panel


def test_real_saved_row_comparisons_keep_primary_reference(historical_saved_panel, monkeypatch):
    from latos.evaluation.compare import HISTORICAL_BASE, compare, paired_interval
    from latos.model.network import LatoModel

    def forbidden(*args, **kwargs):
        raise AssertionError("No model forward in saved-row tests")

    monkeypatch.setattr(LatoModel, "forward", forbidden)
    report = evidence.summarize(historical_saved_panel)
    assert set(report["fixed_gate_comparisons"]) == {"D2", "E2", "E4"}
    for name in c.ENDPOINTS:
        assert report["fixed_gate_comparisons"][name] == compare(
            historical_saved_panel / "evaluation-reference",
            historical_saved_panel / f"evaluation-{name}",
            "base",
        )
        assert report["fixed_gate_comparisons"][name]["reference_weights"] == HISTORICAL_BASE
    for left, right in c.CONTRASTS:
        for benchmark in ("ARC-Easy", "ARC-Challenge"):
            assert report["descriptive_contrasts"][left + ":" + right]["paired"][
                benchmark
            ] == paired_interval(
                c.read(historical_saved_panel / f"evaluation-{left}/{benchmark}.json"),
                c.read(historical_saved_panel / f"evaluation-{right}/{benchmark}.json"),
            )
        assert "checks" not in report["descriptive_contrasts"][left + ":" + right]
    # Historical fixture fails gates; this is NOT a future model result.
    assert report["fixed_subset_direction"] == "stop/rethink-data-training"
    assert report["accepted_base"] is False


def test_complete_package_uses_real_comparisons_and_verified_readback(
    tmp_path, historical_saved_panel, monkeypatch
):
    import shutil

    root = tmp_path / "study/run"
    shutil.copytree(historical_saved_panel, root)
    (root / "tools").mkdir()
    c.write(root / "tools/bundle.json", {"scripted_controller_evidence": True})
    c.write(root / "outcome.json", {"execution_complete": True, "accepted_base": False})
    monkeypatch.setattr(evidence, "verify_training", lambda *a, **k: {"scripted_accounting": True})
    monkeypatch.setattr(
        evidence, "verify_execution", lambda *a, **k: {"scripted_source_control": True}
    )
    monkeypatch.setenv("LATOS_SUBSET_PACKAGE", "1")
    evidence.package(root, time.monotonic() + 60)
    receipt = c.read(root / "return.receipt.json")
    assert receipt["verified"] and receipt["sha256"] == c.sha(
        root / "subset-confirmation-return.zip"
    )
    with zipfile.ZipFile(root / "subset-confirmation-return.zip") as z:
        returned = json.loads(z.read("return.json"))
        assert returned["attempt"] == "subset-confirmation-v1"
        assert (
            json.loads(z.read("comparisons.json"))["fixed_subset_direction"]
            == "stop/rethink-data-training"
        )
        assert set(z.namelist()) == set(returned["files"]) | {"return.json"}


def test_invalid_reference_is_still_blocked(historical_saved_panel, monkeypatch):
    from latos.evaluation import compare as comparator

    original = comparator.verified_run

    def altered(path):
        manifest, summary = original(path)
        if Path(path).name == "evaluation-reference":
            manifest["model"]["weights_sha256"] = "f" * 64
        return manifest, summary

    monkeypatch.setattr(comparator, "verified_run", altered)
    with pytest.raises(ValueError, match="frozen Phase 5"):
        evidence.summarize(historical_saved_panel)
