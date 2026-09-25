"""Phase 17.1 CPU engineering checks; no optimizer updates or held-out readers."""

import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest
import torch

DIRECTORY = Path(__file__).resolve().parent
if not (DIRECTORY / "common.py").exists():
    DIRECTORY = DIRECTORY.parent / "experiments/phase-17-1"
sys.path.insert(0, str(DIRECTORY))
import common as c  # noqa: E402 - standalone bundle and repository share the same checks
import runtime as r  # noqa: E402


def module(name):
    spec = importlib.util.spec_from_file_location("phase171_" + name, DIRECTORY / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner, worker = module("run"), module("worker")


@pytest.fixture(autouse=True)
def prohibit_updates(monkeypatch):
    from latos.training.engine import Trainer

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU checks must not optimize")

    monkeypatch.setattr(Trainer, "update", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def plan():
    return c.read(DIRECTORY / "plan.json")


def tiny():
    from latos.model import create_model
    from latos.training.engine import Trainer

    model, config, data, seed, _ = r.fixture(plan(), "ordinary")
    trainer = Trainer(create_model(model, seed), config, *data, single_pass=True)
    return trainer, data, config, r.fingerprints()


def bound(tmp_path):
    trainer, data, config, m = tiny()
    binding = {"attempt_id": c.ATTEMPT, "test_only": True, "numerical_policy": "CPU no updates"}
    path = tmp_path / "step-00000000"
    r.save_bound(trainer, path, binding, m)
    return path, binding, data, config, m


def row(step=4):
    return {
        "state": {"model": "exact", "optimizer": "exact", "sampler": step, "global_rng": "exact"},
        "input": {"sha256": "exact"},
        "metric": {
            "step": step,
            "targets": 3,
            "tokens_seen": step * 3,
            "windows_seen": step,
            "epoch": 0,
            "microbatches": 2,
            "learning_rate": 0.001,
            "loss": 0.0,
        },
    }


def test_approved_plan_and_authorization_are_bound():
    require = c.require
    require(c.hash_file(DIRECTORY / "plan.json") == c.PLAN, "Plan changed")
    require(c.hash_file(DIRECTORY / "PROPOSAL.md") == c.PROPOSAL, "Proposal changed")
    auth = c.read(DIRECTORY / "authorization.json")
    assert auth["owner_authorized"] and auth["plan_sha256"] == c.PLAN
    assert not plan()["training_authorized"]  # Historical proposal snapshot is unchanged.


def test_prespecified_fixture_bytes_and_tail_without_updates():
    for name, windows, final in (("ordinary", 32, 2), ("partial-tail", 30, 1)):
        model, config, data, seed, sha = r.fixture(plan(), name)
        assert sha == c.read(DIRECTORY / "fixture-identities.json")[name]
        assert len(data[0].windows) == windows and model.n_layers == 1
        assert (windows // 2) % 2 == (0 if final == 2 else 1)
        assert config.max_steps == 8 and seed in (17110, 17111)
        assert sum(data[0].target_count(i) for i in range(windows)) > 0


def test_exact_52_update_dispatch_budget():
    jobs = runner.fixture_jobs()
    assert len(jobs) == 8 and sum(j["reserved_updates"] for j in jobs) == 52
    assert len({j["id"] for j in jobs}) == 8


def test_checkpoint_roundtrip_and_restore_at_zero_updates(tmp_path):
    path, binding, data, config, m = bound(tmp_path)
    trainer = r.load_bound(path, data, config, binding, m, device="cpu", precision="float32")
    assert trainer.step == 0 and not trainer.optimizer.state
    assert m.snapshot(trainer) == c.read(path / "policy.json")["state"]


def test_checkpoint_wrong_policy_is_rejected(tmp_path):
    path, binding, data, config, m = bound(tmp_path)
    with pytest.raises(ValueError, match="policy"):
        r.load_bound(
            path,
            data,
            config,
            {**binding, "attempt_id": "old-run"},
            m,
            device="cpu",
            precision="float32",
        )


def test_checkpoint_missing_sidecar_is_rejected(tmp_path):
    path, binding, _, _, _ = bound(tmp_path)
    (path / "policy.json").unlink()
    with pytest.raises(FileNotFoundError):
        r.verify_policy(path, binding)


def test_checkpoint_tensor_corruption_is_rejected(tmp_path):
    path, binding, _, _, _ = bound(tmp_path)
    with (path / "training.safetensors").open("r+b") as f:
        f.write(b"bad")
    with pytest.raises(ValueError, match="changed"):
        r.verify_policy(path, binding)


def test_policy_write_failure_does_not_publish_checkpoint(tmp_path, monkeypatch):
    trainer, _, _, m = tiny()

    def fail(*args):
        raise OSError("simulated policy write interruption")

    monkeypatch.setattr(r, "write_once", fail)
    with pytest.raises(OSError):
        r.save_bound(trainer, tmp_path / "checkpoint", {"test": True}, m)
    assert not (tmp_path / "checkpoint").exists()
    assert list(tmp_path.glob("checkpoint.*.pending/payload/state.json"))


def test_partial_update_checkpoint_refused(tmp_path):
    trainer, _, _, m = tiny()
    trainer.ready = False
    with pytest.raises(ValueError, match="complete"):
        r.save_bound(trainer, tmp_path / "checkpoint", {}, m)
    assert not (tmp_path / "checkpoint").exists()


def test_old_pending_directory_preserved(tmp_path):
    old = tmp_path / "checkpoint.old.pending"
    old.mkdir()
    (old / "evidence").write_bytes(b"preserve")
    trainer, _, _, m = tiny()
    r.save_bound(trainer, tmp_path / "checkpoint", {}, m)
    assert (old / "evidence").read_bytes() == b"preserve"


def test_checkpoint_never_overwrites_existing(tmp_path):
    path, binding, _, _, m = bound(tmp_path)
    before = c.hash_file(path / "policy.json")
    trainer, _, _, _ = tiny()
    with pytest.raises(ValueError, match="exists"):
        r.save_bound(trainer, path, binding, m)
    assert c.hash_file(path / "policy.json") == before


def test_replay_loss_tolerance_and_state_are_independent():
    a = row()
    b = copy.deepcopy(a)
    b["metric"]["loss"] = 1e-5
    c.equal_replay(a, b)
    b["metric"]["loss"] = 0
    b["state"]["optimizer"] = "drift"
    with pytest.raises(ValueError, match="full state"):
        c.equal_replay(a, b)
    b = copy.deepcopy(a)
    b["metric"]["loss"] = 0.00001 + 1e-10
    with pytest.raises(ValueError, match="loss"):
        c.equal_replay(a, b)


def test_replay_schedule_and_input_mismatch_rejected():
    a = row()
    b = copy.deepcopy(a)
    b["metric"]["learning_rate"] += 1e-15
    with pytest.raises(ValueError, match="schedule"):
        c.equal_replay(a, b)
    b = copy.deepcopy(a)
    b["input"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="input"):
        c.equal_replay(a, b)


def test_schedule_table_is_write_once_and_exact(tmp_path):
    _, config, _, _, _ = r.fixture(plan(), "ordinary")
    p = tmp_path / "schedule.json"
    worker.schedule(config, p, True)
    worker.schedule(config, p, False)
    with pytest.raises(FileExistsError):
        worker.schedule(config, p, True)
    v = c.read(p)
    v["rates"][0] += 1e-15
    p.write_text(json.dumps(v))
    with pytest.raises(ValueError, match="schedule"):
        worker.schedule(config, p, False)


def test_journal_append_does_not_replace_existing_file(tmp_path):
    p = tmp_path / "events.jsonl"
    j = c.Journal(p)
    j.add({"n": 1})
    first = p.read_bytes()
    j.add({"n": 2})
    j.close()
    assert p.read_bytes().startswith(first) and len(c.trace(p)) == 2
    with pytest.raises(FileExistsError):
        c.Journal(p)


def test_torn_tail_is_never_a_successful_record(tmp_path):
    p = tmp_path / "trace.jsonl"
    p.write_bytes(b'{"step":1}\n{"step":')
    with pytest.raises(ValueError, match="Torn"):
        c.trace(p)
    assert c.trace(p, allow_torn=True) == [{"step": 1}]
    p.write_bytes(b'{bad}\n{"step":2}\n')
    with pytest.raises(ValueError):
        c.trace(p, allow_torn=True)


def test_second_live_process_lock_is_rejected(tmp_path):
    p = tmp_path / "lease"
    with c.lock(p), pytest.raises(OSError), c.lock(p):
        pass


def test_watchdog_kills_child_at_deadline(tmp_path):
    j = c.Journal(tmp_path / "events")
    with (tmp_path / "log").open("wb") as log, pytest.raises(TimeoutError):
        c.bounded_child(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            os.environ.copy(),
            log,
            j,
            time.monotonic(),
            0.05,
            lambda: 0,
            100,
        )
    j.close()
    assert c.trace(tmp_path / "events")


def test_watchdog_rejects_failed_child(tmp_path):
    j = c.Journal(tmp_path / "events")
    with (tmp_path / "log").open("wb") as log, pytest.raises(ValueError, match="Worker exited"):
        c.bounded_child(
            [sys.executable, "-c", "raise SystemExit(7)"],
            os.environ.copy(),
            log,
            j,
            time.monotonic(),
            30,
            lambda: 0,
            100,
        )
    j.close()


def test_exhausted_artifacts_refuse_child_dispatch(tmp_path):
    j = c.Journal(tmp_path / "events")
    with (tmp_path / "log").open("wb") as log, pytest.raises(ValueError, match="Budget"):
        c.bounded_child(["must-not-run"], {}, log, j, time.monotonic(), 30, lambda: 100, 100)
    j.close()


def test_artifact_scan_handles_only_disappeared_files(tmp_path, monkeypatch):
    a = tmp_path / "a"
    a.write_bytes(b"123")
    a.rename(tmp_path / "b")
    assert c.file_size(a) == 0 and c.tree_bytes(tmp_path) == 3

    def fail(*args):
        raise PermissionError("fixture")

    monkeypatch.setattr(Path, "stat", fail)
    with pytest.raises(PermissionError):
        c.file_size(tmp_path / "b")


def test_integrity_and_unsafe_paths_rejected(tmp_path):
    (tmp_path / "ok").write_bytes(b"state")
    expected = {"ok": c.record(tmp_path / "ok")}
    c.verify_records(tmp_path, expected)
    (tmp_path / "ok").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Integrity"):
        c.verify_records(tmp_path, expected)
    for name in ("../escape", "C:/escape", ".private/secret", "a\\b"):
        with pytest.raises(ValueError):
            c.safe_path(tmp_path, name)


def test_final_and_resume_require_review_before_reading_payloads(tmp_path):
    with pytest.raises(ValueError, match="Final requires"):
        c.check_final_review(tmp_path, None, "a")
    with pytest.raises(ValueError, match="Resume requires"):
        runner.review_resume(tmp_path, None, "a")


def test_cpu_preflight_rejects_skipped_and_wrong_count(tmp_path):
    p = tmp_path / "junit.xml"
    p.write_text('<testsuite><testcase name="a"/></testsuite>')
    assert runner.checked_cpu_result(p, 1)["passed"] == 1
    with pytest.raises(ValueError):
        runner.checked_cpu_result(p, 2)
    p.write_text('<testsuite><testcase name="a"><skipped/></testcase></testsuite>')
    with pytest.raises(ValueError):
        runner.checked_cpu_result(p, 1)


def test_incomplete_update_is_charged_conservatively(tmp_path):
    (tmp_path / "requests").mkdir()
    job = tmp_path / "jobs/train-0"
    job.mkdir(parents=True)
    c.write_once(
        tmp_path / "requests/train-0.json",
        {"id": "train-0", "kind": "train", "reserved_updates": 7485},
    )
    (job / "updates.jsonl").write_bytes(c.canonical(row()))
    assert runner.update_accounting(tmp_path)["serious_updates_upper"] == 2


def scripted_session(tmp_path, monkeypatch, fail_cpu=False):
    """Simulate child receipts, never call any worker or optimizer."""
    from types import SimpleNamespace

    transfer = tmp_path / "transfer"
    bundle = transfer / "phase17-1-tools"
    bundle.mkdir(parents=True)
    c.write_once(bundle / "bundle.json", {"scripted_test_only": True})
    # Simulate availability without creating fake environment trees in return evidence.
    actual_is_file = Path.is_file
    interpreters = {
        transfer / "source" / env / "Scripts/python.exe" for env in (".venv", ".wheel-env")
    }
    monkeypatch.setattr(Path, "is_file", lambda p: p in interpreters or actual_is_file(p))
    output = transfer / "phase17-1-results"
    for name in ("sessions", "requests", "jobs"):
        (output / name).mkdir(parents=True)
    dispatched = []

    def fake_child(command, environment, log, journal, started, seconds, measure, cap, **kwargs):
        assert seconds > 0 and measure() < cap
        if "pytest" in command:
            dispatched.append("cpu")
            if fail_cpu:
                raise ValueError("scripted CPU failure")
            p = Path(command[command.index("--junitxml") + 1])
            p.write_text('<testsuite><testcase name="scripted"/></testsuite>')
            return
        req = c.read(command[-1])
        dispatched.append(req["kind"])
        jobdir = output / "jobs" / req["id"]
        jobdir.mkdir()
        if req["kind"].startswith("fixture-"):
            start = 1 if req["kind"] == "fixture-reference" else 4
            rows = [row(i) for i in range(start, 9)]
            if req["fixture"] == "partial-tail":
                rows[-1]["metric"]["microbatches"] = 1
            (jobdir / "updates.jsonl").write_bytes(b"".join(c.canonical(v) for v in rows))
        c.write_once(
            jobdir / "result.json",
            {
                "status": "complete",
                "optimizer_updates": req["reserved_updates"],
                "scripted_test_only": True,
            },
        )

    monkeypatch.setattr(runner, "bounded_child", fake_child)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    args = SimpleNamespace(stage="start", review=None)
    return args, bundle, transfer, output, dispatched


def test_supervisor_exact_scripted_sequence_without_execution(tmp_path, monkeypatch):
    args, bundle, transfer, output, calls = scripted_session(tmp_path, monkeypatch)
    runner.execute_session(args, bundle, transfer, output, plan(), {"cpu_test_count": 1})
    assert calls == [
        "integrity",
        "cpu",
        "cpu",
        *(["fixture-reference", "fixture-resumed"] * 4),
        "train",
        "development",
    ]
    receipt = c.read(output / "sessions/00/receipt.json")
    assert receipt["status"] == "complete"
    assert receipt["preflight_updates_upper"] == 52 and receipt["serious_updates_upper"] == 7485
    assert c.read(output / "preflight.json")["passed"]


def test_failed_preflight_stops_before_any_optimizer_job(tmp_path, monkeypatch):
    args, bundle, transfer, output, calls = scripted_session(tmp_path, monkeypatch, fail_cpu=True)
    with pytest.raises(ValueError, match="scripted CPU"):
        runner.execute_session(args, bundle, transfer, output, plan(), {"cpu_test_count": 1})
    assert calls == ["integrity", "cpu"] and not (output / "preflight.json").exists()
    assert c.read(output / "sessions/00/receipt.json")["status"] == "failed"


def test_final_declined_when_development_gate_failed(tmp_path):
    c.write_once(tmp_path / "training-result.json", {"candidate_weights": "candidate"})
    c.write_once(tmp_path / "development-comparison.json", {"passed": False})
    p = tmp_path / "review.json"
    c.write_once(
        p,
        {
            "kind": "phase17.1-final-Mac-review",
            "attempt_id": c.ATTEMPT,
            "bundle_sha256": "bundle",
            "candidate_weights": "candidate",
            "accepted_for_final": True,
        },
    )
    with pytest.raises(ValueError, match="Development"):
        c.check_final_review(tmp_path, p, "bundle")


def test_changed_sidecar_state_fails_actual_zero_update_restore(tmp_path):
    path, binding, data, config, m = bound(tmp_path)
    policy = c.read(path / "policy.json")
    policy["state"]["global_rng"] = "wrong"
    (path / "policy.json").write_bytes(c.canonical(policy))
    with pytest.raises(ValueError, match="full state"):
        r.load_bound(path, data, config, binding, m, device="cpu", precision="float32")


def test_bundle_rejects_removed_authorization_even_with_updated_hash(tmp_path):
    for name in ("plan.json", "PROPOSAL.md", "authorization.json"):
        (tmp_path / name).write_bytes((DIRECTORY / name).read_bytes())
    auth = c.read(tmp_path / "authorization.json")
    auth["owner_authorized"] = False
    (tmp_path / "authorization.json").write_bytes(c.canonical(auth))
    c.write_once(
        tmp_path / "bundle.json",
        {
            "attempt_id": c.ATTEMPT,
            "files": {
                name: c.record(tmp_path / name)
                for name in ("plan.json", "PROPOSAL.md", "authorization.json")
            },
        },
    )
    with pytest.raises(ValueError, match="authorization"):
        c.verify_bundle(tmp_path)


def test_internal_worker_refuses_no_ticket_without_torch_job(tmp_path, monkeypatch):
    p = tmp_path / "request.json"
    c.write_once(p, {"ticket_sha256": "not-valid"})
    monkeypatch.delenv("LATOS_PHASE171_TICKET", raising=False)
    monkeypatch.setattr(sys, "argv", ["worker.py", "--request", str(p)])
    with pytest.raises(ValueError, match="live supervisor ticket"):
        worker.main()


def test_preservation_hash_uses_the_existing_lock_handle(tmp_path):
    p = tmp_path / "lease"
    p.write_bytes(b"0")
    expected = {"lease": c.record(p)}
    with c.lock(p) as stream:
        c.verify_records(tmp_path, expected, {"lease": stream})
        assert c.record(p, stream) == expected["lease"]


def test_return_package_retains_final_and_inventories_omitted_tensors(tmp_path, monkeypatch):
    import zipfile

    p = module("package_return")
    transfer = tmp_path / "transfer"
    bundle = transfer / "tools"
    bundle.mkdir(parents=True)
    c.write_once(bundle / "bundle.json", {"scripted_test_only": True})
    monkeypatch.setattr(p, "__file__", str(bundle / "package_return.py"))
    monkeypatch.setattr(p, "verify_bundle", lambda _: None)
    root = transfer / "phase17-1-results"
    model = root / "checkpoints/step-00007485/model/model.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"synthetic packaging fixture, not weights")
    optimizer = model.parent.parent / "training.safetensors"
    optimizer.write_bytes(b"retain original")
    destination = tmp_path / "return.zip"
    receipt = p.package(transfer, destination)
    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(archive.read("return.json"))
        assert "run/checkpoints/step-00007485/model/model.safetensors" in manifest["files"]
        assert (
            "run/checkpoints/step-00007485/training.safetensors"
            in manifest["omitted_retained_on_Windows"]
        )
    assert (
        optimizer.read_bytes() == b"retain original"
        and c.hash_file(destination) == receipt["sha256"]
    )
    with c.lock(root / "supervisor.lock"), pytest.raises(OSError):
        p.package(transfer, tmp_path / "blocked.zip")
    assert not (tmp_path / "blocked.zip").exists()


def test_internal_worker_rejects_changed_dispatch_request(tmp_path, monkeypatch):
    import hashlib

    p = tmp_path / "request.json"
    c.write_once(p, {"ticket_sha256": hashlib.sha256(b"ticket").hexdigest()})
    expected = c.hash_file(p)
    p.write_bytes(b'{"changed": true}')
    monkeypatch.setenv("LATOS_PHASE171_TICKET", "ticket")
    monkeypatch.setenv("LATOS_PHASE171_REQUEST_SHA256", expected)
    monkeypatch.setattr(sys, "argv", ["worker.py", "--request", str(p)])
    with pytest.raises(ValueError, match="unchanged request"):
        worker.main()


def test_second_interruption_preserves_charges_without_charging_unexecuted_tail(
    tmp_path, monkeypatch
):
    args, bundle, transfer, output, _ = scripted_session(tmp_path, monkeypatch)
    fake_child = runner.bounded_child

    def interrupt_train(command, *values, **kwargs):
        if "--request" in command:
            req = c.read(command[-1])
            if req["kind"] in ("train", "resume"):
                d = output / "jobs" / req["id"]
                d.mkdir()
                count = 2 if req["kind"] == "train" else 1
                (d / "updates.jsonl").write_bytes(
                    b"".join(c.canonical(row(i)) for i in range(1, count + 1))
                )
                raise KeyboardInterrupt("scripted interruption; no real updates")
        return fake_child(command, *values, **kwargs)

    monkeypatch.setattr(runner, "bounded_child", interrupt_train)
    with pytest.raises(KeyboardInterrupt):
        runner.execute_session(args, bundle, transfer, output, plan(), {"cpu_test_count": 1})
    prior = c.read(output / "sessions/00/receipt.json")
    assert prior["serious_updates_upper"] == 3
    cp = output / "checkpoints/step-00000000"
    cp.mkdir(parents=True)
    c.write_once(cp / "policy.json", {"scripted_test_only": True})
    c.write_once(cp / "state.json", {"step": 0})
    review = tmp_path / "Mac-review.json"
    c.write_once(
        review,
        {
            "kind": "phase17.1-external-interruption-Mac-review",
            "attempt_id": c.ATTEMPT,
            "bundle_sha256": c.hash_file(bundle / "bundle.json"),
            "external_interruption_verified": True,
            "training_seconds_charged": prior["training_seconds"],
            "evaluation_seconds_charged": 0,
            "serious_updates_upper": 3,
            "checkpoint": "checkpoints/step-00000000",
            "checkpoint_policy_sha256": c.hash_file(cp / "policy.json"),
            "prior_files": {
                p.relative_to(output).as_posix(): c.record(p)
                for p in output.rglob("*")
                if p.is_file()
            },
        },
    )
    args.stage = "resume"
    args.review = review
    with pytest.raises(KeyboardInterrupt):
        runner.execute_session(args, bundle, transfer, output, plan(), {"cpu_test_count": 1})
    assert c.read(output / "sessions/01/receipt.json")["serious_updates_upper"] == 5
