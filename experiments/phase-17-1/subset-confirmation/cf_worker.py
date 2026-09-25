"""Ticketed Windows jobs for the single small-subset study; no public training CLI."""

import argparse
import gc
import hashlib
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from cf_common import (
    ARMS,
    BASE,
    ENDPOINTS,
    PARAMETERS,
    SEED,
    budget,
    canonical,
    read,
    record,
    require,
    selfcheck,
    sha,
    verify,
    write,
)


def datasets(bundle):
    import numpy as np

    from latos.training.data import TokenDataset

    result = []
    for split in ("train", "development"):
        meta = read(bundle / "cache" / f"{split}.json")
        ids = np.fromfile(bundle / "cache" / f"{split}.u32", dtype="<u4")
        offsets = np.fromfile(bundle / "cache" / f"{split}.u64", dtype="<u8")
        expected = meta["identity"]
        require(
            len(offsets) == expected["windows"] + 1 and offsets[0] == 0 and offsets[-1] == len(ids),
            "Cache offsets",
        )
        windows = tuple(
            tuple(int(x) for x in ids[int(a) : int(b)])
            for a, b in zip(offsets[:-1], offsets[1:], strict=True)
        )
        data = TokenDataset(
            windows,
            512,
            16384,
            expected["tokenizer_sha256"],
            expected["split"],
            expected["source_sha256"],
        )
        require(data.identity == expected, "Fixed Phase 16 subset differs")
        result.append(data)
    return tuple(result)


def evaluate(request, bundle, transfer, output, guard):
    from latos.evaluation.runner import run

    name = request["endpoint"]
    if name == "reference":
        model, codec, weights = (
            transfer / "evaluation/base",
            transfer / "evaluation/tokenizer",
            BASE,
        )
    else:
        arm, step = ENDPOINTS[name]
        checkpoint = output / "jobs" / arm / f"step-{step:08d}"
        saved = read(checkpoint / "diagnostic.json")
        require(
            saved["kind"] == "subset-confirmation-no-resume-v1" and saved["readback_exact"],
            "Incomplete checkpoint",
        )
        verify(checkpoint, saved["files"])
        require(
            saved["binding"]["bundle_sha256"] == sha(bundle / "bundle.json"), "Checkpoint binding"
        )
        model, codec = checkpoint / "model", transfer / "inputs/tokenizer"
        weights = sha(model / "model.safetensors")
    source = transfer / "source"
    guard()
    run(
        SimpleNamespace(
            protocol=source / "configs/evaluation/protocol-v1.json",
            suite=source / "configs/evaluation/suite-v1.json",
            external_manifest=source / "configs/evaluation/external-v1.json",
            external_dir=transfer / "evaluation/external",
            data_manifest=source / "data/manifests/english-books-v1.json",
            corpus_dir=transfer / "evaluation/corpus",
            model_dir=model,
            tokenizer_dir=codec,
            weights_sha256=weights,
            output=output / f"evaluation-{name}",
            device="cuda",
            threads=4,
            batch_size=8,
            acceptance_plan=None,
        )
    )
    return {
        "endpoint": name,
        "optimizer_updates": 0,
        "resources": guard(),
        "final_access": False,
        "accepted_base": False,
    }


def execute(req, bundle, transfer, output, job):
    manifest = read(bundle / "bundle.json")
    verify(bundle, manifest["files"])
    sys.path.insert(0, str(bundle / "legacy"))
    import runtime as legacy

    legacy_plan = read(bundle / "legacy-plan.json")
    controls = legacy.setup(legacy_plan, job)
    import torch

    import latos
    from latos.model import ModelConfig, create_model
    from latos.training.config import TrainingConfig
    from latos.training.engine import Trainer
    from latos.training.probe import memory, snapshot_source

    expected_root = transfer / (
        "source/src/latos"
        if req["runtime"] == "checkout"
        else "source/.wheel-env/Lib/site-packages/latos"
    )
    require(Path(latos.__file__).resolve().is_relative_to(expected_root.resolve()), "Import origin")
    snapshot_source(job)
    m = legacy.fingerprints()

    def guard():
        budget(req["deadline"])
        legacy.check_controls(legacy_plan)
        value = memory()
        require(
            value["host_peak_working_set_bytes"] is not None
            and value["host_peak_working_set_bytes"] <= 8 * 1024**3
            and value["peak_reserved_bytes"] <= controls["gpu_total_bytes"] * 0.85,
            "Resource bound",
        )
        return value

    write(job / "controls.json", controls)
    guard()
    if req["kind"] == "prepare":
        from launch import guard_selftest

        write(job / "guardian-test.json", guard_selftest(job))
        verify(transfer, read(bundle / "required-inputs.json"))
        core = {
            p.relative_to(transfer / "source/src/latos").as_posix(): record(p)
            for p in (transfer / "source/src/latos").rglob("*.py")
        }
        wheel = transfer / "source/.wheel-env/Lib/site-packages/latos"
        require(
            {p.relative_to(wheel).as_posix(): record(p) for p in wheel.rglob("*.py")} == core,
            "Installed wheel differs",
        )
        data = datasets(bundle)
        write(output / "subset-identities.json", [d.identity for d in data])
        return {"optimizer_updates": 0, "checks": selfcheck(), "resources": guard()}
    if req["kind"] == "evaluate":
        return evaluate(req, bundle, transfer, output, guard)

    fixture = req["kind"].startswith("fixture")
    if fixture:
        cfg, training, data, seed, identity = legacy.fixture(legacy_plan, req["fixture"])
        require(
            identity == read(bundle / "fixture-identities.json")[req["fixture"]], "Fixture changed"
        )
        reference = output / "jobs" / req["reference"]
        binding = {
            "fixture": req["fixture"],
            "runtime": req["runtime"],
            "controls": controls,
            "bundle_sha256": sha(bundle / "bundle.json"),
        }
        legacy.seed_all(seed)
        if req["kind"] == "fixture-reference":
            trainer = Trainer(
                create_model(cfg, seed),
                training,
                *data,
                "cuda",
                precision="bfloat16",
                single_pass=True,
            )
        else:
            trainer = legacy.load_bound(reference / "step-00000003", data, training, binding, m)
        limits, checkpoints = (8 if req["kind"] == "fixture-reference" else 5), [3, 8]
    else:
        from cf_adapter import advance_pass, check_progress, save_checkpoint, update

        arm = req["arm"]
        layers, passes, limits = ARMS[arm]
        config = read(transfer / "source/configs/training2/selected-model.json")
        config["n_layers"] = layers
        cfg = ModelConfig(**config)
        require(cfg.parameter_count == PARAMETERS, "Model count")
        training = TrainingConfig(
            sequence_length=512,
            batch_size=2,
            accumulation_steps=8,
            max_steps=limits,
            warmup_steps=19,
            learning_rate=0.0003,
            min_lr_ratio=0.1,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            max_grad_norm=1.0,
            seed=SEED,
        )
        data = datasets(bundle)
        legacy.seed_all(SEED)
        trainer = Trainer(create_model(cfg, SEED), training, *data, "cuda", precision="bfloat16")
        require(not trainer.optimizer.state and trainer.step == 0, "Not fresh")
        binding = {
            "arm": arm,
            "controls": controls,
            "bundle_sha256": sha(bundle / "bundle.json"),
            "model_config": cfg.to_dict(),
            "training_config": training.to_dict(),
            "datasets": trainer.data_identities,
            "adapter": "flushed-pass-tails-v1",
            "resumption_authorized": False,
        }
        checkpoints = list(range(382, limits + 1, 382))
    initial = m.snapshot(trainer)
    write(job / "initial.json", {"state": initial, "binding": binding})
    rates = [training.learning_rate_at(i) for i in range(1, training.max_steps + 1)]
    write(job / "schedule.json", rates)
    if not fixture:
        if req["arm"] == "E":
            a = read(output / "jobs/D/initial.json")["state"]
            require(
                initial["model"] == a["model"] and initial["sampler"] == a["sampler"],
                "D/E initialization or data ordering differs",
            )
        before = m.snapshot(trainer)
        write(job / "monitor-0.json", trainer.validate())
        require(before == m.snapshot(trainer), "Monitoring mutated state")
        save_checkpoint(trainer, job / "step-00000000", binding, m)
    executed = 0
    with (job / "updates.jsonl").open("xb") as log:
        while trainer.step < training.max_steps:
            guard()
            if not fixture:
                advance_pass(trainer)
            signature = m.input_signature(trainer)
            metric = trainer.update() if fixture else update(trainer)
            executed += 1
            require(executed <= limits, "Optimizer budget")
            require(
                metric["targets"] == signature["targets"]
                and metric["learning_rate"] == rates[trainer.step - 1],
                "Update conservation",
            )
            require(
                all(
                    torch.isfinite(t).all().item()
                    for slots in trainer.optimizer.state.values()
                    for t in slots.values()
                    if isinstance(t, torch.Tensor)
                ),
                "Nonfinite Adam state",
            )
            if not fixture:
                check_progress(trainer)
            state = m.snapshot(trainer)
            require(state["global_rng"] == initial["global_rng"], "Unexpected RNG use")
            row = {"metric": metric, "input": signature, "state": state, "resources": guard()}
            log.write(canonical(row))
            log.flush()
            os.fsync(log.fileno())
            if trainer.step in checkpoints:
                if fixture:
                    legacy.save_bound(trainer, job / f"step-{trainer.step:08d}", binding, m)
                else:
                    before = m.snapshot(trainer)
                    write(job / f"monitor-{trainer.step}.json", trainer.validate())
                    require(before == m.snapshot(trainer), "Monitoring mutated state")
                    save_checkpoint(trainer, job / f"step-{trainer.step:08d}", binding, m)
    require(executed == limits, "Incomplete update budget")
    result = {
        "optimizer_updates": executed,
        "step": trainer.step,
        "targets": trainer.tokens_seen,
        "windows": trainer.windows_seen,
        "resources": guard(),
        "accepted_base": False,
    }
    del trainer
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    require(os.name == "nt", "Windows only; no Mac training")
    token = os.environ.pop("LATOS_SUBSET_TICKET", "")
    req = read(args.request)
    require(
        token and hashlib.sha256(token.encode()).hexdigest() == req["ticket"], "Supervisor ticket"
    )
    require(sha(args.request) == os.environ.pop("LATOS_SUBSET_REQUEST", ""), "Request changed")
    budget(req["deadline"])
    bundle, transfer, output = (Path(req[k]).resolve() for k in ("bundle", "transfer", "output"))
    job = output / "jobs" / req["id"]
    job.mkdir(exist_ok=False)
    try:
        result = execute(req, bundle, transfer, output, job)
        write(job / "result.json", {**result, "status": "complete"})
    except Exception as exc:
        write(
            job / "failure.json",
            {"type": type(exc).__name__, "error": str(exc), "monotonic": time.monotonic()},
        )
        raise


if __name__ == "__main__":
    main()
