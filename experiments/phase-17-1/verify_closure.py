"""Independent read-only reconstruction of the completed, quality-failing attempt.

No returned Python is imported, no model forward/optimizer is called, and no final
acceptance inputs are read. Saved development outputs are aggregated again only.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import torch
from safetensors import safe_open

from latos.data.manifest import canonical_json
from latos.evaluation.compare import compare
from latos.training.full_data import FullDataset, expected_accounting

SOURCE = "7cb5afd54d33fda4d18b84c5f0645d661b76e002"
RETURN = "66c2ddee8e85a289998461deb0e86945022b9e368a142f86ca9a68ba3a96e358"
TOKENIZER = "23b9182d5943e7f8b32c6f415f108b7d06f229c80f5cd8f236258db40a6b94c4"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            require(line.endswith("\n"), "Torn production trace")
            yield json.loads(line)


def state_check(state):
    core = {k: v for k, v in state.items() if k != "state_sha256"}
    require(
        hashlib.sha256(canonical_json(core)).hexdigest() == state["state_sha256"],
        "State digest differs",
    )


def check_metric(metric, step, count, targets, cumulative, rate):
    expected = {
        "step": step,
        "windows_seen": count,
        "targets": targets,
        "tokens_seen": cumulative,
        "epoch": 0,
        "microbatches": math.ceil((count - (step - 1) * 16) / 2),
        "learning_rate": rate,
    }
    require(all(metric[k] == v for k, v in expected.items()), "Training exposure/rate differs")
    require(
        all(
            math.isfinite(metric[k])
            for k in ("loss", "grad_norm_before_clip", "seconds", "targets_per_second")
        ),
        "Nonfinite completed metric",
    )


def disposition(execution_ok, development_passed):
    require(execution_ok, "Execution verification is not complete")
    return (
        "development-quality-failed; final prohibited; Phase 18 blocked"
        if not development_passed
        else "requires final-selection review; not accepted"
    )


def verify(archive, receipt_path, root, cache):
    require(sha(archive) == RETURN, "Return archive identity mismatch")
    receipt, manifest = read(receipt_path), read(root / "return.json")
    require(
        receipt["sha256"] == RETURN and receipt["bytes"] == archive.stat().st_size,
        "Receipt differs",
    )
    included, omitted = manifest["files"], manifest["omitted_retained_on_Windows"]
    require(
        len(included) == receipt["included"] == 1366
        and len(omitted) == receipt["omitted_retained"] == 81,
        "Unexpected file counts",
    )
    require(not set(included) & set(omitted), "Inventory overlap")
    all_files = included | omitted
    with zipfile.ZipFile(archive) as z:
        require(
            len(z.namelist()) == len(set(z.namelist()))
            and set(z.namelist()) == set(included) | {"return.json"},
            "Archive member set differs",
        )
        require(
            z.read("return.json") == (root / "return.json").read_bytes(), "Return manifest changed"
        )
        for name, record in included.items():
            require(
                not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and "\\" not in name
                and ":" not in name,
                "Unsafe return member",
            )
            path = root / name
            require(
                not path.is_symlink()
                and path.stat().st_size == record["bytes"]
                and sha(path) == record["sha256"],
                "Extracted evidence identity differs",
            )
            with z.open(name) as stream:
                require(
                    hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"],
                    "ZIP payload differs",
                )
    bundle = read(root / "bundle/bundle.json")
    require(
        bundle["source_commit"] == SOURCE
        and manifest["bundle_sha256"] == sha(root / "bundle/bundle.json"),
        "Controller source binding differs",
    )
    with zipfile.ZipFile("outputs/phase17-1-windows-handoff.zip") as z:
        require(
            z.read("bundle.json") == (root / "bundle/bundle.json").read_bytes(),
            "Original handoff differs",
        )
        for name, expected_record in bundle["files"].items():
            path = root / "bundle" / name
            require(
                path.stat().st_size == expected_record["bytes"]
                and sha(path) == expected_record["sha256"]
                and path.read_bytes() == z.read(name),
                "Handoff payload binding differs",
            )
        for name, source in bundle["source_paths"].items():
            original = subprocess.check_output(["git", "show", f"{SOURCE}:{source}"])
            require(
                original == z.read(name) == (root / "bundle" / name).read_bytes(),
                "Controller payload differs from Git",
            )
    expected = {}
    for path in (
        subprocess.check_output(["git", "ls-tree", "-r", "--name-only", SOURCE, "src/latos"])
        .decode()
        .splitlines()
    ):
        if path.endswith(".py"):
            expected[path.removeprefix("src/latos/")] = hashlib.sha256(
                subprocess.check_output(["git", "show", f"{SOURCE}:{path}"])
            ).hexdigest()
    run = root / "run"
    sources = [*(run / "jobs").glob("*/source/latos"), *(run.glob("development-*/source/latos"))]
    require(len(sources) == 12, "Expected ten worker and two evaluator snapshots")
    for directory in sources:
        require(
            {p.relative_to(directory).as_posix(): sha(p) for p in directory.rglob("*.py")}
            == expected,
            "Executed native source differs",
        )
    for inv in (run / "jobs").glob("*/inventory.json"):
        prefix = inv.parent.relative_to(root).as_posix() + "/"
        require(
            read(inv)
            == {
                name.removeprefix(prefix): v
                for name, v in all_files.items()
                if name.startswith(prefix) and name != prefix + "inventory.json"
            },
            "Worker inventory differs",
        )
    plan = read("experiments/phase-17-1/plan.json")
    require(read(root / "bundle/plan.json") == plan, "Frozen plan differs")
    cpu = {}
    for mode in ("checkout", "wheel"):
        cases = list(ET.parse(run / f"cpu-{mode}.xml").getroot().iter("testcase"))
        require(
            len(cases) == 35
            and not any(
                c.find(k) is not None for c in cases for k in ("error", "failure", "skipped")
            ),
            "CPU preflight failed/skipped",
        )
        cpu[mode] = {"passed": 35, "failed": 0, "skipped": 0, "optimizer_updates": 0}
    requests = {p.stem: read(p) for p in (run / "requests").glob("*.json")}
    expected_jobs = ["integrity", "cpu-checkout", "cpu-wheel"]
    for mode in ("checkout", "wheel"):
        for fixture in ("ordinary", "partial-tail"):
            expected_jobs.extend([f"{mode}-{fixture}-reference", f"{mode}-{fixture}-resumed"])
    expected_jobs.extend(["train-0", "development"])
    require(set(requests) == set(expected_jobs), "Unexpected/retry/final job")
    require(len(list((run / "sessions").iterdir())) == 1, "Unexpected session/resumption")
    events = list(rows(run / "sessions/00/events.jsonl"))
    for event in ("job-start", "job-complete"):
        require(
            [e["id"] for e in events if e["event"] == event] == expected_jobs, "Job order differs"
        )
    session = read(run / "sessions/00/receipt.json")
    require(
        events[-1] == {"event": "session-end", "receipt": session}
        and session["status"] == "complete",
        "Supervisor did not complete",
    )
    require(
        not any(e["event"] == "failure" for e in events)
        and not list((run / "jobs").glob("*/failure.json")),
        "Production failure present",
    )
    require(
        not (run / "acceptance-plan.json").exists() and not list(run.glob("final-*")),
        "Unexpected final access evidence",
    )
    transitions = [e for e in events if e["event"] == "evaluation-start"]
    require(
        len(transitions) == 1 and transitions[0]["training_seconds"] == session["training_seconds"],
        "Time transition differs",
    )
    require(
        0 < session["training_seconds"] < 7200
        and 0 < session["evaluation_seconds"] < 3600
        and session["training_seconds"] + session["evaluation_seconds"] < 10800,
        "Time cap exceeded",
    )
    peak_charge = max(e.get("artifact_bytes", 0) for e in events)
    require(
        max(peak_charge, session["artifact_bytes_charged"]) < 20 * 1024**3, "Artifact cap exceeded"
    )
    binding = read(run / "training-binding.json")
    require(
        binding["attempt_id"] == manifest["attempt_id"] == "phase17-1-base2-attempt1"
        and binding["initial_seed"] == 160
        and binding["fixture_sha256"] is None
        and binding["bundle_sha256"] == manifest["bundle_sha256"]
        and binding["plan_sha256"] == sha(root / "bundle/plan.json"),
        "Fresh attempt identity differs",
    )
    require(
        binding["controls"]["numerical_policy"] == plan["numerical_policy"],
        "Numerical policy differs",
    )
    require(
        binding["model_config"] == read("configs/training2/selected-model.json"),
        "Selected architecture differs",
    )
    expected_config = {
        "schema_version": 1,
        "sequence_length": 512,
        **{
            k: plan["training"][k]
            for k in (
                "seed",
                "batch_size",
                "accumulation_steps",
                "max_steps",
                "warmup_steps",
                "learning_rate",
                "min_lr_ratio",
                "weight_decay",
                "beta1",
                "beta2",
                "eps",
                "max_grad_norm",
            )
        },
    }
    require(
        binding["training_config"] == expected_config, "Approved training configuration differs"
    )
    runtime = binding["controls"]["runtime"]
    for directory in (run / "jobs").iterdir():
        if directory.name == "integrity":
            continue
        env = read(directory / "environment.json")
        require(
            env["request"] == requests[directory.name]
            and env["controller_source_commit"] == SOURCE,
            "Worker request/source identity differs",
        )
        require(
            all(env[k] == v for k, v in binding["controls"].items()),
            "Worker runtime/policy differs",
        )
        imported = env["imported_package"].replace("\\", "/").lower()
        suffix = (
            "/source/.wheel-env/lib/site-packages/latos/__init__.py"
            if requests[directory.name]["runtime"] == "wheel"
            else "/source/src/latos/__init__.py"
        )
        require(imported.endswith(suffix), "Wrong runtime import path")
        complete = read(directory / "result.json")
        require(
            complete["status"] == "complete"
            and complete["optimizer_updates"] == requests[directory.name]["reserved_updates"],
            "Worker completion/update reservation differs",
        )
    require(
        runtime["implementation_sha256"] == plan["source"]["original_core_implementation_sha256"]
        and runtime["torch"] == plan["numerical_policy"]["torch"]
        and runtime["python"] == plan["numerical_policy"]["python"]
        and runtime["deterministic_algorithms"] is True
        and runtime["cuda_matmul_tf32"] is False,
        "Native runtime differs",
    )
    checkpoints = sorted((run / "checkpoints").glob("step-*"))
    require(
        [int(p.name.split("-")[1]) for p in checkpoints] == plan["training"]["checkpoint_steps"],
        "Checkpoint set differs",
    )
    start = read(run / "jobs/train-0/start.json")
    state_check(start["state"])
    require(
        start["fresh_random"] is True
        and start["resume_checkpoint"] == "None"
        and start["binding"] == binding
        and not start["state"]["optimizer"]
        and start["state"]["sampler"]["step"] == 0,
        "Training did not start fresh",
    )
    spec = importlib.util.spec_from_file_location(
        "local_fingerprints", "experiments/recovery-diagnostic/mechanics.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    layout = read("experiments/phase-16/full-corpus-layout.json")
    train = FullDataset(cache / "train", expected_accounting(layout, "train"), TOKENIZER, "train")
    dev = FullDataset(
        cache / "development", expected_accounting(layout, "development"), TOKENIZER, "development"
    )
    require(
        binding["datasets"] == {"train": train.identity, "validation": dev.identity},
        "Accepted data differs",
    )
    for split in ("train", "development"):
        require(
            (run / f"cache-{split}/cache.json").read_bytes()
            == (cache / split / "cache.json").read_bytes(),
            "Cache metadata differs",
        )
    generator = torch.Generator().manual_seed(160)
    order_tensor = torch.randperm(len(train.windows), generator=generator)
    require(
        m.tensor_digest(order_tensor) == start["state"]["sampler"]["order"]
        and m.tensor_digest(generator.get_state()) == start["state"]["sampler"]["rng"],
        "Fixed shuffle differs",
    )
    order = order_tensor.tolist()
    schedule = read(run / "schedule.json")
    require(
        sha(run / "schedule.json") == binding["schedule_sha256"]
        and schedule["config"] == binding["training_config"]
        and len(schedule["rates"]) == 7485,
        "Schedule binding differs",
    )
    initial_groups = read(checkpoints[0] / "state.json")["optimizer_groups"]
    require(len(initial_groups) == 2, "Optimizer grouping differs")
    for index, group in enumerate(initial_groups):
        require(
            group["betas"] == [0.9, 0.95]
            and group["eps"] == 1e-8
            and group["weight_decay"] == (0.01 if index == 0 else 0)
            and group["lr"] == 0.0003
            and group["foreach"] is False
            and group["fused"] is False,
            "Approved optimizer settings differ",
        )
    peak_host, peak_gpu, cumulative, snapshots, count = 0, 0, 0, {0: start["state"]}, 0
    for step, row in enumerate(rows(run / "jobs/train-0/updates.jsonl"), 1):
        require(step <= 7485, "Too many updates")
        indices = order[(step - 1) * 16 : min(step * 16, len(order))]
        windows = [train.windows[i] for i in indices]
        targets = sum(len(w) - 1 for w in windows)
        cumulative += targets
        input_digest = hashlib.sha256()
        for window in windows:
            input_digest.update(canonical_json(window))
        require(
            row["input"]
            == {
                "sha256": input_digest.hexdigest(),
                "windows": len(windows),
                "targets": targets,
                "lengths": [len(w) for w in windows],
            },
            "Input signature differs",
        )
        check_metric(
            row["metric"],
            step,
            min(step * 16, len(order)),
            targets,
            cumulative,
            schedule["rates"][step - 1],
        )
        state = row["state"]
        state_check(state)
        for key in (
            "global_rng",
            "model_config",
            "training_config",
            "datasets",
            "optimizer_parameter_names",
        ):
            require(state[key] == start["state"][key], "State configuration/RNG differs")
        require(
            state["sampler"]
            == {
                **start["state"]["sampler"],
                "step": step,
                "targets": cumulative,
                "windows": min(step * 16, len(order)),
                "cursor": min(step * 16, len(order)),
                "epoch": 0,
            },
            "Sampler counters differ",
        )
        groups = copy.deepcopy(initial_groups)
        for group in groups:
            group["lr"] = schedule["rates"][step - 1]
        require(
            hashlib.sha256(canonical_json(groups)).hexdigest() == state["optimizer_groups"],
            "Optimizer settings differ",
        )
        resource = row["resources"]
        peak_host = max(peak_host, resource["host_peak_working_set_bytes"])
        peak_gpu = max(peak_gpu, resource["peak_reserved_bytes"])
        if step in plan["training"]["checkpoint_steps"]:
            snapshots[step] = state
        count = step
    require(count == 7485 and cumulative == 37811418, "Incomplete serious training")
    fixture_updates = 0
    for mode in ("checkout", "wheel"):
        for fixture in ("ordinary", "partial-tail"):
            ref = run / "jobs" / f"{mode}-{fixture}-reference"
            resumed = run / "jobs" / f"{mode}-{fixture}-resumed"
            a, b = list(rows(ref / "updates.jsonl")), list(rows(resumed / "updates.jsonl"))
            require(len(a) == 8 and len(b) == 5, "Incomplete CUDA preflight pair")
            fplan = plan["adapter_preflight"]
            fp = next(p for p in fplan["pairs"] if p["name"] == fixture)
            synthetic = []
            for i in range(fp["windows"]):
                length = 2 + (29 * i) % 511
                synthetic.append(
                    tuple([1, *[4 + ((7 * i + 3 * j) % 252) for j in range(1, length - 1)], 2])
                )
            fixture_digest = hashlib.sha256(
                (json.dumps(synthetic, sort_keys=True, allow_nan=False) + "\n").encode()
            ).hexdigest()
            fstart = read(ref / "start.json")
            require(
                fstart["fresh_random"] is True
                and not fstart["state"]["optimizer"]
                and fstart["binding"]["initial_seed"] == fp["model_seed"]
                and fstart["binding"]["fixture_sha256"] == fixture_digest
                and fstart["state"]["model_config"] == fplan["model_config"]
                and fstart["state"]["training_config"]
                == {"schema_version": 1, **fplan["training"]},
                "Prespecified fixture configuration differs",
            )
            permutation = torch.randperm(
                fp["windows"], generator=torch.Generator().manual_seed(160)
            ).tolist()
            fixture_targets = 0
            for step, item in enumerate(a, 1):
                batch = [synthetic[i] for i in permutation[(step - 1) * 4 : step * 4]]
                signature = hashlib.sha256()
                for window in batch:
                    signature.update(canonical_json(window))
                targets = sum(len(w) - 1 for w in batch)
                fixture_targets += targets
                state_check(item["state"])
                require(
                    item["input"]
                    == {
                        "sha256": signature.hexdigest(),
                        "windows": len(batch),
                        "targets": targets,
                        "lengths": [len(w) for w in batch],
                    }
                    and item["metric"]["tokens_seen"] == fixture_targets
                    and item["metric"]["step"] == step
                    and item["state"]["global_rng"] == fstart["state"]["global_rng"],
                    "Fixture data/exposure/RNG differs",
                )
            require(
                read(resumed / "start.json")["state"]
                == read(ref / "checkpoints/step-00000003/policy.json")["state"],
                "Fixture restore differs",
            )
            for old, new in zip(a[3:], b, strict=True):
                state_check(old["state"])
                state_check(new["state"])
                require(
                    old["state"] == new["state"] and old["input"] == new["input"],
                    "Fixture state/input differs",
                )
                require(
                    old["metric"]["learning_rate"] == new["metric"]["learning_rate"]
                    and abs(old["metric"]["loss"] - new["metric"]["loss"]) <= 1e-5,
                    "Fixture replay gate failed",
                )
            require(
                a[-1]["metric"]["microbatches"] == (1 if fixture == "partial-tail" else 2),
                "Fixture tail differs",
            )
            fixture_updates += 13
    checkpoint_ids = []
    for cp in checkpoints:
        policy, metadata = read(cp / "policy.json"), read(cp / "state.json")
        state_check(policy["state"])
        require(
            policy["binding"] == binding
            and policy["roundtrip_exact"] is True
            and policy["state"] == snapshots[metadata["step"]],
            "Checkpoint policy/trace differs",
        )
        prefix = cp.relative_to(root).as_posix() + "/"
        native = {
            n.removeprefix(prefix): v
            for n, v in all_files.items()
            if n.startswith(prefix) and n != prefix + "policy.json"
        }
        require(native == policy["native_files"], "Checkpoint payload inventory differs")
        require(
            metadata["runtime"] == runtime
            and metadata["config"] == binding["training_config"]
            and metadata["datasets"] == binding["datasets"]
            and metadata["precision"] == "bfloat16",
            "Checkpoint native contract differs",
        )
        require(
            metadata["training_sha256"] == native["training.safetensors"]["sha256"]
            and metadata["model_metadata_sha256"] == sha(cp / "model/metadata.json"),
            "Checkpoint tensor binding differs",
        )
        sampler = policy["state"]["sampler"]
        require(
            metadata["tokens_seen"] == sampler["targets"]
            and metadata["windows_seen"] == sampler["windows"]
            and metadata["cursor"] == sampler["cursor"]
            and metadata["epoch"] == 0
            and metadata["kind"] == "latos-training-single-pass-v3",
            "Checkpoint counters/kind differ",
        )
        checkpoint_ids.append(
            {
                "step": metadata["step"],
                "policy_sha256": sha(cp / "policy.json"),
                "model_sha256": native["model/model.safetensors"]["sha256"],
            }
        )
    trained = read(run / "training-result.json")
    require(
        trained == read(run / "jobs/train-0/result.json")
        and trained["status"] == "complete"
        and trained["optimizer_updates"] == 7485
        and trained["recorded_overlap_updates"] == 0
        and trained["replayed_logged_targets"] == 0
        and session["serious_updates_upper"] == 7485
        and session["preflight_updates_upper"] == fixture_updates == 52,
        "Training/replay accounting differs",
    )
    model = run / "checkpoints/step-00007485/model/model.safetensors"
    require(sha(model) == trained["candidate_weights"], "Final candidate bytes differ")
    tensor_hashes, elements = {}, 0
    with safe_open(model, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            require(
                tensor.dtype == torch.float32 and torch.isfinite(tensor).all().item(),
                "Invalid final model tensor",
            )
            tensor_hashes[name] = m.tensor_digest(tensor)
            elements += tensor.numel()
    require(
        tensor_hashes == snapshots[7485]["model"] and elements == 34087424,
        "Final model does not match training trace",
    )
    result = compare(run / "development-reference", run / "development-candidate", "base")
    require(
        result == read(run / "development-comparison.json")
        and result["candidate_weights"] == trained["candidate_weights"],
        "Development comparison differs",
    )
    for label in ("reference", "candidate"):
        ev = read(run / f"development-{label}/manifest.json")
        require(
            ev["final_access"] is None
            and ev["data"]["split"] == "validation"
            and ev["protocol_sha256"] == plan["identities"]["configs/evaluation/protocol-v1.json"]
            and ev["environment"]["batch_size"] == 8
            and ev["environment"]["threads"] == 4,
            "Evaluation scope differs",
        )
    inference = read(run / "inference.json")
    require(
        inference["median_first_token_seconds"]
        == statistics.median(x["first_token_seconds"] for x in inference["repeats"])
        and inference["median_decode_tokens_per_second"]
        == statistics.median(x["decode_tokens_per_second"] for x in inference["repeats"]),
        "Inference medians differ",
    )
    inference_pass = (
        inference["median_first_token_seconds"] <= 1
        and inference["median_decode_tokens_per_second"] >= 10
    )
    require(inference["passed"] == inference_pass, "Inference gate differs")
    for p in (run / "jobs").glob("*/result.json"):
        resource = read(p).get("resources", {})
        peak_host = max(peak_host, resource.get("host_peak_working_set_bytes", 0))
        peak_gpu = max(peak_gpu, resource.get("peak_reserved_bytes", 0))
    require(
        peak_host <= 8 * 1024**3 and peak_gpu <= 0.85 * binding["controls"]["gpu_total_bytes"],
        "Memory cap exceeded",
    )
    monitoring = [{"step": 0, **read(run / "jobs/train-0/initial-monitoring.json")}]
    monitoring += [
        {"step": step, **read(run / f"jobs/train-0/monitoring-{step}.json")}
        for step in plan["training"]["checkpoint_steps"]
        if step
    ]
    require(
        all(
            x["targets"] == 804337
            and x["windows"] == 2552
            and x["split"] == "validation"
            and math.isfinite(x["loss"])
            for x in monitoring
        ),
        "Monitoring scope differs",
    )
    return {
        "disposition": disposition(True, result["passed"]),
        "return_sha256": RETURN,
        "controller_source_commit": SOURCE,
        "included_files_verified": 1366,
        "omitted_tensor_cache_records_not_rehashed": 81,
        "executed_source_snapshots_verified": 12,
        "cpu_preflight": cpu,
        "CUDA_preflight_pairs": 4,
        "CUDA_preflight_updates": 52,
        "serious_updates": count,
        "serious_replay_updates": 0,
        "windows": len(order),
        "targets": cumulative,
        "full_input_signatures_reconstructed": count,
        "checkpoint_identities": checkpoint_ids,
        "candidate_weights_sha256": trained["candidate_weights"],
        "candidate_bytes": model.stat().st_size,
        "candidate_parameters": elements,
        "final_tensor_fingerprints_match_trace": True,
        "training_preflight_seconds": session["training_seconds"],
        "development_inference_seconds": session["evaluation_seconds"],
        "total_active_seconds": session["training_seconds"] + session["evaluation_seconds"],
        "peak_host_bytes": peak_host,
        "peak_reserved_GPU_bytes": peak_gpu,
        "artifact_charge_bytes": session["artifact_bytes_charged"],
        "peak_journal_charge_bytes": peak_charge,
        "development_comparison": result,
        "inference": inference,
        "training_monitoring": monitoring,
        "final_scoring_executed": False,
        "accepted_base_model_2": False,
        "scope": "Saved execution/state records and development outputs reconstructed; "
        "included final tensors checked without forward pass. Omitted tensors not replayed "
        "on Mac. No final scoring or new optimization.",
        "phase18_authorized": False,
        "new_attempt_authorized": False,
        "publication_authorized": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.archive, args.receipt, args.root, args.cache)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in (
                    "development_comparison",
                    "checkpoint_identities",
                    "training_monitoring",
                    "inference",
                )
            },
            indent=2,
        )
    )
