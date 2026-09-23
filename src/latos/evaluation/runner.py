"""Reproducible local native-model evaluation, separate from training and acquisition."""

import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch

from latos.chat import CHAT_CONTRACT
from latos.data.manifest import canonical_json, sha256
from latos.doctor import available_backends, select_device
from latos.evaluation.access import evaluation_access
from latos.evaluation.behavior import generate_case, instruction_correct
from latos.evaluation.inputs import acquire, external_cases, read_json, validate_suite, write_json
from latos.evaluation.scoring import (
    accuracy,
    continuation,
    legacy_windows,
    lm_summary,
    matched_windows,
    score_sequences,
)
from latos.inference.stream import synchronize
from latos.model.storage import bind_tokenizer, file_hash, load_model
from latos.tokenization.corpus import read_split

ROOT = Path(__file__).resolve().parents[3]


def evaluate_lm(model, codec, texts, documents, *, mode, batch_size, context):
    windows, counts = [], []
    for text in texts:
        rows = (
            legacy_windows(codec, text, context)
            if mode == "legacy"
            else matched_windows(codec, text)
        )
        counts.append(len(rows))
        windows.extend(rows)
    scores = score_sequences(model, windows, batch_size=batch_size, context=context)
    cursor, results = 0, []
    for i, (text, count, document) in enumerate(zip(texts, counts, documents, strict=True)):
        group = scores[cursor : cursor + count]
        cursor += count
        results.append(
            {
                "id": str(i),
                "document": document,
                "text_sha256": sha256(text.encode("utf-8")),
                "nll": math.fsum(s["nll"] for s in group),
                "targets": sum(s["targets"] for s in group),
                **({"bytes": len(text.encode("utf-8"))} if mode == "matched" else {}),
            }
        )
    return results, {
        **lm_summary(results),
        "documents": {
            name: lm_summary([r for r in results if r["document"] == name])
            for name in sorted(set(documents))
        },
    }


def evaluate_mc(model, codec, cases, *, batch_size, context):
    results = []
    for case in cases:
        prefix = "Question: " + case["question"] + "\nAnswer:"
        sequences = [continuation(codec, prefix, answer) for answer in case["choices"]]
        if any(len(row.ids) > context for row in sequences):
            results.append(
                {
                    "id": case["id"],
                    "status": "context_overflow",
                    "correct": False,
                    "correct_byte_normalized": False,
                    "gold": case["gold"],
                    "choices": len(case["choices"]),
                    "content_sha256": case["content_sha256"],
                }
            )
            continue
        scored = score_sequences(model, sequences, batch_size=batch_size, context=context)
        logps = [-s["nll"] for s in scored]
        normalized = [
            p / len((" " + a).encode("utf-8")) for p, a in zip(logps, case["choices"], strict=True)
        ]
        prediction = max(range(len(logps)), key=logps.__getitem__)
        norm_prediction = max(range(len(logps)), key=normalized.__getitem__)
        results.append(
            {
                "id": case["id"],
                "status": "ok",
                "gold": case["gold"],
                "choices": len(logps),
                "content_sha256": case["content_sha256"],
                "log_likelihoods": logps,
                "byte_normalized_log_likelihoods": normalized,
                "target_counts": [s["targets"] for s in scored],
                "answer_bytes": [len((" " + a).encode("utf-8")) for a in case["choices"]],
                "prediction": prediction,
                "byte_normalized_prediction": norm_prediction,
                "ties": logps.count(max(logps)),
                "correct": prediction == case["gold"],
                "correct_byte_normalized": norm_prediction == case["gold"],
            }
        )
    return results, {
        "raw": accuracy(results),
        "byte_normalized": accuracy(results, "correct_byte_normalized"),
        "uniform_chance": math.fsum(1 / r["choices"] for r in results) / len(results),
        "first_choice": sum(r["gold"] == 0 for r in results) / len(results),
        "failures": sum(r["status"] != "ok" for r in results),
    }


def summarize_behavior(instructions, generations, suite):
    by_category = {
        category: accuracy([r for r in instructions if r["category"] == category])
        for category in sorted({r["category"] for r in instructions})
    }
    generated = [r for r in generations if r["status"] == "ok"]
    if len(generated) != len(generations):
        gen_summary = {"failures": len(generations) - len(generated)}
    else:
        gen_summary = {
            "failures": 0,
            "cases": len(generated),
            **{
                key: math.fsum(r["observations"][key] for r in generated) / len(generated)
                for key in (
                    "empty",
                    "replacement_character",
                    "eos",
                    "word_count",
                    "repeated_trigram_fraction",
                )
            },
        }
    controls = {"empty": "", "constant_yes": "yes"}
    control_results = {
        name: sum(instruction_correct(value, c) for c in suite["instructions"])
        for name, value in controls.items()
    }
    control_results["prompt_echo"] = sum(
        instruction_correct(c["prompt"], c) for c in suite["instructions"]
    )
    return {
        "overall": accuracy(instructions),
        "categories": by_category,
        "failures": sum(r["status"] != "ok" for r in instructions),
        "trivial_control_correct": control_results,
    }, gen_summary


def capture_source(output, protocol_path, suite_path, external_path):
    source_root = Path(__file__).resolve().parents[1]
    paths = list(source_root.rglob("*.py"))
    source = output / "source"
    source.mkdir()
    identities = {}
    for path in paths:
        relative = Path("latos") / path.relative_to(source_root)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        identities[relative.as_posix()] = file_hash(path)
    for path in [
        protocol_path,
        suite_path,
        external_path,
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
    ]:
        if path.exists():
            shutil.copyfile(path, source / path.name)
            identities[path.name] = file_hash(path)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
    except OSError, subprocess.CalledProcessError:
        commit, dirty = None, None
    return {
        "commit": commit,
        "dirty": dirty,
        "files": identities,
        "snapshot_sha256": sha256(canonical_json(identities)),
    }


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        protocol_path, suite_path, external_path = map(
            Path, [args.protocol, args.suite, args.external_manifest]
        )
        protocol = read_json(protocol_path)
        fixed = {
            "schema_version": 1,
            "matched_span_bytes": 192,
            "temperature": 0,
            "top_k": 0,
            "seed": 0,
            "chat_contract": 1,
        }
        if any(protocol.get(key) != value for key, value in fixed.items()):
            raise ValueError("Unsupported evaluation implementation contract")
        split, access = evaluation_access(
            protocol, suite_path, args.weights_sha256, getattr(args, "acceptance_plan", None)
        )
        for path, key in [(external_path, "external_sha256")]:
            if file_hash(path) != protocol[key]:
                raise ValueError("Frozen suite/manifest checksum mismatch")
        suite = read_json(suite_path)
        validate_suite(suite)
        if (
            suite["version"] not in (protocol["suite_version"], "latos-eval-final-1")
            or not suite["provenance"]
        ):
            raise ValueError("Suite identity/provenance mismatch")
        if len({c["id"] for c in suite["instructions"] + suite["generation"]}) != len(
            suite["instructions"]
        ) + len(suite["generation"]):
            raise ValueError("Duplicate suite case IDs")
        torch.set_num_threads(args.threads)
        device = torch.device(select_device(args.device, available_backends()))
        model_dir, tokenizer_dir = Path(args.model_dir), Path(args.tokenizer_dir)
        if file_hash(model_dir / "model.safetensors") != args.weights_sha256:
            raise ValueError("Selected model weights checksum mismatch")
        model = load_model(model_dir)
        codec = bind_tokenizer(model.config, tokenizer_dir)
        model.to(device).eval()
        context = protocol["context"]
        if model.config.context_length < context:
            raise ValueError("Model cannot support fixed evaluation context")
        texts, data_identity = read_split(Path(args.data_manifest), Path(args.corpus_dir), split)
        if data_identity["split_sha256"] != protocol[f"lm_{split}_sha256"]:
            raise ValueError("Frozen language-model split checksum mismatch")
        documents = [
            read_json_line["document_id"]
            for read_json_line in (
                json.loads(line)
                for line in (Path(args.corpus_dir) / f"{split}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        ]
        benchmarks = external_cases(external_path, args.external_dir)
        manifest = {
            "schema_version": 1,
            "status": "running",
            "kind": "retrospective-or-candidate-evaluation",
            "source": capture_source(output, protocol_path, suite_path, external_path),
            "protocol": protocol,
            "protocol_sha256": file_hash(protocol_path),
            "suite_sha256": file_hash(suite_path),
            "external_sha256": file_hash(external_path),
            "model": {
                "weights_sha256": args.weights_sha256,
                "config": model.config.to_dict(),
                "files": {p.name: file_hash(p) for p in model_dir.iterdir() if p.is_file()},
            },
            "tokenizer": {p.name: file_hash(p) for p in tokenizer_dir.iterdir() if p.is_file()},
            "data": data_identity,
            "final_access": access,
            "chat_contract": CHAT_CONTRACT,
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "platform": platform.platform(),
                "device": str(device),
                "threads": args.threads,
                "batch_size": args.batch_size,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "packages": {
                    name: version(name)
                    for name in ("latos", "torch", "tokenizers", "safetensors", "numpy")
                },
                "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cuda_device": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
            },
            "failures": [],
            "skips": [],
        }
        write_json(output / "manifest.json", manifest)
        write_json(
            output / "inputs.json",
            {
                "lm": [
                    {
                        "id": str(i),
                        "document": document,
                        "text_sha256": sha256(text.encode("utf-8")),
                        "bytes": len(text.encode("utf-8")),
                    }
                    for i, (text, document) in enumerate(zip(texts, documents, strict=True))
                ],
                "external": {
                    name: [
                        {
                            "id": c["id"],
                            "gold": c["gold"],
                            "choices": len(c["choices"]),
                            "content_sha256": c["content_sha256"],
                        }
                        for c in cases
                    ]
                    for name, cases in benchmarks.items()
                },
            },
        )
        summaries = {}
        for mode in ("legacy", "matched"):
            rows, summaries[mode] = evaluate_lm(
                model,
                codec,
                texts,
                documents,
                mode=mode,
                batch_size=args.batch_size,
                context=context,
            )
            write_json(output / f"lm-{mode}.json", rows)
        summaries["uniform_token_control"] = {
            "legacy_nats_per_token": math.log(codec.vocab_size),
            "matched_bits_per_byte": math.log2(codec.vocab_size)
            * summaries["matched"]["targets"]
            / summaries["matched"]["bytes"],
            "interpretation": "Analytical uniform token predictor; not learned model output",
        }
        summaries["external"] = {}
        for name, cases in benchmarks.items():
            rows, summaries["external"][name] = evaluate_mc(
                model, codec, cases, batch_size=args.batch_size, context=context
            )
            write_json(output / f"{name}.json", rows)
        instructions = [
            generate_case(
                model,
                codec,
                case,
                instruction=True,
                context=context,
                budget=protocol["instruction_tokens"],
            )
            for case in suite["instructions"]
        ]
        generations = [
            generate_case(
                model,
                codec,
                case,
                instruction=False,
                context=context,
                budget=protocol["generation_tokens"],
            )
            for case in suite["generation"]
        ]
        write_json(output / "instructions.json", instructions)
        write_json(output / "generation.json", generations)
        summaries["instructions"], summaries["generation"] = summarize_behavior(
            instructions, generations, suite
        )
        synchronize(device)
        summaries["resources"] = {
            "seconds_including_load_and_capture": time.perf_counter() - start,
            "model_parameters": model.parameter_count,
            "generated_tokens": sum(
                len(r.get("generated_ids", [])) for r in instructions + generations
            ),
        }
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            summaries["resources"]["process_lifetime_peak_rss_bytes"] = (
                rss if sys.platform == "darwin" else rss * 1024
            )
        except ImportError:
            summaries["resources"]["process_lifetime_peak_rss_bytes"] = None
        if device.type == "mps":
            summaries["resources"]["mps_end_boundary_allocated_bytes"] = (
                torch.mps.current_allocated_memory()
            )
            summaries["resources"]["mps_end_boundary_driver_bytes"] = (
                torch.mps.driver_allocated_memory()
            )
        if file_hash(model_dir / "model.safetensors") != args.weights_sha256:
            raise ValueError("Model artifact changed during evaluation")
        manifest["status"] = "complete"
        manifest["failures"] = [
            {"section": name, "count": value["failures"]}
            for name, value in [
                ("instructions", summaries["instructions"]),
                ("generation", summaries["generation"]),
                *summaries["external"].items(),
            ]
            if value["failures"]
        ]
        if manifest["failures"]:
            manifest["status"] = "incomplete"
        write_json(output / "summary.json", summaries)
        write_json(output / "manifest.json", manifest)
        inventory = {
            p.relative_to(output).as_posix(): file_hash(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        }
        write_json(output / "inventory.json", inventory)
        return summaries
    except Exception as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Fixed evaluation; never trains or reads reserved tests"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("acquire")
    download.add_argument("--manifest", default="configs/evaluation/external-v1.json")
    download.add_argument("--output", default="data/cache/evaluation-v1")
    evaluate = sub.add_parser("run")
    for name in ("model-dir", "weights-sha256", "tokenizer-dir", "output"):
        evaluate.add_argument("--" + name, required=True)
    evaluate.add_argument("--acceptance-plan", type=Path)
    evaluate.add_argument("--protocol", default="configs/evaluation/protocol-v1.json")
    evaluate.add_argument("--suite", default="configs/evaluation/suite-v1.json")
    evaluate.add_argument("--external-manifest", default="configs/evaluation/external-v1.json")
    evaluate.add_argument("--external-dir", default="data/cache/evaluation-v1")
    evaluate.add_argument("--data-manifest", default="data/manifests/english-books-v1.json")
    evaluate.add_argument("--corpus-dir", default="data/processed/english-books-v1")
    evaluate.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    evaluate.add_argument("--threads", type=int, choices=range(1, 65), default=1)
    evaluate.add_argument("--batch-size", type=int, choices=range(1, 65), default=8)
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(Path(args.manifest), Path(args.output))
    else:
        run(args)


if __name__ == "__main__":
    main()
