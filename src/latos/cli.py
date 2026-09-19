"""LatoS commands for diagnostics, data, tokenizers, and model checks."""

import argparse
import json
from pathlib import Path

from latos import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="latos", description="LatoS — English-first small language model development."
    )
    parser.add_argument("--version", action="version", version=f"LatoS {__version__}")
    commands = parser.add_subparsers(dest="command")
    doctor = commands.add_parser("doctor", help="Inspect this host and test a PyTorch backend")
    doctor.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable diagnostics")
    data = commands.add_parser(
        "data", help="Acquire, prepare, or audit a documented English corpus"
    )
    actions = data.add_subparsers(dest="data_action", required=True)
    for name in ("acquire", "prepare", "audit"):
        action = actions.add_parser(name)
        action.add_argument("--manifest", type=Path, required=True)
        if name != "audit":
            action.add_argument("--raw-dir", type=Path, required=True)
        if name != "acquire":
            action.add_argument("--output-dir", type=Path, required=True)
    tokenizer = commands.add_parser("tokenizer", help="Train, inspect, evaluate, or use LatoS BPE")
    token_actions = tokenizer.add_subparsers(dest="token_action", required=True)
    for name in ("train", "evaluate", "inspect", "encode"):
        action = token_actions.add_parser(name)
        if name in ("train", "evaluate"):
            action.add_argument("--manifest", type=Path, required=True)
            action.add_argument("--corpus-dir", type=Path, required=True)
        if name == "train":
            action.add_argument("--config", type=Path, required=True)
            action.add_argument("--output-dir", type=Path, required=True)
        else:
            action.add_argument("--artifact-dir", type=Path, required=True)
        if name == "evaluate":
            action.add_argument("--split", choices=("train", "validation", "test"), required=True)
        if name == "encode":
            action.add_argument("--text", required=True)
            action.add_argument("--bos", action="store_true")
            action.add_argument("--eos", action="store_true")
    model = commands.add_parser("model", help="Inspect, check, initialize, or sample a dense model")
    model_actions = model.add_subparsers(dest="model_action", required=True)
    for name in ("inspect", "check", "initialize", "sample"):
        action = model_actions.add_parser(name)
        if name != "sample":
            action.add_argument("--config", type=Path, required=True)
        if name in ("check", "sample"):
            action.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
        if name in ("initialize", "sample"):
            action.add_argument("--tokenizer-dir", type=Path, required=True)
            action.add_argument("--seed", type=int, default=0)
        if name == "initialize":
            action.add_argument("--output-dir", type=Path, required=True)
        if name == "sample":
            action.add_argument("--model-dir", type=Path, required=True)
            action.add_argument("--prompt", required=True)
            action.add_argument("--max-new-tokens", type=int, default=16)
            action.add_argument("--temperature", type=float, default=0.0)
            action.add_argument("--top-k", type=int, default=0)
    training = commands.add_parser("train", help="Train or resume a float32 dense model")
    training.add_argument("--manifest", type=Path, required=True)
    training.add_argument("--corpus-dir", type=Path, required=True)
    training.add_argument("--tokenizer-dir", type=Path, required=True)
    training.add_argument("--config", type=Path, required=True)
    origin = training.add_mutually_exclusive_group(required=True)
    origin.add_argument("--model-config", type=Path)
    origin.add_argument("--resume", type=Path)
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    training.add_argument("--threads", type=int, default=1)
    training.add_argument("--validate-every", type=int, default=10)
    training.add_argument("--checkpoint-every", type=int, default=25)
    training.add_argument(
        "--stop-after", type=int, help="Stop at this update without changing the schedule"
    )
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "train":
        from latos.training.run import run_training

        try:
            report = run_training(args)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
            return 1
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "data":
        from latos.data.acquire import acquire
        from latos.data.prepare import audit, prepare

        try:
            if args.data_action == "acquire":
                report = acquire(args.manifest, args.raw_dir)
            elif args.data_action == "prepare":
                report = prepare(args.manifest, args.raw_dir, args.output_dir)
            else:
                report = audit(args.manifest, args.output_dir)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
            return 1
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "tokenizer":
        from latos.tokenization import LatoTokenizer
        from latos.tokenization.training import evaluate, train

        try:
            if args.token_action == "train":
                report = train(args.manifest, args.corpus_dir, args.config, args.output_dir)
            elif args.token_action == "evaluate":
                report = evaluate(args.manifest, args.corpus_dir, args.artifact_dir, args.split)
            else:
                codec = LatoTokenizer.load(args.artifact_dir)
                if args.token_action == "inspect":
                    report = json.loads((args.artifact_dir / "metadata.json").read_text())
                else:
                    ids = codec.encode(args.text, add_bos=args.bos, add_eos=args.eos)
                    report = {"ids": ids, "decoded": codec.decode(ids)}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
            return 1
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "model":
        from latos.doctor import available_backends, select_device
        from latos.model import ModelConfig, create_model
        from latos.model.diagnostics import check_model
        from latos.model.sampling import generate
        from latos.model.storage import bind_tokenizer, load_model, save_model

        try:
            if args.model_action == "sample":
                model = load_model(args.model_dir)
                codec = bind_tokenizer(model.config, args.tokenizer_dir)
                model.to(select_device(args.device, available_backends()))
                report = generate(
                    model,
                    codec.encode(args.prompt, add_bos=True),
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=args.seed,
                )
                report["text"] = codec.decode(report["ids"])
                report["quality_note"] = (
                    "Mechanism check; snapshot training history is not included"
                )
            else:
                config = ModelConfig.load(args.config)
                if args.model_action == "check":
                    report = check_model(config, args.device)
                elif args.model_action == "inspect":
                    model = create_model(config)
                    report = {
                        "config": config.to_dict(),
                        "parameter_count": model.parameter_count,
                        "float32_parameter_bytes": model.parameter_count * 4,
                        "weights": "random initialization",
                    }
                else:
                    bind_tokenizer(config, args.tokenizer_dir)
                    report = save_model(create_model(config, seed=args.seed), args.output_dir)
                    report = {
                        **report,
                        "initialization_seed": args.seed,
                        "weights": "random initialization",
                    }
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
            return 1
        print(json.dumps(report, indent=2))
        return 0

    # Help and version remain usable when the tensor runtime cannot be imported.
    try:
        from latos.doctor import diagnose, format_report

        report = diagnose(args.device)
    except (ImportError, OSError, RuntimeError) as exc:
        report = {"schema_version": 1, "status": "error", "error": str(exc)}
        print(json.dumps(report, indent=2) if args.json else f"Doctor failed: {exc}")
        return 1
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    return 0 if report["status"] == "ok" else 1
