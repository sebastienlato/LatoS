"""The Phase 0 command line: help, version, and environment diagnostics."""

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
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
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
