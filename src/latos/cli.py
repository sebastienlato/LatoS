"""The Phase 0 command line: help, version, and environment diagnostics."""

import argparse
import json

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
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
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
