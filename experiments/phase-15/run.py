"""Execute data preparation only; no model construction, fitting or scoring."""

import argparse
import platform
import shutil
import sys
import time
from pathlib import Path

from latos.data2.acquire import file_hash, write_json
from latos.data2.prepare import prepare
from latos.data2.protected import build_guard
from latos.data2.resources import peak_rss_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    start = time.monotonic()
    snapshot = args.output.parent / (args.output.name + "-source")
    snapshot.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        root / "src/latos/data2", snapshot / "data2", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(root / "configs/data2", snapshot / "configs")
    shutil.copy2(__file__, snapshot / "run.py")
    guard = build_guard(root)
    print("Protected comparison index built without exporting examples", flush=True)
    try:
        report = prepare(
            root / "configs/data2/sources-v1.json", root / "data/raw/data2-v1", args.output, guard
        )
        write_json(
            args.output / "resources.json",
            {
                "seconds": time.monotonic() - start,
                "peak_rss_bytes": peak_rss_bytes(),
                "host": platform.platform(),
                "python": sys.version,
                "source": {
                    p.relative_to(snapshot).as_posix(): file_hash(p)
                    for p in sorted((snapshot / "data2").glob("*.py"))
                },
            },
        )
        print(report["splits"], flush=True)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
