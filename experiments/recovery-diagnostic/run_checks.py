"""CPU-only engineering preflight for the existing Windows environment."""

import json
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from run import hash_file, verify_bundle, write_once


def main():
    started = time.monotonic()
    bundle = Path(__file__).resolve().parent
    plan, _ = verify_bundle(bundle)
    if sys.platform != "win32" or platform.python_version() != plan["python"]:
        raise ValueError("Use the original Windows Python 3.14.7 environment")
    output = bundle / "windows-preflight.json"
    junit = bundle / "windows-preflight.xml"
    if output.exists() or junit.exists():
        raise ValueError("Preflight already attempted; retain evidence")
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(bundle / "test_diagnostic.py"),
        "-q",
        "--junitxml",
        str(junit),
    ]
    with (bundle / "windows-preflight.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=bundle, stdout=log, stderr=subprocess.STDOUT, timeout=120
        )
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    if (
        result.returncode
        or len(cases) != 10
        or any(c.find(k) is not None for c in cases for k in ("failure", "error", "skipped"))
    ):
        raise ValueError("Diagnostic preflight tests did not all pass")
    receipt = {
        "passed": True,
        "tests": len(cases),
        "optimizer_updates": 0,
        "scope": "Tiny CPU state/data/control fixtures; Trainer.update forbidden in tests",
        "seconds": time.monotonic() - started,
        "platform": sys.platform,
        "bundle_sha256": hash_file(bundle / "bundle.json"),
        "junit_sha256": hash_file(junit),
    }
    write_once(output, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
