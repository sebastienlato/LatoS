"""Run the independent administration checks before the reviewed Windows recovery."""

import json
import platform
import sys
import unittest
from pathlib import Path

from recover import digest, verify_bundle, write_once


def main():
    root = Path(__file__).resolve().parent
    verify_bundle(root)
    if sys.platform != "win32" or platform.python_version() != "3.14.7":
        raise ValueError("Windows Python 3.14.7 required for execution receipt")
    output = root / "windows-checks.json"
    if output.exists():
        raise ValueError("Checks receipt exists; preserve the original")
    suite = unittest.defaultTestLoader.discover(str(root), pattern="test_recovery.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun != 9:
        raise ValueError(
            "All nine administrative tests, including actual Windows sharing, must pass"
        )
    receipt = {
        "passed": True,
        "platform": sys.platform,
        "python": platform.python_version(),
        "tests_passed": result.testsRun,
        "skipped": 0,
        "source_sha256": {
            name: digest(root / name)
            for name in ("recover.py", "test_recovery.py", "run_checks.py")
        },
        "scope": "Administration mechanics only; original CUDA controls remain separately bound",
    }
    write_once(output, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
