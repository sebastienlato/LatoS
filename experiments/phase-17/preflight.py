"""Bind passing checkout/wheel JUnit and actual imported source to a transfer."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from latos.data.manifest import canonical_json
from latos.model.storage import file_hash
from latos.training.base2 import frozen, verify_transfer
from latos.training.probe import cuda_environment

REQUIRED = {
    "test_cuda_bf16_numerical_control",
    "test_full_single_pass_cuda",
    "test_partial_update_matches_independent_token_sum",
    "test_compact_cache_transitions_and_corruption",
}


def check_junit(path):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    passed = {
        case.attrib["name"]
        for case in cases
        if not any(case.find(k) is not None for k in ("failure", "error", "skipped"))
    }
    if (
        not cases
        or not REQUIRED <= passed
        or any(case.find(k) is not None for case in cases for k in ("failure", "error"))
    ):
        raise ValueError("Required CUDA/cache/gradient tests did not execute and pass")
    return {
        "sha256": file_hash(path),
        "cases": len(cases),
        "passed": len([case for case in cases if case.find("skipped") is None]),
        "skipped": sum(case.find("skipped") is not None for case in cases),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--checkout-junit", type=Path, required=True)
    parser.add_argument("--wheel-junit", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    root = args.transfer.resolve()
    destination = root / "preflight.json"
    if destination.exists():
        raise ValueError("Preflight already exists; retain it")
    if args.checkout_junit.resolve() == args.wheel_junit.resolve():
        raise ValueError("Checkout and wheel must have separate executions")
    if (
        args.checkout_junit.resolve() != root / "validation/checkout.xml"
        or args.wheel_junit.resolve() != root / "validation/wheel.xml"
    ):
        raise ValueError("Keep JUnit evidence at validation/checkout.xml and validation/wheel.xml")
    manifest = verify_transfer(root)
    frozen(root / "source")
    result = {
        "schema_version": 1,
        "source_commit": manifest["source_commit"],
        "handoff_sha256": file_hash(root / "handoff.json"),
        "checkout": check_junit(args.checkout_junit),
        "wheel": check_junit(args.wheel_junit),
        "wheel_sha256": file_hash(args.wheel),
        "environment": cuda_environment(root),
        "passed": True,
    }
    # Receipts are evidence, not an authentication boundary. Work must retain exact logs.
    destination.write_bytes(canonical_json(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
