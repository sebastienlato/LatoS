"""Independently recount saved tool evidence and verify exact captured artifact bytes."""

import argparse
import json
from pathlib import Path

from latos.model.storage import file_hash


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root: Path):
    inventory = read(root / "artifacts.json")
    for name, identity in inventory.items():
        path = root / name
        if path.stat().st_size != identity["bytes"] or file_hash(path) != identity["sha256"]:
            raise ValueError(f"Artifact mismatch: {name}")
    plan = read(root / "plan.json")
    for name, expected in plan["source_sha256"].items():
        if file_hash(root / "source" / name) != expected:
            raise ValueError(f"Captured source mismatch: {name}")
    cases = {case["id"]: case for case in plan["cases"]}
    assert len(cases) == 16
    summary = read(root / "results.json")
    checked = 0
    for label in ("base", "sft", "dpo", "scripted"):
        result = read(root / f"{label}.json")
        assert [s["id"] for s in result["sessions"]] == list(cases)
        counters = dict.fromkeys(
            (
                "turns",
                "json",
                "envelopes",
                "arguments",
                "proposed",
                "correct",
                "expected",
                "executed",
                "execution_ok",
                "final",
                "success",
                "failure",
                "missing",
                "tokens",
            ),
            0,
        )
        stops = {}
        for session in result["sessions"]:
            case = cases[session["id"]]
            events = session["events"]
            counters["turns"] += len(events)
            counters["expected"] += len(case["calls"])
            missing = "error" in case["expected"]
            counters["missing"] += missing
            proposed, executed, outputs = [], [], []
            for event in events:
                counters["tokens"] += event["generated_tokens"]
                try:
                    value = json.loads(event["raw"])
                except ValueError, TypeError:
                    assert not event["json_valid"]
                    continue
                # This fixed evidence has only plain valid JSON or non-JSON model text.
                assert event["json_valid"]
                counters["json"] += 1
                assert isinstance(value, dict)
                assert set(value) in ({"tool", "args"}, {"final"}, {"error"})
                counters["envelopes"] += 1
                if "tool" not in value:
                    continue
                proposed.append(value)
                name, args = value["tool"], value["args"]
                if name == "add":
                    assert set(args) == {"a", "b"}
                    assert all(type(v) is int and -1000 <= v <= 1000 for v in args.values())
                    output = {"ok": True, "value": args["a"] + args["b"]}
                else:
                    assert name == "lookup" and set(args) == {"key"}
                    catalog = {"oak": 7, "elm": 11, "ash": 13}
                    output = (
                        {"ok": True, "value": catalog[args["key"]]}
                        if args["key"] in catalog
                        else {"ok": False, "error": "not_found"}
                    )
                counters["arguments"] += 1
                if event["result"] is not None:
                    assert event["generation_stop"] == "eos"
                    assert event["result"] == output
                    executed.append(value)
                    outputs.append(output)
            counters["proposed"] += len(proposed)
            counters["executed"] += len(executed)
            counters["execution_ok"] += sum(o["ok"] for o in outputs)
            counters["correct"] += sum(
                a == b for a, b in zip(proposed, case["calls"], strict=False)
            )
            final = session["final"] == case["expected"]
            counters["final"] += final
            exact = proposed == executed == case["calls"]
            counters["success"] += not missing and exact and final and all(o["ok"] for o in outputs)
            counters["failure"] += (
                missing and exact and final and outputs == [{"ok": False, "error": "not_found"}]
            )
            assert session["calls"] == len(executed) <= 2 and len(events) <= 4
            stops[session["stop_reason"]] = stops.get(session["stop_reason"], 0) + 1
            checked += 1
        pairs = {
            "parsing": (counters["json"], counters["turns"]),
            "envelopes": (counters["envelopes"], counters["turns"]),
            "argument_schema": (counters["arguments"], counters["proposed"]),
            "correct_tool_and_arguments": (counters["correct"], counters["expected"]),
            "execution_success": (counters["execution_ok"], counters["executed"]),
            "final_answer_correct": (counters["final"], 16),
            "task_success": (counters["success"], 16 - counters["missing"]),
            "failure_handling": (counters["failure"], counters["missing"]),
        }
        expected_metrics = {"cases": 16, "stops": stops}
        for name, (n, d) in pairs.items():
            expected_metrics[name] = {
                "numerator": n,
                "denominator": d,
                "rate": n / d if d else None,
            }
        assert result["metrics"] == expected_metrics
        if label != "scripted":
            assert summary[label]["metrics"] == expected_metrics
            assert counters["tokens"] == result["generated_tokens"]
        else:
            assert summary["scripted_control"] == expected_metrics
    return {
        "status": "pass",
        "sessions_recounted": checked,
        "inventory_files_verified": len(inventory),
        "source_files_verified": len(plan["source_sha256"]),
        "scope": "Fixed saved evidence recount; not model re-execution or general JSON validator",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2))
