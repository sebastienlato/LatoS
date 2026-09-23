"""Original development cases and separate protocol/task metrics; no reserved data."""

from latos.tools import Reply, dumps, run_session


def call(name, **args):
    return {"tool": name, "args": args}


def development_cases() -> list[dict]:
    """Fixed literals: expected values are not computed by the dispatcher under test."""
    cases = []
    for a, b, answer in [(2, 3, "5"), (-8, 5, "-3"), (0, 0, "0"), (1000, -1000, "0")]:
        cases.append(
            {
                "id": f"add-{len(cases)}",
                "prompt": f"Use add to sum {a} and {b}.",
                "calls": [call("add", a=a, b=b)],
                "expected": {"final": answer},
            }
        )
    for key, answer in [("oak", "7"), ("elm", "11"), ("ash", "13"), ("oak", "7")]:
        cases.append(
            {
                "id": f"lookup-{len(cases)}",
                "prompt": f"Use lookup to find {key}.",
                "calls": [call("lookup", key=key)],
                "expected": {"final": answer},
            }
        )
    for key, value, offset, answer in [
        ("oak", 7, 5, "12"),
        ("elm", 11, -4, "7"),
        ("ash", 13, 9, "22"),
        ("oak", 7, -10, "-3"),
    ]:
        cases.append(
            {
                "id": f"chain-{len(cases)}",
                "prompt": f"Look up {key}, then use add to add {offset} to its value.",
                "calls": [call("lookup", key=key), call("add", a=value, b=offset)],
                "expected": {"final": answer},
            }
        )
    for key in ("pine", "birch", "cedar", "maple"):
        cases.append(
            {
                "id": f"missing-{key}",
                "prompt": f"Use lookup to find {key}; report any error.",
                "calls": [call("lookup", key=key)],
                "expected": {"error": "not_found"},
            }
        )
    # Deliberately distinct wording for the fourth simple lookup; no duplicate prompt.
    cases[7]["prompt"] = "Find the value for oak with lookup and return it."
    return cases


def score_case(case: dict, session: dict) -> dict:
    events = session["events"]
    proposed = [e for e in events if e["call"] is not None]
    executed = [e for e in proposed if e["result"] is not None]
    correct = sum(
        e["arguments_valid"] and e["call"] == expected
        for e, expected in zip(proposed, case["calls"], strict=False)
    )
    exact_calls = (
        len(executed) == len(case["calls"])
        and [e["call"] for e in executed] == case["calls"]
        and len(proposed) == len(executed)
    )
    failure = "error" in case["expected"]
    results_ok = all(e["result"]["ok"] for e in executed)
    expected_failure = (
        failure
        and len(executed) == 1
        and executed[0]["result"] == {"ok": False, "error": "not_found"}
    )
    final_correct = session["final"] == case["expected"]
    return {
        "id": case["id"],
        "failure_case": failure,
        "turns": len(events),
        "json_valid": sum(e["json_valid"] for e in events),
        "envelope_valid": sum(e["envelope_valid"] for e in events),
        "proposed_calls": len(proposed),
        "arguments_valid": sum(e["arguments_valid"] for e in proposed),
        "correct_calls": correct,
        "expected_calls": len(case["calls"]),
        "executed_calls": len(executed),
        "successful_executions": sum(e["result"]["ok"] for e in executed),
        "final_correct": final_correct,
        "task_success": not failure and exact_calls and results_ok and final_correct,
        "failure_handled": bool(expected_failure and exact_calls and final_correct),
        "stop_reason": session["stop_reason"],
    }


def fraction(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def summarize(scores: list[dict]) -> dict:
    def total(key):
        return sum(s[key] for s in scores)

    return {
        "cases": len(scores),
        "parsing": fraction(total("json_valid"), total("turns")),
        "envelopes": fraction(total("envelope_valid"), total("turns")),
        "argument_schema": fraction(total("arguments_valid"), total("proposed_calls")),
        "correct_tool_and_arguments": fraction(total("correct_calls"), total("expected_calls")),
        "execution_success": fraction(total("successful_executions"), total("executed_calls")),
        "final_answer_correct": fraction(total("final_correct"), len(scores)),
        "task_success": fraction(total("task_success"), sum(not s["failure_case"] for s in scores)),
        "failure_handling": fraction(total("failure_handled"), total("failure_case")),
        "stops": {
            reason: sum(s["stop_reason"] == reason for s in scores)
            for reason in sorted({s["stop_reason"] for s in scores})
        },
    }


def evaluate(respond) -> dict:
    sessions, scores = [], []
    for case in development_cases():
        session = run_session(case["prompt"], respond)
        sessions.append({"id": case["id"], **session})
        scores.append(score_case(case, session))
    return {"metrics": summarize(scores), "scores": scores, "sessions": sessions}


def scripted_control() -> dict:
    """Gold-script mechanics control; never label this learned performance."""
    scores, sessions = [], []
    for case in development_cases():
        replies = iter([*case["calls"], case["expected"]])
        session = run_session(
            case["prompt"], lambda _, replies=replies: Reply(dumps(next(replies)))
        )
        sessions.append({"id": case["id"], **session})
        scores.append(score_case(case, session))
    return {
        "label": "scripted gold control, not a model",
        "metrics": summarize(scores),
        "scores": scores,
        "sessions": sessions,
    }
