"""Recount paired evidence, verify attempts and replay saved responses without generation."""

import argparse
import json
import runpy
from pathlib import Path

from latos.chat import format_chat
from latos.model.storage import file_hash
from latos.tokenization import LatoTokenizer
from latos.tool_eval import score_case
from latos.tools import ContextLimit, Reply, run_session


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class EvidenceError(Exception):
    """Evidence failures must escape the production conversation error handler."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def check_inventory(root):
    inventory = read(root / "artifacts.json")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    require(set(inventory) == actual - {"artifacts.json"}, "Inventory is incomplete")
    for name, identity in inventory.items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()), "Unsafe inventory path")
        require(
            path.stat().st_size == identity["bytes"] and file_hash(path) == identity["sha256"],
            f"Artifact mismatch: {name}",
        )


def replay(result, cases, codec, context):
    attempts = iter(result["attempts"])
    for case, session in zip(cases, result["sessions"], strict=True):
        events = iter(session["events"])

        def respond(messages, events=events):
            attempt = next(attempts)
            ids, _ = format_chat(codec, messages, max_length=4096, generation_prompt=True)
            fits = len(ids) + 64 <= context
            require(
                attempt["prompt_tokens"] == len(ids) and attempt["fits"] == fits,
                "Attempt token budget mismatch",
            )
            if not fits:
                require(
                    attempt
                    == {"prompt_tokens": len(ids), "fits": False, "outcome": "context_limit"},
                    "False overflow evidence",
                )
                raise ContextLimit
            event = next(events)
            require(
                attempt
                == {
                    "prompt_tokens": len(ids),
                    "fits": True,
                    "outcome": "reply",
                    "generated_tokens": event["generated_tokens"],
                },
                "Reply attempt mismatch",
            )
            require(0 <= event["generated_tokens"] <= 64, "Generation bound exceeded")
            return Reply(event["raw"], event["generation_stop"], event["generated_tokens"])

        rebuilt = run_session(case["prompt"], respond)
        require({"id": case["id"], **rebuilt} == session, "Saved session replay mismatch")
        require(next(events, None) is None, "Unconsumed events")
    require(next(attempts, None) is None, "Unconsumed attempts")


def verify(root, tokenizer):
    # The unchanged Phase 11 independent scorer uses assertions; do not run under -O.
    require(__debug__, "Verification requires normal Python assertion mode")
    check_inventory(root)
    plan = read(root / "plan.json")
    require(
        plan["contexts"] == [256, 512] and plan["max_new_tokens"] == 64,
        "Unexpected experimental conditions",
    )
    codec = LatoTokenizer.load(tokenizer, expected_sha256=plan["tokenizer_sha256"])
    old_verify = runpy.run_path(str(Path(__file__).parents[1] / "phase-11/verify.py"))["verify"]
    original = read(Path(__file__).parents[1] / "phase-11/plan.json")
    for field in (
        "models",
        "tokenizer_sha256",
        "system_prompt",
        "cases",
        "protocol",
        "temperature",
        "seed",
        "training_updates",
        "threads",
    ):
        require(plan[field] == original[field], f"Changed fixed baseline: {field}")
    for name, expected_hash in plan["source_sha256"].items():
        current = Path(__file__).resolve().parents[2] / name
        require(
            current.resolve().is_relative_to(Path(__file__).resolve().parents[2]),
            "Unsafe source path",
        )
        require(file_hash(current) == expected_hash, f"Current source differs: {name}")
    summary = read(root / "results.json")
    require(summary["total_seconds"] <= plan["max_seconds"], "Time budget exceeded")
    rss = summary["rss_high_water_bytes"]
    require(rss is None or rss <= plan["max_rss_bytes"], "RSS budget exceeded")
    paired, recounted, generated = {}, 0, 0
    data = {}
    for context in (256, 512):
        condition = root / str(context)
        check_inventory(condition)
        current_plan = read(condition / "plan.json")
        require(current_plan == {**plan, "context_limit": context}, "Condition plan mismatch")
        counted = old_verify(condition)
        recounted += counted["sessions_recounted"]
        require(
            summary["conditions"][str(context)] == read(condition / "results.json"),
            "Top summary mismatch",
        )
        data[context] = {}
        for label in ("base", "sft", "dpo"):
            result = read(condition / f"{label}.json")
            replay(result, plan["cases"], codec, context)
            require(
                result["scores"]
                == [
                    score_case(c, s) for c, s in zip(plan["cases"], result["sessions"], strict=True)
                ],
                "Per-case score mismatch",
            )
            require(result["weights_unchanged"] and result["no_gradients"], "Model state changed")
            expected = {
                k: v for k, v in result.items() if k not in ("sessions", "scores", "attempts")
            }
            require(
                summary["conditions"][str(context)][label] == expected, "Condition summary mismatch"
            )
            data[context][label] = result
            generated += result["generated_tokens"]
    previous = read(Path(__file__).parents[1] / "phase-11/samples.json")
    for label in ("base", "sft", "dpo"):
        base, extended = data[256][label], data[512][label]
        require(base["sessions"] == previous[label], "Baseline session drift")
        transitions = sum(
            b["stop_reason"] == "context_limit" and len(e["events"]) >= 2
            for b, e in zip(base["sessions"], extended["sessions"], strict=True)
        )
        attempts = extended["attempts"]
        paired[label] = {
            "formerly_context_limited_now_multiple_replies": transitions,
            "sessions_with_multiple_replies_256": sum(
                len(s["events"]) > 1 for s in base["sessions"]
            ),
            "sessions_with_multiple_replies_512": sum(
                len(s["events"]) > 1 for s in extended["sessions"]
            ),
            "max_prompt_tokens_512": max(a["prompt_tokens"] for a in attempts),
            "generated_replies_with_prompt_over_256": sum(
                a["outcome"] == "reply" and a["prompt_tokens"] > 256 for a in attempts
            ),
            "task_successes_512": extended["metrics"]["task_success"]["numerator"],
        }
    require(generated <= 24576, "Token budget exceeded")
    require(
        summary["training_updates"] == 0 and summary["baseline_sessions_scores_metrics_match"],
        "Invalid baseline or training assertion",
    )
    return {
        "status": "pass",
        "sessions_recounted_including_controls": recounted,
        "learned_sessions_replayed": 96,
        "generated_tokens": generated,
        "paired": paired,
        "H1_more_retry_room": any(
            v["formerly_context_limited_now_multiple_replies"] > 0 for v in paired.values()
        ),
        "H2_at_least_one_successful_normal_task": any(
            v["task_successes_512"] > 0 for v in paired.values()
        ),
        "scope": (
            "Saved evidence replay and independent Phase 11 metric recount; "
            "no model re-execution or general long-context quality claim"
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("artifacts/tokenizers/english-bpe-v1")
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.tokenizer), indent=2))
