"""Boundaries, independent answers, multi-turn accounting and real model integration."""

import json

import pytest
import torch
from test_inference import BACKENDS, codec, model, threads

from latos.chat import format_chat
from latos.cli import main
from latos.inference import stream_generate
from latos.tool_eval import development_cases, score_case, scripted_control, summarize
from latos.tools import (
    ContextLimit,
    ModelResponder,
    ProtocolError,
    Reply,
    dumps,
    execute,
    parse_json,
    run_session,
    validate_envelope,
)

# Imported fixtures retain their local CPU/MPS tiny-only scope.
__all__ = ["codec", "model", "threads"]


def script(*responses):
    items = iter(responses)
    return lambda _: Reply(next(items))


@pytest.mark.parametrize(
    "text",
    [
        "",
        'prose {"final":"7"}',
        "```json\n{}\n```",
        "{} {}",
        '{"tool":"add","tool":"lookup","args":{}}',
        '{"tool":"add","args":{"a":1,"a":2,"b":3}}',
        '{"final":NaN}',
        '{"final":1e999}',
        '{"final":Infinity}',
        '{"final":-Infinity}',
        "[" * 500 + "]" * 500,
        '{"args":{"a":[]}}',
        "x" * 1025,
        '"' + "é" * 600 + '"',
        "\ud800",
    ],
)
def test_reject_ambiguous_or_unbounded_json(text):
    with pytest.raises(ProtocolError):
        parse_json(text)


@pytest.mark.parametrize(
    "value",
    [
        [],
        1,
        None,
        {"final": True},
        {"final": ""},
        {"final": "x" * 129},
        {"final": "a\nb"},
        {"final": "7", "tool": "add"},
        {"tool": 3, "args": {}},
        {"tool": "add", "args": []},
    ],
)
def test_envelope_rejection(value):
    with pytest.raises(ProtocolError):
        validate_envelope(value)


@pytest.mark.parametrize(
    "call",
    [
        {"tool": "shell", "args": {"command": "echo nope"}},
        {"tool": "add", "args": {"a": True, "b": 1}},
        {"tool": "add", "args": {"a": 1.0, "b": 1}},
        {"tool": "add", "args": {"a": "1", "b": 1}},
        {"tool": "add", "args": {"a": 1001, "b": 1}},
        {"tool": "add", "args": {"a": -1001, "b": 1}},
        {"tool": "add", "args": {"a": 1}},
        {"tool": "add", "args": {"a": 1, "b": 2, "extra": 0}},
        {"tool": "lookup", "args": {"key": "../secrets"}},
        {"tool": "lookup", "args": {"key": "oak\n"}},
        {"tool": "lookup", "args": {"key": "Oak"}},
        {"tool": "lookup", "args": {"key": "x" * 17}},
        {"final": "7"},
    ],
)
def test_direct_executor_cannot_bypass_validation(call):
    with pytest.raises(ProtocolError):
        execute(call)


def test_literal_tool_oracles_and_escaped_json():
    assert execute(parse_json('{"tool":"add","args":{"a":1000,"b":1000}}')) == {
        "ok": True,
        "value": 2000,
    }
    assert execute(parse_json('{"tool":"add","args":{"a":-1000,"b":-1000}}')) == {
        "ok": True,
        "value": -2000,
    }
    assert execute({"tool": "lookup", "args": {"key": "elm"}}) == {"ok": True, "value": 11}
    assert execute({"tool": "lookup", "args": {"key": "pine"}}) == {
        "ok": False,
        "error": "not_found",
    }
    assert parse_json(dumps({"final": 'braces [{ and quote "'})) == {
        "final": 'braces [{ and quote "'
    }


def test_multi_turn_feedback_and_context_role_contract():
    history = []
    responses = iter(
        [
            '{"tool":"lookup","args":{"key":"oak"}}',
            '{"tool":"add","args":{"a":7,"b":5}}',
            '{"final":"12"}',
        ]
    )

    def respond(messages):
        history.append(messages)
        return Reply(next(responses))

    result = run_session("Look up oak, then add 5.", respond)
    assert result["stop_reason"] == "final" and result["calls"] == 2
    assert [m["role"] for m in history[-1]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert json.loads(history[1][-1]["content"]) == {"tool_result": {"ok": True, "value": 7}}
    assert json.loads(history[2][-1]["content"]) == {"tool_result": {"ok": True, "value": 12}}


def test_recovery_and_turn_call_bounds():
    result = run_session(
        "Add.", script("bad", '{"tool":"add","args":{"a":2,"b":3}}', '{"final":"5"}')
    )
    assert result["calls"] == 1 and result["stop_reason"] == "final"
    assert result["events"][0]["rejection"] == "invalid_json"
    result = run_session("Add.", lambda _: Reply('{"tool":"add","args":{"a":2,"b":3}}'))
    assert result["calls"] == 2 and result["stop_reason"] == "call_limit"
    assert len(result["events"]) == 3 and result["events"][-1]["result"] is None
    result = run_session("Add.", lambda _: Reply("bad"))
    assert result["calls"] == 0 and result["stop_reason"] == "turn_limit"
    assert len(result["events"]) == 4


@pytest.mark.parametrize("reason", ["max_new_tokens", "context_limit", "cancelled"])
def test_incomplete_calls_never_execute(reason):
    result = run_session("Add.", lambda _: Reply('{"tool":"add","args":{"a":2,"b":3}}', reason))
    assert result["calls"] == 0
    assert result["stop_reason"] == (
        "cancelled" if reason == "cancelled" else "incomplete_generation"
    )


def test_cancellation_before_and_after_generation_and_context_failure():
    def never(_):
        pytest.fail("Must not generate")

    assert run_session("Add.", never, cancelled=lambda: True)["stop_reason"] == "cancelled"
    cancelled = False

    def respond(_):
        nonlocal cancelled
        cancelled = True
        return Reply('{"tool":"add","args":{"a":2,"b":3}}')

    result = run_session("Add.", respond, cancelled=lambda: cancelled)
    assert result["calls"] == 0 and result["stop_reason"] == "cancelled"

    def overflow(_):
        raise ContextLimit

    assert run_session("Add.", overflow)["stop_reason"] == "context_limit"


def test_metrics_do_not_conflate_answer_syntax_arguments_or_missing_failure():
    cases = development_cases()
    case = cases[0]
    answer_only = score_case(case, run_session(case["prompt"], script('{"final":"5"}')))
    assert answer_only["json_valid"] == 1 and answer_only["final_correct"]
    assert not answer_only["task_success"] and answer_only["correct_calls"] == 0
    wrong = score_case(
        case,
        run_session(case["prompt"], script('{"tool":"add","args":{"a":1,"b":4}}', '{"final":"5"}')),
    )
    assert wrong["arguments_valid"] == 1 and wrong["successful_executions"] == 1
    assert wrong["correct_calls"] == 0 and not wrong["task_success"]
    missing = cases[-1]
    skipped = score_case(missing, run_session(missing["prompt"], script('{"error":"not_found"}')))
    assert skipped["final_correct"] and not skipped["failure_handled"]
    empty = summarize([answer_only])
    assert empty["argument_schema"]["rate"] is None
    assert empty["failure_handling"]["denominator"] == 0
    control = scripted_control()["metrics"]
    assert control["cases"] == 16
    assert control["parsing"] == {"numerator": 36, "denominator": 36, "rate": 1.0}
    assert control["correct_tool_and_arguments"]["numerator"] == 20
    assert control["task_success"]["numerator"] == 12
    assert control["failure_handling"]["numerator"] == 4
    assert control["execution_success"] == {"numerator": 16, "denominator": 20, "rate": 0.8}


@pytest.mark.parametrize("device", BACKENDS)
def test_real_model_uses_shared_format_and_preserves_weights_rng(model, codec, device):
    model.to(device)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    messages = [{"role": "user", "content": "Add two and three."}]
    responder = ModelResponder(model, codec, context_limit=512, max_new_tokens=8)
    reply = responder(messages)
    ids, _ = format_chat(codec, messages, max_length=504, generation_prompt=True)
    expected = list(stream_generate(model, codec, list(ids), max_new_tokens=8))[-1]
    assert reply.text == expected["text"] and reply.stop_reason == expected["stop_reason"]
    assert reply.generated_tokens == len(expected["generated_ids"])
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters()) and not model.training
    with pytest.raises(ContextLimit):
        ModelResponder(model, codec, context_limit=16, max_new_tokens=8)(messages)
    with pytest.raises(ValueError, match="limits"):
        ModelResponder(model, codec, context_limit=True)


def test_cli_schema_execution_and_errors(capsys):
    assert main(["tools", "schema"]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == 1
    assert main(["tools", "execute", "--call", '{"tool":"add","args":{"a":4,"b":8}}']) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "value": 12}
    assert main(["tools", "execute", "--call", '{"tool":"lookup","args":{"key":"pine"}}']) == 1
    assert json.loads(capsys.readouterr().out) == {"ok": False, "error": "not_found"}
    assert main(["tools", "execute", "--call", "null"]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "invalid_envelope"


def test_oversized_reply_trace_is_bounded_and_cannot_execute():
    result = run_session("Add.", lambda _: Reply("x" * 100_000))
    assert result["calls"] == 0 and result["stop_reason"] == "turn_limit"
    assert all(e["raw_truncated"] and len(e["raw"]) == 1024 for e in result["events"])
    assert all(e["rejection"] == "response_limit" for e in result["events"])


def test_verifier_recounts_and_rejects_corruption(tmp_path):
    import importlib.util
    from pathlib import Path

    from latos.model.storage import file_hash

    path = Path(__file__).resolve().parents[1] / "experiments/phase-11/verify.py"
    spec = importlib.util.spec_from_file_location("tool_verifier", path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    control = scripted_control()
    plan = {"source_sha256": {}, "cases": development_cases()}
    (tmp_path / "plan.json").write_text(dumps(plan))
    summary = {"scripted_control": control["metrics"]}
    for label in ("base", "sft", "dpo", "scripted"):
        # Synthetic fixture only; not evidence for the real learned artifacts.
        result = {**control, "generated_tokens": 0}
        (tmp_path / f"{label}.json").write_text(dumps(result))
        if label != "scripted":
            summary[label] = {"metrics": result["metrics"]}
    (tmp_path / "results.json").write_text(dumps(summary))
    inventory = {
        p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in tmp_path.iterdir()
    }
    (tmp_path / "artifacts.json").write_text(dumps(inventory))
    assert verifier.verify(tmp_path)["sessions_recounted"] == 64
    (tmp_path / "base.json").write_text("corrupt")
    with pytest.raises(ValueError, match="Artifact mismatch"):
        verifier.verify(tmp_path)
