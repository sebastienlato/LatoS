"""Strict, side-effect-free tool protocol and bounded local conversation runner."""

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

MAX_BYTES = 1024
MAX_TURNS = 4
MAX_CALLS = 2
CATALOG = MappingProxyType({"oak": 7, "elm": 11, "ash": 13})
PROTOCOL = {
    "version": 1,
    "max_response_bytes": MAX_BYTES,
    "max_json_depth": 2,
    "max_turns": MAX_TURNS,
    "max_calls": MAX_CALLS,
    "envelopes": [{"tool": "name", "args": {}}, {"final": "text"}, {"error": "code"}],
    "tools": {
        "add": {"a": "integer [-1000,1000]", "b": "integer [-1000,1000]"},
        "lookup": {"key": "1-16 lowercase ASCII letters; missing keys return not_found"},
    },
    "catalog": dict(CATALOG),
}
SYSTEM_PROMPT = (
    'Reply with JSON only: {"tool":"add","args":{"a":1,"b":2}} or '
    '{"tool":"lookup","args":{"key":"oak"}}. '
    'After results use {"final":"answer"}; after failure use {"error":"not_found"}. '
    "Use tools. add accepts integers -1000..1000. lookup keys are lowercase letters."
)


class ProtocolError(ValueError):
    """A stable error code; raw untrusted text is never an exception message."""


class ContextLimit(ValueError):
    """The full history plus reserved generation cannot fit."""


@dataclass(frozen=True)
class Reply:
    text: str
    stop_reason: str = "eos"
    generated_tokens: int = 0


def dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("invalid_json")
        result[key] = value
    return result


def _constant(_):
    raise ProtocolError("invalid_json")


def _float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ProtocolError("invalid_json")
    return value


def parse_json(text: str) -> object:
    """Decode one bounded JSON value, rejecting ambiguity and excessive nesting."""
    if not isinstance(text, str) or len(text) > MAX_BYTES:
        raise ProtocolError("response_limit")
    try:
        if len(text.encode("utf-8")) > MAX_BYTES:
            raise ProtocolError("response_limit")
    except UnicodeError as exc:
        raise ProtocolError("invalid_json") from exc
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[":
            depth += 1
            if depth > 2:
                raise ProtocolError("response_limit")
        elif char in "}]":
            depth -= 1
    try:
        return json.loads(
            text, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float
        )
    except (ValueError, RecursionError) as exc:
        raise ProtocolError("invalid_json") from exc


def validate_envelope(value: object) -> dict:
    if not isinstance(value, dict):
        raise ProtocolError("invalid_envelope")
    keys = set(value)
    if keys == {"tool", "args"}:
        if isinstance(value["tool"], str) and isinstance(value["args"], dict):
            return value
    elif keys in ({"final"}, {"error"}):
        text = next(iter(value.values()))
        if isinstance(text, str) and 1 <= len(text) <= 128 and text.isprintable():
            return value
    raise ProtocolError("invalid_envelope")


def validate_call(call: dict) -> None:
    validate_envelope(call)
    if set(call) != {"tool", "args"}:
        raise ProtocolError("invalid_envelope")
    name, args = call["tool"], call["args"]
    if name == "add":
        if set(args) == {"a", "b"} and all(
            type(v) is int and -1000 <= v <= 1000 for v in args.values()
        ):
            return
    elif name == "lookup":
        if (
            set(args) == {"key"}
            and isinstance(args["key"], str)
            and re.fullmatch("[a-z]{1,16}", args["key"])
        ):
            return
    else:
        raise ProtocolError("unknown_tool")
    raise ProtocolError("invalid_arguments")


def execute(call: dict) -> dict:
    """Revalidate even direct API calls. The fixed tools perform no external I/O."""
    validate_call(call)
    args = call["args"]
    if call["tool"] == "add":
        return {"ok": True, "value": args["a"] + args["b"]}
    if args["key"] not in CATALOG:
        return {"ok": False, "error": "not_found"}
    return {"ok": True, "value": CATALOG[args["key"]]}


def run_session(
    prompt: str,
    respond: Callable[[list[dict]], Reply],
    *,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict:
    """At most four generations and two executions; callbacks must be trusted.

    Invalid requests get feedback, but consume a turn. Execution failures consume
    a call. No returned final/error is trusted as a claim of task correctness.
    """
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 512:
        raise ValueError("Task must be nonempty text of at most 512 characters")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    events, calls = [], 0

    def finish(reason, final=None):
        return {"stop_reason": reason, "final": final, "calls": calls, "events": events}

    for _ in range(MAX_TURNS):
        if cancelled():
            return finish("cancelled")
        try:
            reply = respond([dict(message) for message in messages])
        except ContextLimit:
            return finish("context_limit")
        except ValueError, RuntimeError:
            return finish("generation_error")
        if not isinstance(reply, Reply):
            return finish("generation_error")
        if cancelled() or reply.stop_reason == "cancelled":
            return finish("cancelled")
        event = {
            "raw": reply.text[:MAX_BYTES] if isinstance(reply.text, str) else None,
            "raw_truncated": isinstance(reply.text, str) and len(reply.text) > MAX_BYTES,
            "generation_stop": reply.stop_reason,
            "generated_tokens": reply.generated_tokens,
            "json_valid": False,
            "envelope_valid": False,
            "arguments_valid": False,
            "call": None,
            "result": None,
        }
        events.append(event)
        try:
            value = parse_json(reply.text)
            event["json_valid"] = True
            action = validate_envelope(value)
            event["envelope_valid"] = True
            if "tool" in action:
                event["call"] = action
                validate_call(action)
                event["arguments_valid"] = True
        except ProtocolError as exc:
            event["rejection"] = str(exc)
        # Parsing is measured even for truncated output, but it cannot execute.
        if reply.stop_reason != "eos":
            return finish("incomplete_generation")
        if "rejection" in event:
            feedback = {"ok": False, "error": event["rejection"]}
        else:
            if "tool" not in action:
                return finish("final" if "final" in action else "reported_error", action)
            if calls == MAX_CALLS:
                return finish("call_limit")
            if cancelled():
                return finish("cancelled")
            calls += 1
            feedback = execute(action)
            event["result"] = feedback
        # Bounded transcript: invalid/oversized output is not echoed into history.
        content = reply.text if event["envelope_valid"] else "[invalid response]"
        messages.extend(
            [
                {"role": "assistant", "content": content},
                {"role": "user", "content": dumps({"tool_result": feedback})},
            ]
        )
    return finish("turn_limit")


class ModelResponder:
    """Use the unchanged chat serializer and greedy local inference; never truncate."""

    def __init__(self, model, codec, *, context_limit=256, max_new_tokens=64, cancelled=None):
        if (
            type(context_limit) is not int
            or not 4 <= context_limit <= min(4096, model.config.context_length)
            or type(max_new_tokens) is not int
            or not 1 <= max_new_tokens <= context_limit - 2
        ):
            raise ValueError("Invalid tool generation limits")
        self.model, self.codec = model, codec
        self.context_limit, self.max_new_tokens = context_limit, max_new_tokens
        self.cancelled = cancelled

    def __call__(self, messages):
        from latos.chat import format_chat
        from latos.inference import stream_generate

        try:
            ids, _ = format_chat(
                self.codec,
                messages,
                max_length=self.context_limit - self.max_new_tokens,
                generation_prompt=True,
            )
        except ValueError as exc:
            if "exceeds context" in str(exc):
                raise ContextLimit from exc
            raise
        for event in stream_generate(
            self.model,
            self.codec,
            list(ids),
            max_new_tokens=self.max_new_tokens,
            cancelled=self.cancelled,
        ):
            if event["event"] == "done":
                return Reply(event["text"], event["stop_reason"], len(event["generated_ids"]))
        raise RuntimeError("Generation did not complete")
