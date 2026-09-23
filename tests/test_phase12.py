"""Context-ablation engineering checks; tiny fixtures, never retained learned models."""

import importlib.util
import time
from pathlib import Path

import pytest
from test_inference import BACKENDS, codec, model, threads

from latos.chat import format_chat
from latos.inference import stream_generate
from latos.tools import ContextLimit, ModelResponder, Reply, run_session

__all__ = ["codec", "model", "threads"]
ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"experiments/phase-12/{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_observed_overflow_and_saved_replay(codec):
    runner, verifier = module("run"), module("verify")

    # Deterministic synthetic codec lengths make the extra-turn effect explicit.
    class SizedCodec:
        def encode(self, text):
            return [4] * (120 if text.startswith("Reply with JSON") else 15)

    sized = SizedCodec()
    for context, expected in ((256, 1), (512, 4)):

        def respond(messages, context=context):
            ids, _ = format_chat(sized, messages, max_length=4096, generation_prompt=True)
            if len(ids) + 64 > context:
                raise ContextLimit
            return Reply("invalid", "eos", 2)

        observed = runner.ObservedResponder(respond, sized, context, time.monotonic() + 10)
        session = {"id": "x", **run_session("Use lookup.", observed)}
        result = {"sessions": [session], "attempts": observed.attempts}
        assert len(session["events"]) == expected
        verifier.replay(result, [{"id": "x", "prompt": "Use lookup."}], sized, context)
        result["attempts"][0]["prompt_tokens"] += 1
        with pytest.raises(verifier.EvidenceError, match="token budget"):
            verifier.replay(result, [{"id": "x", "prompt": "Use lookup."}], sized, context)


def test_budget_exhaustion_is_visible_even_when_session_catches_error(codec):
    runner = module("run")
    observed = runner.ObservedResponder(lambda _: Reply("x"), codec, 512, time.monotonic() - 1)
    result = run_session("Use lookup.", observed)
    assert result["stop_reason"] == "generation_error"
    assert observed.exhausted and not observed.attempts


def test_tiny_model_capacity_not_silently_extended(model, codec):
    with pytest.raises(ValueError, match="generation limits"):
        ModelResponder(model, codec, context_limit=model.config.context_length + 1)


def test_inventory_detects_modified_and_missing_entries(tmp_path):
    runner, verifier = module("run"), module("verify")
    (tmp_path / "evidence.json").write_text("{}")
    runner.write(tmp_path / "artifacts.json", runner.inventory(tmp_path))
    verifier.check_inventory(tmp_path)
    (tmp_path / "evidence.json").write_text('{"changed":true}')
    with pytest.raises(verifier.EvidenceError, match="Artifact mismatch"):
        verifier.check_inventory(tmp_path)
    runner.write(tmp_path / "artifacts.json", {})
    with pytest.raises(verifier.EvidenceError, match="incomplete"):
        verifier.check_inventory(tmp_path)


@pytest.mark.parametrize("device", BACKENDS)
def test_tiny_real_generation_beyond_256_matches_direct_generator(model, codec, device):
    model.to(device)
    messages = [{"role": "system", "content": "a" * 280}, {"role": "user", "content": "lookup"}]
    ids, _ = format_chat(codec, messages, max_length=448, generation_prompt=True)
    assert 256 < len(ids) <= 448
    with pytest.raises(ContextLimit):
        ModelResponder(model, codec, context_limit=256)(messages)
    reply = ModelResponder(model, codec, context_limit=512)(messages)
    events = list(stream_generate(model, codec, list(ids), max_new_tokens=64))
    final = events[-1]
    assert reply == Reply(final["text"], final["stop_reason"], len(final["generated_ids"]))


def test_replay_validation_error_cannot_be_swallowed_as_generation_error():
    verifier = module("verify")

    class TinyCodec:
        def encode(self, text):
            return [4]

    forged = {
        "sessions": [
            {"id": "x", "events": [], "calls": 0, "stop_reason": "generation_error", "final": None}
        ],
        "attempts": [{"prompt_tokens": -1, "fits": True, "outcome": "reply"}],
    }
    with pytest.raises(verifier.EvidenceError, match="token budget"):
        verifier.replay(forged, [{"id": "x", "prompt": "lookup"}], TinyCodec(), 512)
