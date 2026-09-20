"""Cache, streaming, history and real terminal acceptance with independent oracles."""

import copy
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import trainers

from latos.chat import format_chat
from latos.inference import KVCache, stream_generate
from latos.inference.stream import TextStream
from latos.inference.terminal import ChatSession, terminal_text
from latos.model import ModelConfig, create_model
from latos.model.sampling import generate
from latos.model.storage import save_model
from latos.tokenization import LatoTokenizer
from latos.tokenization.core import SPECIAL_TOKENS, new_engine


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def codec():
    from tokenizers.pre_tokenizers import ByteLevel

    engine = new_engine()
    engine.train_from_iterator(
        ["Original local inference test text."],
        trainers.BpeTrainer(
            vocab_size=260,
            initial_alphabet=ByteLevel.alphabet(),
            special_tokens=list(SPECIAL_TOKENS),
            show_progress=False,
        ),
    )
    return LatoTokenizer(engine)


@pytest.fixture
def model():
    return create_model(ModelConfig(1, 260, 512, 32, 4, 2, 64, 10000.0, 1e-5, "0" * 64), 73).eval()


BACKENDS = [
    "cpu",
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="MPS hardware unavailable"
        ),
    ),
]


@pytest.mark.parametrize("device", BACKENDS)
@pytest.mark.parametrize("chunks", [(1, 1, 2, 7, 21), (9, 8, 15), (128, 128, 255, 1)])
def test_cache_all_positions_chunks_causality_reset(model, device, chunks):
    model.to(device)
    ids = (torch.arange(sum(chunks) * 2).reshape(2, -1) % 256 + 4).to(device)
    tolerance = 1e-4 if device == "mps" else 1e-5
    cache = KVCache(model)
    with torch.inference_mode():
        expected = model(ids).logits
        parts, offset = [], 0
        for length in chunks:
            parts.append(cache.append(ids[:, offset : offset + length]))
            offset += length
        torch.testing.assert_close(torch.cat(parts, 1), expected, atol=tolerance, rtol=tolerance)
        assert cache.length == sum(chunks)
        assert cache.nbytes == 2 * 2 * 2 * sum(chunks) * 32 * 4
        changed = ids.clone()
        changed[:, -1] = 7
        cache.reset()
        changed_logits = cache.append(changed)
        torch.testing.assert_close(
            changed_logits[:, :-1], expected[:, :-1], atol=tolerance, rtol=tolerance
        )
        cache.reset()
        torch.testing.assert_close(
            cache.append(ids[:, :3]), model(ids[:, :3]).logits, atol=tolerance, rtol=tolerance
        )
    assert cache.length == 3
    assert set(model.state_dict()) == set(create_model(model.config).state_dict())


def test_cache_errors_are_transactional_and_model_changes_rejected(model, monkeypatch):
    cache = KVCache(model)
    ids = torch.tensor([[1, 4, 5]])
    cache.append(ids)
    saved = [(k.clone(), v.clone()) for k, v in cache._layers]
    for invalid in (
        torch.tensor([[260]]),
        torch.ones(1, 510, dtype=torch.long),
        ids.float(),
        ids.expand(2, -1),
    ):
        with pytest.raises(ValueError):
            cache.append(invalid)
        assert cache.length == 3
    original = model.blocks[1].ffn.forward

    def broken(*args):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(model.blocks[1].ffn, "forward", broken)
    with pytest.raises(RuntimeError):
        cache.append(ids)
    assert cache.length == 3
    for before, after in zip(saved, cache._layers, strict=True):
        for a, b in zip(before, after, strict=True):
            assert torch.equal(a, b)
    monkeypatch.setattr(model.blocks[1].ffn, "forward", original)
    model.train()
    with pytest.raises(ValueError, match="eval"):
        cache.append(ids)
    model.eval()
    with torch.no_grad():
        model.embedding.weight.add_(0.01)
    with pytest.raises(ValueError, match="changed"):
        cache.append(ids)
    cache.reset()
    assert cache.append(ids).shape == (1, 3, 260)


@pytest.mark.parametrize("device", BACKENDS)
def test_stream_cached_uncached_and_old_sampler(model, codec, device):
    model.to(device)
    for temperature in (0.0, 0.8):
        settings = dict(max_new_tokens=16, temperature=temperature, top_k=12, seed=73)
        old = generate(model, [1, 4, 7, 10], **settings)
        rng = torch.get_rng_state().clone()
        outputs = []
        for cached in (False, True):
            events = list(stream_generate(model, codec, [1, 4, 7, 10], cached=cached, **settings))
            assert events[-1]["event"] == "done"
            assert "".join(e["text"] for e in events[:-1]) == events[-1]["text"]
            assert events[-1]["text"] == codec.decode(events[-1]["generated_ids"])
            outputs.append(events[-1]["generated_ids"])
        assert outputs[0] == outputs[1] == old["generated_ids"]
        assert torch.equal(rng, torch.get_rng_state())


def test_unicode_stream_matches_batch_with_invalid_fragments(codec):
    sequences = [codec.encode(s) for s in ["Café 🌱 中文 é", "a�b", "<|eos|>", "\x1b[2J"]]
    sequences += [ids[:end] for ids in sequences[:1] for end in range(len(ids))]
    rng = random.Random(93)
    sequences += [[rng.randrange(260) for _ in range(64)] for _ in range(100)]
    for ids in sequences:
        stream = TextStream(codec)
        chunks = [stream.push(i) for i in ids]
        chunks.append(stream.finish())
        assert "".join(chunks) == codec.decode(ids)
    assert terminal_text("a\x1b[2J\x9bb\n") == "a\\x1b[2J\\x9bb\n"


def test_stream_stop_cancel_close_and_no_inference_scope_leak(model, codec, monkeypatch):
    monkeypatch.setattr("latos.inference.stream.sample_next", lambda *a: 2)
    done = list(stream_generate(model, codec, [1, 4]))[-1]
    assert done["generated_ids"] == [2] and done["stop_reason"] == "eos"
    for prompt, budget, reason in [
        ([1, 2], 2, "eos"),
        ([1], 0, "max_new_tokens"),
        ([1] * 512, 1, "context_limit"),
    ]:
        result = list(stream_generate(model, codec, prompt, max_new_tokens=budget))[-1]
        assert result["generated_ids"] == [] and result["stop_reason"] == reason
    result = list(stream_generate(model, codec, [1], cancelled=lambda: True))[-1]
    assert result["stop_reason"] == "cancelled" and result["first_token_ms"] is None
    monkeypatch.setattr("latos.inference.stream.sample_next", lambda *a: 200)
    iterator = stream_generate(model, codec, [1])
    next(iterator)
    assert not torch.is_inference_mode_enabled()
    iterator.close()
    assert not torch.is_inference_mode_enabled() and not model.training
    count = 0

    def cancel():
        nonlocal count
        count += 1
        return count > 2

    events = list(stream_generate(model, codec, [1], cancelled=cancel))
    assert events[-1]["stop_reason"] == "cancelled"
    assert len(events[-1]["generated_ids"]) == 2
    assert "".join(e["text"] for e in events[:-1]) == codec.decode([200, 200])

    def interrupt(*a):
        raise KeyboardInterrupt

    monkeypatch.setattr("latos.inference.stream.sample_next", interrupt)
    with pytest.raises(KeyboardInterrupt):
        list(stream_generate(model, codec, [1]))


@pytest.mark.parametrize(
    "settings",
    [
        {"max_new_tokens": -1},
        {"seed": -1},
        {"temperature": float("nan")},
        {"top_k": 261},
        {"cached": 1},
    ],
)
def test_invalid_stream_settings(model, codec, settings):
    with pytest.raises(ValueError):
        list(stream_generate(model, codec, [1], **settings))


def test_multiturn_exact_formatter_overflow_reset_rollback(model, codec, monkeypatch):
    session = ChatSession(model, codec, system="Be brief.", max_new_tokens=8, context_limit=128)

    def answer(text):
        values = iter(codec.encode(text) + [2])
        monkeypatch.setattr("latos.inference.stream.sample_next", lambda *a: next(values))

    answer("Yes.")
    assert list(session.reply("Hello"))[-1]["history_committed"]
    answer("No.")
    prompts = []
    original = model.embedding.forward

    def capture(ids):
        prompts.append(ids.tolist()[0])
        return original(ids)

    monkeypatch.setattr(model.embedding, "forward", capture)
    expected, _ = format_chat(
        codec,
        session.messages + [{"role": "user", "content": "Again?"}],
        max_length=128,
        generation_prompt=True,
    )
    assert list(session.reply("Again?"))[-1]["history_committed"]
    assert prompts[0] == list(expected)
    previous = copy.deepcopy(session.messages)
    with pytest.raises(ValueError, match="context limit"):
        list(session.reply("x" * 120))
    assert session.messages == previous
    assert not list(session.reply("Hi", cancelled=lambda: True))[-1]["history_committed"]
    assert session.messages == previous
    session.reset()
    assert session.messages == [{"role": "system", "content": "Be brief."}]
    monkeypatch.setattr("latos.inference.stream.sample_next", lambda *a: 2)
    assert not list(session.reply("Hi"))[-1]["history_committed"]
    monkeypatch.setattr("latos.inference.stream.sample_next", lambda *a: codec.encode("a")[0])
    assert not list(session.reply("Hi"))[-1]["history_committed"]
    assert len(session.messages) == 1


@pytest.fixture
def cli_artifacts(tmp_path):
    from latos.data.acquire import acquire
    from latos.data.prepare import prepare
    from latos.tokenization.training import train

    root = Path(__file__).resolve().parents[1]
    manifest = root / "data/fixtures/tiny/manifest.json"
    acquire(manifest, tmp_path / "raw")
    prepare(manifest, tmp_path / "raw", tmp_path / "corpus")
    meta = train(
        manifest, tmp_path / "corpus", root / "configs/tokenizer/debug.json", tmp_path / "codec"
    )
    config = ModelConfig(
        1, meta["vocab_size"], 128, 16, 2, 1, 32, 10000.0, 1e-5, meta["tokenizer_sha256"]
    )
    save_model(create_model(config), tmp_path / "model")
    return [
        sys.executable,
        "-m",
        "latos",
        "chat",
        "--model-dir",
        str(tmp_path / "model"),
        "--tokenizer-dir",
        str(tmp_path / "codec"),
        "--max-new-tokens",
        "4",
    ]


def test_real_cli_json_cached_uncached_and_errors(cli_artifacts):
    outputs = []
    for extra in ([], ["--no-cache"]):
        run = subprocess.run(
            cli_artifacts + ["--prompt", "Hello", "--json", *extra], capture_output=True, text=True
        )
        assert run.returncode == 0, run.stderr
        events = [json.loads(line) for line in run.stdout.splitlines()]
        assert events[0]["event"] == "model" and "not established" in events[0]["quality_note"]
        assert events[-1]["event"] == "done"
        assert "".join(e["text"] for e in events[1:-1]) == events[-1]["text"]
        outputs.append(events[-1]["generated_ids"])
    assert outputs[0] == outputs[1]
    for extra in (
        ["--prompt", "x" * 300],
        ["--json"],
        ["--prompt", "Hi", "--temperature", "nan"],
        ["--prompt", "Hi", "--context-limit", "129"],
    ):
        run = subprocess.run(cli_artifacts + extra, capture_output=True, text=True)
        assert run.returncode == 1
        assert json.loads(run.stderr.splitlines()[-1])["event"] == "error"


def test_real_cli_interactive_reset_eof_and_rejection(cli_artifacts):
    run = subprocess.run(
        cli_artifacts, input="x" * 300 + "\n/reset\nHello\n/exit\n", capture_output=True, text=True
    )
    assert run.returncode == 0
    assert "context limit" in run.stderr and "History cleared" in run.stdout
    run = subprocess.run(cli_artifacts, input="", capture_output=True, text=True)
    assert run.returncode == 0 and "You>" in run.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process SIGINT exercise")
def test_real_cli_sigint_finishes_stream_transactionally(cli_artifacts):
    import signal

    # A long enough budget that the signal follows the first streamed token.
    command = cli_artifacts + ["--prompt", "Hello", "--max-new-tokens", "64", "--json"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    events = []
    try:
        for line in process.stdout:
            event = json.loads(line)
            events.append(event)
            if event["event"] == "token":
                process.send_signal(signal.SIGINT)
                break
        rest, errors = process.communicate(timeout=20)
        events += [json.loads(line) for line in rest.splitlines()]
        assert process.returncode == 0, errors
        assert events[-1]["stop_reason"] == "cancelled"
        assert not events[-1]["history_committed"]
        assert "".join(e["text"] for e in events[1:-1]) == events[-1]["text"]
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_cancellation_at_eos_keeps_history_unchanged(model, codec, monkeypatch):
    tokens = iter(codec.encode("OK") + [2])
    monkeypatch.setattr("latos.inference.stream.sample_next", lambda *args: next(tokens))
    session = ChatSession(model, codec, max_new_tokens=8)
    cancelled = False
    events = []

    def is_cancelled():
        return cancelled

    for event in session.reply("Hello", cancelled=is_cancelled):
        events.append(event)
        if event.get("token_id") == 2:
            cancelled = True
    assert events[-1]["stop_reason"] == "cancelled"
    assert not events[-1]["history_committed"] and not session.messages
    assert "".join(e["text"] for e in events[:-1]) == "OK"


def test_stream_exact_context_boundary_and_missing_artifact(model, codec, monkeypatch, tmp_path):
    monkeypatch.setattr("latos.inference.stream.sample_next", lambda *args: 4)
    events = list(stream_generate(model, codec, [1] * 511, max_new_tokens=1))
    assert events[-1]["stop_reason"] == "context_limit"
    assert events[-1]["generated_ids"] == [4]
    with pytest.raises(ValueError, match="no silent truncation"):
        list(stream_generate(model, codec, [1] * 513))
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "latos",
            "chat",
            "--model-dir",
            str(tmp_path / "missing"),
            "--tokenizer-dir",
            str(tmp_path / "missing"),
            "--prompt",
            "Hello",
        ],
        text=True,
        capture_output=True,
    )
    assert run.returncode == 1 and json.loads(run.stderr)["event"] == "error"


def test_cancellation_on_final_unicode_flush(model, codec, monkeypatch):
    tokens = iter([codec.encode("🌱")[0], 2])
    monkeypatch.setattr("latos.inference.stream.sample_next", lambda *args: next(tokens))
    session = ChatSession(model, codec, max_new_tokens=8)
    cancelled = False

    def is_cancelled():
        return cancelled

    events = []
    for event in session.reply("Hello", cancelled=is_cancelled):
        events.append(event)
        if event["event"] == "text":
            cancelled = True
    assert cancelled and events[-1]["text"] == "�"
    assert events[-1]["stop_reason"] == "cancelled"
    assert not session.messages and not events[-1]["history_committed"]
