"""Evaluation arithmetic, evidence and gates; all model tests use synthetic fixtures."""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_inference import BACKENDS, codec, model, threads
from test_tokenization import prepared

from latos.evaluation.access import evaluation_access
from latos.evaluation.behavior import generate_case, instruction_correct, observations
from latos.evaluation.compare import gate_metrics, paired_interval, verified_run
from latos.evaluation.inputs import external_cases, read_json, write_json
from latos.evaluation.runner import evaluate_mc, run
from latos.evaluation.scoring import (
    Sequence,
    accuracy,
    byte_spans,
    continuation,
    legacy_windows,
    lm_summary,
    matched_windows,
    score_sequences,
)
from latos.model import ModelConfig, create_model
from latos.model.storage import file_hash, save_model

__all__ = ["codec", "model", "threads", "prepared"]
ROOT = Path(__file__).resolve().parents[1]


def test_independent_preceding_logit_alignment_and_mask():
    class TableModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(5, 1)
            self.config = SimpleNamespace(vocab_size=5, context_length=8)

        def forward(self, ids):
            # Each timestep has a different probability distribution.
            logits = torch.arange(ids.shape[1] * 5, dtype=torch.float32).reshape(-1, 5) % 7
            return SimpleNamespace(logits=logits.expand(ids.shape[0], -1, -1))

    model = TableModel().eval()
    row = Sequence((1, 4, 2, 3), (False, True, False, True))
    result = score_sequences(model, [row], context=8)[0]
    # Target 4 gets row zero, target 3 gets row two. Neither target 2 nor final row score.
    oracle = math.log(sum(math.exp(x) for x in [0, 1, 2, 3, 4])) - 4
    oracle += math.log(sum(math.exp(x) for x in [3, 4, 5, 6, 0])) - 6
    assert result["targets"] == 2
    assert result["nll"] == pytest.approx(oracle, abs=1e-12)


@pytest.mark.parametrize("device", BACKENDS)
def test_batch_padding_repeat_and_readonly(model, device):
    model.to(device)
    rows = [Sequence((1, 4, 8, 10), (False, False, True, True)), Sequence((1, 6), (False, True))]
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    alone = score_sequences(model, rows, batch_size=1)
    batched = score_sequences(model, rows, batch_size=2)
    repeated = score_sequences(model, rows, batch_size=2)
    assert repeated == batched
    for a, b in zip(alone, batched, strict=True):
        assert a["targets"] == b["targets"]
        assert a["nll"] == pytest.approx(b["nll"], abs=1e-5, rel=1e-6)
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize(
    "row",
    [
        Sequence((1,), (False,)),
        Sequence((1, 4), (True, True)),
        Sequence((1, 4), (False, False)),
        Sequence((1, 260), (False, True)),
        Sequence((1, True), (False, True)),
        Sequence((1, 4), (False, 1)),
        Sequence((1, 4), (False,)),
    ],
)
def test_bad_masks_and_ids(model, row):
    with pytest.raises(ValueError):
        score_sequences(model, [row])


def test_empty_nonfinite_training_mode_and_capacity(model, monkeypatch):
    row = Sequence((1, 4), (False, True))
    for rows, options in [([], {}), ([row], {"batch_size": 0}), ([row], {"context": 513})]:
        with pytest.raises(ValueError):
            score_sequences(model, rows, **options)
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        score_sequences(model, [row])
    model.eval()
    monkeypatch.setattr(
        model, "forward", lambda ids: SimpleNamespace(logits=torch.full((1, 2, 260), float("nan")))
    )
    with pytest.raises(ValueError, match="Nonfinite"):
        score_sequences(model, [row])


def test_legacy_transitions_and_matched_unicode_bytes(codec):
    text = " a café🙂" * 70
    legacy = legacy_windows(codec, text, 16)
    ids = codec.encode(text, add_bos=True, add_eos=True)
    pairs = [(a, b) for row in legacy for a, b in zip(row.ids, row.ids[1:], strict=False)]
    assert pairs == list(zip(ids, ids[1:], strict=False))
    spans = list(byte_spans(text))
    assert "".join(spans) == text
    assert all(0 < len(s.encode()) <= 192 for s in spans)
    matched = matched_windows(codec, text)
    assert "".join(codec.decode(list(row.ids)) for row in matched) == text
    assert all(row.ids[0] == 1 and 2 not in row.ids for row in matched)
    assert sum(sum(row.mask) for row in matched) == len(text.encode())


def test_continuation_has_no_prompt_or_eos_targets(codec):
    prefix, answer = "Question: What color?\nAnswer:", "blue sky"
    row = continuation(codec, prefix, answer)
    start = len(codec.encode(prefix, add_bos=True))
    assert row.ids == tuple(codec.encode(prefix + " " + answer, add_bos=True))
    assert not any(row.mask[:start]) and all(row.mask[start:]) and row.ids[-1] != 2
    assert codec.decode(list(row.ids[start:])) == " " + answer


def test_cross_boundary_tokenization_rejected():
    codec = SimpleNamespace(encode=lambda text, **kw: [1, len(text)])
    with pytest.raises(ValueError, match="boundary"):
        continuation(codec, "Answer:", "word")


def test_mc_sum_length_bias_ties_and_overflow(model, codec, monkeypatch):
    import latos.evaluation.runner as runner

    cases = [
        {
            "id": "x",
            "question": "Pick?",
            "choices": ["a", "elephant"],
            "gold": 1,
            "content_sha256": "a" * 64,
        }
    ]
    monkeypatch.setattr(
        runner,
        "score_sequences",
        lambda *a, **kw: [{"nll": 2.0, "targets": 2}, {"nll": 3.0, "targets": 9}],
    )
    rows, summary = evaluate_mc(model, codec, cases, batch_size=2, context=256)
    assert rows[0]["prediction"] == 0 and rows[0]["byte_normalized_prediction"] == 1
    assert not rows[0]["correct"] and rows[0]["correct_byte_normalized"]
    assert summary["uniform_chance"] == 0.5
    monkeypatch.setattr(
        runner,
        "score_sequences",
        lambda *a, **kw: [{"nll": 2.0, "targets": 2}, {"nll": 2.0, "targets": 9}],
    )
    assert evaluate_mc(model, codec, cases, batch_size=2, context=256)[0][0]["prediction"] == 0
    rows, summary = evaluate_mc(model, codec, cases, batch_size=2, context=2)
    assert summary["failures"] == 1 and summary["raw"]["total"] == 1
    assert not rows[0]["correct"]


def test_weighted_aggregation_not_mean_of_means():
    rows = [{"nll": 2.0, "targets": 1, "bytes": 2}, {"nll": 12.0, "targets": 3, "bytes": 8}]
    summary = lm_summary(rows)
    assert summary["nats_per_token"] == 3.5
    assert summary["bits_per_byte"] == pytest.approx(14 / (10 * math.log(2)))
    with pytest.raises(ValueError):
        lm_summary([])
    with pytest.raises(ValueError):
        accuracy([{"correct": 1}])


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a":1}', True),
        (' {"a": 1} ', True),
        ('{"a":true}', False),
        ('{"a":1,"a":1}', False),
        ('{"a":1} trailing', False),
        ('{"a":NaN}', False),
        ('{"a":1,"b":2}', False),
        ("[]", False),
    ],
)
def test_strict_json_scoring(text, expected):
    assert instruction_correct(text, {"scorer": "json", "expected": {"a": 1}}) is expected


def test_fixture_gold_and_trivial_controls():
    suite = read_json(ROOT / "configs/evaluation/suite-v1.json")
    assert len(suite["instructions"]) == 96
    assert len(suite["generation"]) == 24
    for case in suite["instructions"]:
        answer = json.dumps(case["expected"]) if case["scorer"] == "json" else case["expected"]
        assert instruction_correct(answer, case)
        assert not instruction_correct("", case)
        assert not instruction_correct(case["prompt"], case)
    assert observations("a b c a b c", "max_new_tokens")["repeated_trigram_fraction"] == 0.25


@pytest.mark.parametrize("device", BACKENDS)
def test_generation_repeats_and_rejects_overflow(model, codec, device):
    model.to(device)
    case = {
        "id": "x",
        "category": "qa",
        "family": "fixture",
        "prompt": "Write x.",
        "scorer": "exact",
        "expected": "x",
    }
    a = generate_case(model, codec, case, instruction=True, budget=3)
    b = generate_case(model, codec, case, instruction=True, budget=3)
    assert a["generated_ids"] == b["generated_ids"]
    assert a["text"] == b["text"]
    assert (
        generate_case(model, codec, case, instruction=True, context=40, budget=20)["status"]
        == "context_overflow"
    )


def synthetic_summary():
    return {
        "matched": {"bits_per_byte": 2.0, "documents": {"a": {"bits_per_byte": 2.0}}},
        "external": {
            name: {
                "raw": {"accuracy": 0.4, "wilson95": [0.36, 0.44]},
                "byte_normalized": {"accuracy": 0.4},
                "uniform_chance": 0.25,
            }
            for name in ["ARC-Easy", "ARC-Challenge"]
        },
        "generation": {"empty": 0.0, "repeated_trigram_fraction": 0.05},
        "instructions": {"overall": {"accuracy": 0.0}, "categories": {"qa": {"accuracy": 0.0}}},
    }


def test_controlled_improvement_and_regression_gates():
    protocol = read_json(ROOT / "configs/evaluation/protocol-v1.json")
    before, after = synthetic_summary(), synthetic_summary()
    after["matched"]["bits_per_byte"] = 1.7
    after["external"]["ARC-Easy"]["raw"]["accuracy"] = 0.5
    assert gate_metrics(before, after, protocol, "base")["passed"]
    after["matched"]["documents"]["a"]["bits_per_byte"] = 2.1
    assert not gate_metrics(before, after, protocol, "base")["passed"]
    after = synthetic_summary()
    after["instructions"]["overall"]["accuracy"] = 0.75
    after["instructions"]["categories"]["qa"]["accuracy"] = 0.75
    assert gate_metrics(before, after, protocol, "assistant", historical=before)["passed"]
    after["external"]["ARC-Challenge"]["byte_normalized"]["accuracy"] = 0.30
    assert not gate_metrics(before, after, protocol, "assistant", historical=before)["passed"]


def test_paired_uncertainty_identity_and_repeatability():
    a = [{"id": str(i), "correct": False, "family": str(i // 2)} for i in range(10)]
    b = [{**r, "correct": True} for r in a]
    assert paired_interval(a, b)["bootstrap95"] == [1.0, 1.0]
    assert paired_interval(a, b, cluster_key="family")["clusters"] == 5
    assert paired_interval(a, b) == paired_interval(a, b)
    with pytest.raises(ValueError, match="identical"):
        paired_interval(a, b[::-1])


def test_reserved_suite_access_gate(tmp_path):
    protocol = read_json(ROOT / "configs/evaluation/protocol-v1.json")
    assert evaluation_access(protocol, ROOT / "configs/evaluation/suite-v1.json", "a" * 64) == (
        "validation",
        None,
    )
    with pytest.raises(ValueError, match="development"):
        evaluation_access(protocol, ROOT / "configs/evaluation/final-suite-v1.json", "a" * 64)
    selection = tmp_path / "selection.json"
    write_json(selection, {"schema_version": 1})
    with pytest.raises(ValueError, match="declaration"):
        evaluation_access(
            protocol, ROOT / "configs/evaluation/final-suite-v1.json", "a" * 64, selection
        )


def test_json_serialization_rejects_nonfinite_and_duplicate(tmp_path):
    path = tmp_path / "result.json"
    with pytest.raises(ValueError):
        write_json(path, {"loss": float("nan")})
    path.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_json(path)


def test_full_cpu_runner_offline_and_evidence(tmp_path, prepared, monkeypatch):
    import latos.evaluation.runner as runner

    root, metadata = prepared
    config = ModelConfig(
        1, metadata["vocab_size"], 256, 16, 2, 1, 32, 10000.0, 1e-5, metadata["tokenizer_sha256"]
    )
    model_dir = tmp_path / "model"
    save_model(create_model(config).eval(), model_dir)
    suite = {
        "schema_version": 1,
        "version": "latos-eval-1",
        "provenance": "original test only",
        "instructions": [
            {
                "id": "i",
                "category": "qa",
                "family": "tiny",
                "prompt": "Hi",
                "expected": "x",
                "scorer": "exact",
            }
        ],
        "generation": [{"id": "g", "category": "prose", "prompt": "Hello"}],
    }
    suite_path = tmp_path / "suite.json"
    write_json(suite_path, suite)
    external = tmp_path / "external.json"
    write_json(external, {"fixture": True})
    protocol = read_json(ROOT / "configs/evaluation/protocol-v1.json")
    protocol.update(
        suite_sha256=file_hash(suite_path),
        external_sha256=file_hash(external),
        lm_validation_sha256=file_hash(root / "corpus/validation.jsonl"),
        instruction_tokens=2,
        generation_tokens=2,
    )
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, protocol)
    monkeypatch.setattr(
        runner,
        "external_cases",
        lambda *args: {
            "ARC-Easy": [
                {
                    "id": "x",
                    "question": "Pick",
                    "choices": ["a", "b"],
                    "gold": 0,
                    "content_sha256": "0" * 64,
                }
            ]
        },
    )
    # The entire runnable path must succeed without any test payload.
    real_open = Path.open

    def guard(path, *args, **kwargs):
        if path.name == "test.jsonl":
            raise AssertionError("Reserved test access")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    args = SimpleNamespace(
        output=tmp_path / "run",
        protocol=protocol_path,
        suite=suite_path,
        external_manifest=external,
        external_dir=tmp_path,
        data_manifest=ROOT / "data/fixtures/tiny/manifest.json",
        corpus_dir=root / "corpus",
        model_dir=model_dir,
        weights_sha256=file_hash(model_dir / "model.safetensors"),
        tokenizer_dir=root / "artifact",
        device="cpu",
        threads=1,
        batch_size=2,
    )
    result = run(args)
    assert result["matched"]["bits_per_byte"] > 0
    assert verified_run(args.output)[0]["status"] == "complete"
    with pytest.raises(FileExistsError):
        run(args)
    saved = read_json(args.output / "summary.json")
    forged = copy.deepcopy(saved)
    forged["generation"]["empty"] = 0.987
    write_json(args.output / "summary.json", forged)
    inventory = read_json(args.output / "inventory.json")
    inventory["summary.json"] = file_hash(args.output / "summary.json")
    write_json(args.output / "inventory.json", inventory)
    with pytest.raises(ValueError, match="aggregate"):
        verified_run(args.output)
    write_json(args.output / "summary.json", saved)
    inventory["summary.json"] = file_hash(args.output / "summary.json")
    write_json(args.output / "inventory.json", inventory)
    (args.output / "instructions.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        verified_run(args.output)
    args.output = tmp_path / "failed"
    args.weights_sha256 = "0" * 64
    with pytest.raises(ValueError, match="weights checksum"):
        run(args)
    assert (args.output / "failure.json").exists()


def test_external_missing_and_malformed(tmp_path):
    pytest.importorskip("pyarrow")
    manifest = read_json(ROOT / "configs/evaluation/external-v1.json")
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    with pytest.raises(FileNotFoundError):
        external_cases(path, tmp_path)
    manifest["datasets"][0]["split"] = "test"
    write_json(path, manifest)
    with pytest.raises(ValueError, match="validation"):
        external_cases(path, tmp_path)


def test_valid_final_plan_and_failed_gate_refusal(tmp_path):
    suite = tmp_path / "final.json"
    write_json(suite, {"synthetic": "no real reserved data"})
    protocol = {"suite_sha256": "a" * 64, "final_suite_sha256": file_hash(suite)}
    gate, audit, plan = (tmp_path / name for name in ("gate.json", "audit.json", "plan.json"))
    write_json(gate, {"passed": True, "candidate_weights": "b" * 64, "stage": "assistant"})
    write_json(audit, {"accepted_for_final": True, "candidate_weights": "b" * 64})
    selection = {
        "schema_version": 1,
        "phase": 18,
        "candidate_weights": "b" * 64,
        "reference_weights": ["c" * 64],
        "development_comparison": gate.name,
        "development_comparison_sha256": file_hash(gate),
        "contamination_audit": audit.name,
        "contamination_audit_sha256": file_hash(audit),
        "declaration": "locked candidate; no training or selection from final feedback",
    }
    write_json(plan, selection)
    assert evaluation_access(protocol, suite, "b" * 64, plan)[0] == "test"
    assert evaluation_access(protocol, suite, "c" * 64, plan)[0] == "test"
    with pytest.raises(ValueError, match="locked"):
        evaluation_access(protocol, suite, "d" * 64, plan)
    write_json(gate, {"passed": False, "candidate_weights": "b" * 64, "stage": "assistant"})
    selection["development_comparison_sha256"] = file_hash(gate)
    write_json(plan, selection)
    with pytest.raises(ValueError, match="passing"):
        evaluation_access(protocol, suite, "b" * 64, plan)


def test_malformed_suite_and_bounded_overflow(model, codec):
    from latos.evaluation.inputs import validate_suite

    for suite in [
        {},
        {"schema_version": 1, "provenance": "fixture", "instructions": [], "generation": []},
    ]:
        with pytest.raises(ValueError):
            validate_suite(suite)
    case = {
        "id": "x",
        "prompt": "x" * 300,
        "family": "long",
        "category": "qa",
        "expected": "x",
        "scorer": "exact",
    }
    result = generate_case(model, codec, case, instruction=True)
    assert result["status"] == "context_overflow" and result["prompt"] == case["prompt"]


def test_external_parquet_integrity_and_invalid_labels(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    path = tmp_path / "questions.parquet"
    rows = [
        {
            "id": "x",
            "question": "Pick",
            "choices": {"text": ["a", "b"], "label": ["A", "B"]},
            "answerKey": "A",
        }
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    manifest = {
        "schema_version": 1,
        "provenance": "original fixture",
        "license": "MIT",
        "datasets": [
            {
                "name": "fixture",
                "split": "validation",
                "revision": "a" * 40,
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
                "records": 1,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    assert external_cases(manifest_path, tmp_path)["fixture"][0]["gold"] == 0
    rows[0]["choices"]["label"] = ["A", "A"]
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="checksum"):
        external_cases(manifest_path, tmp_path)
    manifest["datasets"][0].update(bytes=path.stat().st_size, sha256=file_hash(path))
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Malformed"):
        external_cases(manifest_path, tmp_path)


def test_pinned_acquisition_cache_refuses_corruption(tmp_path, monkeypatch):
    import io

    from latos.data.manifest import sha256
    from latos.evaluation.inputs import acquire

    response = io.BytesIO(b"fixture")
    response.url = "https://example.org/fixture"
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: response)
    path = tmp_path / "manifest.json"
    write_json(
        path,
        {
            "datasets": [
                {
                    "file": "input.parquet",
                    "bytes": 7,
                    "sha256": sha256(b"fixture"),
                    "url": "https://huggingface.co/datasets/allenai/ai2_arc/resolve/fixture",
                }
            ]
        },
    )
    acquire(path, tmp_path / "data")
    acquire(path, tmp_path / "data")
    (tmp_path / "data/input.parquet").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        acquire(path, tmp_path / "data")


def test_optional_arrow_wheels_for_supported_platforms():
    import tomllib

    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    arrow = next(p for p in lock["package"] if p["name"] == "pyarrow")
    urls = [w["url"] for w in arrow["wheels"]]
    for ending in ("macosx_12_0_arm64.whl", "manylinux_2_28_x86_64.whl", "win_amd64.whl"):
        assert any("cp314-cp314-" in url and url.endswith(ending) for url in urls)


def test_comparison_rejects_substituted_baseline_and_historical_backend(monkeypatch):
    import copy

    import latos.evaluation.compare as module

    common = {
        "protocol_sha256": "a",
        "suite_sha256": "b",
        "external_sha256": "c",
        "data": {},
        "model": {"weights_sha256": "wrong"},
        "environment": dict.fromkeys(
            (
                "device",
                "torch",
                "python",
                "threads",
                "batch_size",
                "platform",
                "packages",
                "deterministic_algorithms",
                "cuda_matmul_tf32",
                "cudnn_tf32",
                "cudnn_benchmark",
            ),
            "same",
        ),
    }
    monkeypatch.setattr(module, "verified_run", lambda path: (common, {}))
    with pytest.raises(ValueError, match="frozen Phase 5"):
        module.compare("ref", "candidate", "base")
    historical = copy.deepcopy(common)
    historical["environment"]["device"] = "different"
    monkeypatch.setattr(
        module, "verified_run", lambda path: (historical if path == "history" else common, {})
    )
    with pytest.raises(ValueError, match="Historical instructions require"):
        module.compare("ref", "candidate", "assistant", "history")
