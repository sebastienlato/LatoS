"""Fixed same-backend inference evidence; no corpus reads, training or checkpoint writes."""

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from latos import __version__
from latos.chat import format_chat
from latos.doctor import available_backends, select_device
from latos.inference import KVCache, stream_generate
from latos.inference.stream import synchronize
from latos.model.storage import bind_tokenizer, file_hash, load_model

ARTIFACTS = {
    "base": (
        "outputs/phase-5-english-pilot/step-00003000/model",
        "f82ed3b21aeacad6d8b3a88c9100cbc83b8f15af1ba9c17a4fa4dccc59b2befa",
    ),
    "sft": (
        "outputs/phase-6-instruction-reviewed/step-00000200/model",
        "62b890214e5544292cc54b725a4e505730fca0d501b3337cdd3f051a622d09ae",
    ),
}
TOKENIZER = "7dce3d0888f6a37d3eadbd077a9ff73170a54a1f6e301e51b2ae6d998142b0ab"
PROMPTS = [
    [{"role": "user", "content": "What can a small model explain about rain?"}],
    [
        {
            "role": "user",
            "content": (
                "A visitor walks through a quiet garden after the rain. The path is wet, "
                "the leaves are green, and a wooden bench stands beneath an old tree. "
                "Describe the scene in one short sentence, "
                "using only details from this description."
            ),
        }
    ],
    [
        {"role": "user", "content": "Remember that the box is blue."},
        {"role": "assistant", "content": "The box is blue."},
        {"role": "user", "content": "What color is the box?"},
    ],
]


def memory(device):
    result = {}
    try:
        import resource

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["process_lifetime_peak_rss_bytes"] = (
            raw if platform.system() == "Darwin" else raw * 1024
        )
    except ImportError:
        result["process_lifetime_peak_rss_bytes"] = None
    if device.type == "mps":
        result.update(
            mps_allocated_bytes=torch.mps.current_allocated_memory(),
            mps_driver_bytes=torch.mps.driver_allocated_memory(),
        )
    if device.type == "cuda":
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    return result


def agreement(model, device):
    tolerance = 1e-4 if device.type == "mps" else 1e-5
    # One batch of synthetic IDs reaches the model's final supported position.
    ids = (torch.arange(512, device=device)[None, :] * 37) % (model.config.vocab_size - 4) + 4
    rows = []
    with torch.inference_mode():
        full = model(ids).logits
        for name, chunks in [
            ("incremental", [1] * 32),
            ("chunked_capacity", [1, 7, 24, 96, 128, 255, 1]),
        ]:
            cache, offset, parts = KVCache(model), 0, []
            for length in chunks:
                parts.append(cache.append(ids[:, offset : offset + length]))
                offset += length
            actual = torch.cat(parts, 1)
            expected = full[:, :offset]
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            diff = (actual - expected).abs()
            rows.append(
                {
                    "case": name,
                    "tokens": offset,
                    "atol": tolerance,
                    "rtol": tolerance,
                    "max_absolute_error": diff.max().item(),
                    "cache_bytes": cache.nbytes,
                }
            )
            cache.reset()
            torch.testing.assert_close(
                cache.append(ids[:, :7]), full[:, :7], atol=tolerance, rtol=tolerance
            )
        changed = ids.clone()
        changed[:, 256:] = 17
        cache = KVCache(model)
        altered = cache.append(changed)
        torch.testing.assert_close(altered[:, :256], full[:, :256], atol=tolerance, rtol=tolerance)
    return rows


def forced_workload(model, device, cached):
    ids = [(i * 37) % (model.config.vocab_size - 4) + 4 for i in range(160)]
    cache = KVCache(model) if cached else None
    times, observations = [], []
    with torch.inference_mode():
        for end in range(128, 160):
            synchronize(device)
            start = time.perf_counter()
            values = ids[:end] if cache is None or cache.length == 0 else ids[end - 1 : end]
            inputs = torch.tensor([values], dtype=torch.long, device=device)
            output = model(inputs).logits if cache is None else cache.append(inputs)
            synchronize(device)
            times.append((time.perf_counter() - start) * 1000)
            assert torch.isfinite(output).all().item()
            observations.append(memory(device))
    return {
        "cached": cached,
        "prefix_tokens": 128,
        "forward_steps": 32,
        "prefill_ms": times[0],
        "subsequent_forward_ms": times[1:],
        "decode_tokens_per_second": 31000 / sum(times[1:]),
        "cache_bytes": cache.nbytes if cache else 0,
        "memory_max_boundary": {
            key: max(o[key] for o in observations if o[key] is not None)
            for key in observations[0]
            if observations[0][key] is not None
        },
    }


def validate(args):
    directory, expected_hash = ARTIFACTS[args.artifact]
    model_dir = Path(directory)
    assert file_hash(model_dir / "model.safetensors") == expected_hash
    assert file_hash(args.tokenizer_dir / "tokenizer.json") == TOKENIZER
    torch.set_num_threads(1)
    device = torch.device(select_device(args.device, available_backends()))
    model = load_model(model_dir, expected_tokenizer_sha256=TOKENIZER).to(device)
    codec = bind_tokenizer(model.config, args.tokenizer_dir)
    checks = agreement(model, device)
    runs = []
    for index, messages in enumerate(PROMPTS):
        prompt, _ = format_chat(codec, messages, max_length=512, generation_prompt=True)
        expected = None
        for cached in (False, True):
            for repeat in range(4):
                events = list(
                    stream_generate(
                        model,
                        codec,
                        list(prompt),
                        max_new_tokens=32,
                        cached=cached,
                        temperature=0,
                        seed=73,
                    )
                )
                result = events[-1]
                assert "".join(e["text"] for e in events[:-1]) == result["text"]
                if expected is None:
                    expected = result["generated_ids"]
                assert result["generated_ids"] == expected
                if repeat:
                    runs.append(
                        {
                            "prompt_index": index,
                            "repeat": repeat,
                            **result,
                            "memory_boundary": memory(device),
                        }
                    )
    forced = []
    for cached in (False, True):
        forced_workload(model, device, cached)
        forced.extend(forced_workload(model, device, cached) for _ in range(3))
    root = Path(__file__).resolve().parents[2]
    source = {
        str(p.relative_to(root)): file_hash(p) for p in sorted((root / "src/latos").rglob("*.py"))
    }
    source.update(
        {
            "experiments/phase-7/validate.py": file_hash(Path(__file__)),
            "uv.lock": file_hash(root / "uv.lock"),
        }
    )
    return {
        "status": "pass",
        "artifact": args.artifact,
        "weights_sha256": expected_hash,
        "tokenizer_sha256": TOKENIZER,
        "device": str(device),
        "threads": 1,
        "package_version": __version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"])),
        "source_hashes": source,
        "agreement": checks,
        "prompts": PROMPTS,
        "runs": runs,
        "forced_workload": forced,
        "forced_summary": {
            str(c): {
                "median_prefill_ms": statistics.median(
                    r["prefill_ms"] for r in forced if r["cached"] == c
                ),
                "median_decode_tokens_per_second": statistics.median(
                    r["decode_tokens_per_second"] for r in forced if r["cached"] == c
                ),
            }
            for c in (False, True)
        },
        "limits": (
            "Engineering only; no quality, CUDA determinism or cross-device equality claim. "
            "No test corpus opened."
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--artifact", choices=tuple(ARTIFACTS), required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument(
        "--tokenizer-dir", type=Path, default=Path("artifacts/tokenizers/english-bpe-v1")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Reserve a new result before expensive work; failures are kept, never overwritten.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            result = validate(args)
        except Exception as exc:
            json.dump({"status": "failed", "error": str(exc)}, stream, indent=2)
            raise
        json.dump(result, stream, indent=2, ensure_ascii=True)
    print(
        json.dumps(
            {
                "status": result["status"],
                "agreement": result["agreement"],
                "forced_summary": result["forced_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
