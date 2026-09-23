"""Fit exactly two training-only codecs and select by the predeclared efficiency rule."""

import argparse
import time
from pathlib import Path

from latos.data2.acquire import write_json
from latos.data2.resources import peak_rss_bytes
from latos.data2.tokenizer import analyze, fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    candidates = {"historical8192": Path("artifacts/tokenizers/english-bpe-v1")}
    for vocab in (8192, 16384):
        name = f"new{vocab}"
        candidates[name] = args.output / name
        fit(args.corpus, candidates[name], vocab)
        print("Fitted", name, flush=True)
    dev = {name: analyze(args.corpus, path, "development") for name, path in candidates.items()}

    def total(name):
        return sum(
            dev[name]["stats"][kind]["content_tokens"] for kind in ("pretrain", "instruction")
        )

    gain = 1 - total("new16384") / total("new8192")
    no_regression = all(
        dev["new16384"]["stats"][source]["bytes_per_token"]
        >= dev["new8192"]["stats"][source]["bytes_per_token"]
        for source in ("wiki", "dolly", "oasst")
    )
    selected = (
        "new16384"
        if gain >= 0.08 and no_regression
        else "new8192"
        if total("new8192") < total("historical8192")
        else "historical8192"
    )
    write_json(
        args.output / "selection.json",
        {
            "selected": selected,
            "gain_16384_vs_8192": gain,
            "no_source_regression": no_regression,
            "development": dev,
        },
    )
    write_json(
        args.output / "training-accounting.json",
        analyze(args.corpus, candidates[selected], "train"),
    )
    write_json(
        args.output / "resources.json",
        {
            "seconds": time.monotonic() - start,
            "peak_rss_bytes": peak_rss_bytes(),
        },
    )
    print("Selected", selected, flush=True)


if __name__ == "__main__":
    main()
