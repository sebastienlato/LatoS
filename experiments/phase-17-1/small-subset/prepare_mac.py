"""CPU-only reconstruction of the already selected subset; no models/optimization."""

import argparse
import json
import zipfile
from pathlib import Path

from ss_common import read, record, require, write


def prepare(package, destination):
    import numpy as np

    from latos.training.data2 import prepare_probe_dataset

    repo = Path(__file__).resolve().parents[3]
    expected = read(repo / "experiments/phase-16/input-repeat.json")
    destination.mkdir(exist_ok=False)
    result, groups = {}, []
    for split, count in (("train", 4096), ("development", 128)):
        data, selection = prepare_probe_dataset(
            package / "corpus", package / "tokenizer", split, 512, max_documents=count
        )
        require(data.identity == expected[split]["dataset_identity"], "Historical subset changed")
        require(selection["omissions"] == {}, "Unexpected omissions")
        groups.append(set(selection["groups"]))
        offsets = [0]
        with (destination / f"{split}.u32").open("xb") as f:
            for row in data.windows:
                f.write(np.asarray(row, dtype="<u4").tobytes())
                offsets.append(offsets[-1] + len(row))
        np.asarray(offsets, dtype="<u8").tofile(destination / f"{split}.u64")
        write(
            destination / f"{split}.json",
            {
                "identity": data.identity,
                "selection_record_sha256": expected[split]["selection_record_sha256"],
                "documents": len(selection["included"]),
                "group_count": len(selection["groups"]),
            },
        )
        result[split] = data.identity
    require(not groups[0] & groups[1], "Cross-split groups")
    return {
        "optimizer_updates": 0,
        "model_constructions": 0,
        "identities": result,
        "files": {p.name: record(p) for p in destination.iterdir()},
    }


def required_inputs(original_zip):
    with zipfile.ZipFile(original_zip) as z:
        original = z.read("handoff.json")
    import hashlib

    require(
        hashlib.sha256(original).hexdigest()
        == "25c361f9668f55c1624894b33cd8351dc788a94b36d7fb9e05a88438bad87149",
        "Original transfer identity",
    )
    all_files = json.loads(original)["files"]
    names = {
        n
        for n in all_files
        if n.startswith(("source/src/latos/", "inputs/tokenizer/", "evaluation/"))
    }
    names.discard("evaluation/corpus/test.jsonl")  # Reserved payload unnecessary; never touch it.
    names.update(
        "source/" + p
        for p in (
            "configs/training2/selected-model.json",
            "configs/evaluation/protocol-v1.json",
            "configs/evaluation/suite-v1.json",
            "configs/evaluation/external-v1.json",
            "data/manifests/english-books-v1.json",
            "uv.lock",
        )
    )
    return {n: all_files[n] for n in sorted(names)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", required=True, type=Path)
    p.add_argument("--cache", required=True, type=Path)
    args = p.parse_args()
    result = prepare(args.package, args.cache)
    write(Path(__file__).with_name("cache-identities.json"), result)
    write(
        Path(__file__).with_name("required-inputs.json"),
        required_inputs(Path("outputs/phase17-windows-handoff.zip")),
    )
    print(
        json.dumps(
            {"files": len(result["files"]), "optimizer_updates": 0, "model_constructions": 0}
        )
    )
