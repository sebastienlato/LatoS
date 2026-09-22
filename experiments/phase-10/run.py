"""Invoke the predeclared Phase 10 learned experiment; all outputs remain local."""

import argparse
from pathlib import Path

from latos.data.manifest import canonical_json
from latos.preference_run import run_preferences

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    args = parser.parse_args()
    args.base = Path("outputs/phase-6-instruction-reviewed/step-00000200/model")
    # Identity comes from the pre-existing published model card, not a new run.
    args.base_sha256 = "62b890214e5544292cc54b725a4e505730fca0d501b3337cdd3f051a622d09ae"
    args.tokenizer_dir = Path("artifacts/tokenizers/english-bpe-v1")
    args.conversations = Path("data/processed/english-instructions-v1")
    args.manifest = Path("data/manifests/english-books-v1.json")
    args.corpus_dir = Path("data/processed/english-books-v1")
    args.config = Path("configs/training/preference-pilot.json")
    args.threads = 1
    args.protocol = Path(__file__).with_name("PLAN.md")
    args.runner = Path(__file__)
    result = run_preferences(args)
    print(canonical_json({"steps": result["steps"], "pairs_seen": result["pairs_seen"]}).decode())
