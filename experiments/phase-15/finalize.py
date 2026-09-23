"""Apply recorded sample-review quarantines and remove final-chunk near copies."""

import argparse
from pathlib import Path

from latos.data2.curation import finalize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quarantine", type=Path, default=Path("configs/data2/output-quarantine-v1.json")
    )
    args = parser.parse_args()
    finalize(args.corpus, args.quarantine, args.output)
