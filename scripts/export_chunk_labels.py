"""Export chunk-level causal labels from a MemFaith experiment log."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memfaith import export_chunk_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export chunk-level labels from MemFaith logs.")
    parser.add_argument("--log-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = export_chunk_labels(args.log_path, args.output_path)
    print(f"Wrote {len(rows)} chunk labels to {Path(args.output_path)}")


if __name__ == "__main__":
    main()
