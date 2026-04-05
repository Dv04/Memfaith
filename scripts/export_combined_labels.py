"""Export combined FEVER + HotpotQA chunk labels for the discriminator handoff."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memfaith import export_chunk_labels
from src.memfaith.metrics import load_experiment_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge FEVER and HotpotQA logs and export combined chunk labels."
    )
    parser.add_argument(
        "--fever-log",
        type=str,
        default="outputs/memfaith/fever_smoke_ccs.jsonl",
    )
    parser.add_argument(
        "--hotpot-log",
        type=str,
        default="outputs/memfaith/hotpot_smoke_ccs.jsonl",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/memfaith/combined_chunk_labels.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Merge logs into a single temporary file
    merged_path = Path(args.output_path).parent / "combined_ccs.jsonl"
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    import json

    all_records = []
    for log_path in [args.fever_log, args.hotpot_log]:
        p = Path(log_path)
        if p.exists():
            all_records.extend(load_experiment_log(str(p)))
            print(f"  Loaded {log_path}: {sum(1 for r in load_experiment_log(str(p)))} records")
        else:
            print(f"  WARNING: {log_path} not found, skipping.")

    with merged_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    # Export chunk labels
    rows = export_chunk_labels(str(merged_path), args.output_path)

    # Print label distribution
    causal_pos = sum(1 for r in rows if int(r["causal_label"]) == 1)
    causal_neg = len(rows) - causal_pos
    datasets = {}
    for r in rows:
        ds = r["dataset"]
        datasets[ds] = datasets.get(ds, 0) + 1

    print(f"\n--- Combined Chunk Label Export ---")
    print(f"Total rows: {len(rows)}")
    print(f"Causal (flipped=1): {causal_pos}  ({100*causal_pos/len(rows):.1f}%)" if rows else "")
    print(f"Non-causal (flipped=0): {causal_neg}  ({100*causal_neg/len(rows):.1f}%)" if rows else "")
    print(f"By dataset: {datasets}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
