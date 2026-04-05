"""Review HotpotQA ablation flips — extract them for qualitative analysis.

Produces a human-readable markdown file listing all flipped ablations
from HotpotQA experiments, sorted by comparison score (most ambiguous first).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memfaith.metrics import load_experiment_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HotpotQA flips for qualitative review.")
    parser.add_argument("--log-path", type=str, default="outputs/memfaith/hotpot_smoke_ccs.jsonl")
    parser.add_argument("--output-path", type=str, default="outputs/memfaith/hotpot_flip_review.md")
    parser.add_argument("--max-flips", type=int, default=100, help="Max number of flips to include.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_experiment_log(args.log_path)

    flips = []
    for record in records:
        if record.get("dataset") != "hotpotqa":
            continue
        k = int(record.get("k", 0))
        if k == 0:
            continue
        for abl in record.get("ablations") or []:
            if abl["comparison_to_full"]["flipped"]:
                flips.append(
                    {
                        "example_id": record["example_id"],
                        "query": record["query"],
                        "gold_answer": record["gold_answer"],
                        "k": k,
                        "full_prediction": record["full_context"]["prediction"]["raw_text"],
                        "chunk_id": abl["chunk_id"],
                        "chunk_text": abl["chunk_text"][:300],
                        "ablated_prediction": abl["prediction"]["raw_text"],
                        "comparison_method": abl["comparison_to_full"]["method"],
                        "comparison_score": abl["comparison_to_full"]["score"],
                        "has_gold": bool(abl.get("gold_segment_ids")),
                    }
                )

    # Sort by score ascending (most ambiguous first)
    flips.sort(key=lambda x: x["comparison_score"])
    flips = flips[: args.max_flips]

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# HotpotQA Flip Review",
        "",
        f"Total flips extracted: **{len(flips)}**",
        "",
        "> **NOTE:** This review is for qualitative validation of the LLM-as-a-judge",
        "> and Token-F1 comparison logic. Verify that flips represent genuine semantic",
        "> divergence, not formatting artifacts.",
        "",
    ]

    for i, flip in enumerate(flips, 1):
        gold_marker = " 🟡 **GOLD CHUNK**" if flip["has_gold"] else ""
        lines.extend(
            [
                f"## Flip {i}: `{flip['example_id']}` (K={flip['k']}, Chunk {flip['chunk_id']}){gold_marker}",
                "",
                f"**Query:** {flip['query']}",
                "",
                f"**Gold Answer:** {flip['gold_answer']}",
                "",
                f"**Full-Context Prediction:** {flip['full_prediction']}",
                "",
                f"**Ablated Prediction (chunk {flip['chunk_id']} removed):** {flip['ablated_prediction']}",
                "",
                f"**Comparison Method:** `{flip['comparison_method']}` | **Score:** {flip['comparison_score']:.4f}",
                "",
                "**Removed Chunk Preview:**",
                "```",
                flip["chunk_text"],
                "```",
                "",
                "---",
                "",
            ]
        )

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(flips)} flip reviews to {out}")


if __name__ == "__main__":
    main()
