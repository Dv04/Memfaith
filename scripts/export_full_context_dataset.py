#!/usr/bin/env python3
"""Export full-context dataset (undivided segments) for GPT-2 evaluation.

Produces a CSV with one row per example containing the complete assembled
context text, query, and gold answer.  This is the dataset Dev runs GPT-2
small and GPT-2 XL against to measure baseline accuracy on long contexts.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memfaith import LongContextBuilder, load_prepared_examples
from src.memfaith.prompts import build_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full-context dataset")
    parser.add_argument("--fever-path", default="data/memfaith/counterfactual_fever.jsonl")
    parser.add_argument("--hotpot-path", default="data/memfaith/counterfactual_hotpotqa.jsonl")
    parser.add_argument("--output-path", default="outputs/memfaith/full_context_dataset.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    examples = []
    for path in [args.fever_path, args.hotpot_path]:
        if Path(path).exists():
            examples.extend(load_prepared_examples(path))

    builder = LongContextBuilder(seed=args.seed)
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "example_id", "query", "gold_answer", "task_type",
            "n_segments", "n_evidence", "n_distractors",
            "full_context", "full_prompt",
        ])
        for ex in examples:
            built = builder.build(ex)
            prompt = build_prompt(ex, built.context_text)
            n_ev = len(ex.evidence_segments)
            n_dist = len(ex.distractor_segments)
            writer.writerow([
                ex.dataset, ex.example_id, ex.query, ex.gold_answer,
                ex.task_type, n_ev + n_dist, n_ev, n_dist,
                built.context_text, prompt,
            ])

    print(f"Exported {len(examples)} examples -> {out}")


if __name__ == "__main__":
    main()
