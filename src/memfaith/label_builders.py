"""Utilities for converting MemFaith ablation logs into chunk-label datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import csv

from .metrics import load_experiment_log


def export_chunk_labels(log_path: str, output_path: str) -> List[Dict]:
    records = load_experiment_log(log_path)
    rows: List[Dict] = []

    for record in records:
        for ablation in record.get("ablations") or []:
            rows.append(
                {
                    "dataset": record["dataset"],
                    "example_id": record["example_id"],
                    "k": record["k"],
                    "query": record["query"],
                    "gold_answer": record["gold_answer"],
                    "full_prediction": record["full_context"]["prediction"]["raw_text"],
                    "chunk_id": ablation["chunk_id"],
                    "chunk_text": ablation["chunk_text"],
                    "segment_ids": ",".join(str(value) for value in ablation.get("segment_ids") or []),
                    "gold_segment_ids": ",".join(str(value) for value in ablation.get("gold_segment_ids") or []),
                    "ablation_prediction": ablation["prediction"]["raw_text"],
                    "causal_label": int(ablation["comparison_to_full"]["flipped"]),
                    "comparison_method": ablation["comparison_to_full"]["method"],
                    "comparison_score": ablation["comparison_to_full"]["score"],
                }
            )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "example_id",
        "k",
        "query",
        "gold_answer",
        "full_prediction",
        "chunk_id",
        "chunk_text",
        "segment_ids",
        "gold_segment_ids",
        "ablation_prediction",
        "causal_label",
        "comparison_method",
        "comparison_score",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows
