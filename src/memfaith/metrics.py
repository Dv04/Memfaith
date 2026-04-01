"""Aggregation utilities for MemFaith logs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import csv
import json


def load_experiment_log(path: str) -> List[Dict]:
    records: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def aggregate_records(records: Iterable[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, int, str], Dict[str, List[float]]] = defaultdict(
        lambda: {
            "ccs_values": [],
            "full_accuracy": [],
            "ablation_count": [],
            "flip_count": [],
            "example_count": [],
        }
    )

    for record in records:
        key = (record["dataset"], int(record["k"]), record.get("backend", "unknown"))
        bucket = grouped[key]
        bucket["example_count"].append(1.0)
        bucket["full_accuracy"].append(float(record["full_context"]["is_correct"]))
        bucket["ablation_count"].append(float(len(record.get("ablations") or [])))
        bucket["flip_count"].append(
            float(sum(int(ablation["comparison_to_full"]["flipped"]) for ablation in record.get("ablations") or []))
        )
        if record.get("ccs_example") is not None:
            bucket["ccs_values"].append(float(record["ccs_example"]))

    summary_rows: List[Dict] = []
    for (dataset, k, backend), bucket in sorted(grouped.items()):
        total_examples = int(sum(bucket["example_count"]))
        total_ablations = int(sum(bucket["ablation_count"]))
        total_flips = int(sum(bucket["flip_count"]))
        summary_rows.append(
            {
                "dataset": dataset,
                "k": k,
                "backend": backend,
                "num_examples": total_examples,
                "avg_ccs": round(sum(bucket["ccs_values"]) / len(bucket["ccs_values"]), 6)
                if bucket["ccs_values"]
                else "",
                "full_accuracy": round(sum(bucket["full_accuracy"]) / len(bucket["full_accuracy"]), 6),
                "total_ablations": total_ablations,
                "flip_rate_over_ablations": round(total_flips / total_ablations, 6) if total_ablations else "",
            }
        )
    return summary_rows


def write_summary_csv(summary_rows: List[Dict], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "k",
        "backend",
        "num_examples",
        "avg_ccs",
        "full_accuracy",
        "total_ablations",
        "flip_rate_over_ablations",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
