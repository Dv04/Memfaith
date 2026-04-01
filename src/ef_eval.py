"""Utilities to compute Editability–Faithfulness metrics from experiment logs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

VALID_LABELS = {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}

def load_ef_log(path: Union[str, Path]) -> List[Dict]:
    records: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _get_field(record: Dict, *keys: str):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _normalize_label(value: Optional[str]) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    return value.strip().upper().replace(" ", "_").replace("-", "_")


def _filtered_triplets(records: List[Dict]) -> List[Tuple[str, str, str]]:
    filtered: List[Tuple[str, str, str]] = []
    for record in records:
        pre = _normalize_label(_get_field(record, "answer_pre", "orig_answer"))
        tgt = _normalize_label(
            _get_field(record, "answer_post_target", "post_edit_answer_target")
        )
        ctrl = _normalize_label(
            _get_field(record, "answer_post_control", "post_edit_answer_control")
        )
        if pre not in VALID_LABELS:
            continue
        if tgt is None or ctrl is None:
            continue
        filtered.append((pre, tgt, ctrl))
    return filtered


def compute_flip_rates(records: List[Dict]) -> Dict[str, float]:
    target_changes = []
    control_changes = []
    filtered = _filtered_triplets(records)
    for pre, tgt, ctrl in filtered:
        target_changes.append(float(tgt != pre))
        control_changes.append(float(ctrl != pre))
    flip_tgt = sum(target_changes) / len(target_changes) if target_changes else 0.0
    flip_ctrl = sum(control_changes) / len(control_changes) if control_changes else 0.0
    return {
        "flip_target": flip_tgt,
        "flip_control": flip_ctrl,
        "ef": flip_tgt - flip_ctrl,
        "n_target": len(target_changes),
        "n_control": len(control_changes),
        "n_records": len(records),
        "n_valid": len(filtered),
    }


def bootstrap_ci(values: List[float], num_samples: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    estimates = []
    for _ in range(num_samples):
        sample = [random.choice(values) for _ in values]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    lower_idx = int((alpha / 2) * len(estimates))
    upper_idx = int((1 - alpha / 2) * len(estimates))
    upper_idx = min(upper_idx, len(estimates) - 1)
    return estimates[lower_idx], estimates[upper_idx]


def summarize_ef(log_path: Union[str, Path], n_bootstrap: int = 1000) -> Dict[str, float]:
    records = load_ef_log(log_path)
    flip_stats = compute_flip_rates(records)
    filtered = _filtered_triplets(records)
    flip_target_vals = [float(tgt != pre) for pre, tgt, _ in filtered]
    flip_control_vals = [float(ctrl != pre) for pre, _, ctrl in filtered]
    flip_stats["flip_target_ci"] = bootstrap_ci(flip_target_vals, num_samples=n_bootstrap)
    flip_stats["flip_control_ci"] = bootstrap_ci(flip_control_vals, num_samples=n_bootstrap)
    flip_stats["ef_ci"] = (
        flip_stats["flip_target_ci"][0] - flip_stats["flip_control_ci"][1],
        flip_stats["flip_target_ci"][1] - flip_stats["flip_control_ci"][0],
    )
    return flip_stats


def _pretty_print(stats: Dict[str, float]) -> None:
    def pct(x):
        return f"{100 * x:.1f}%"

    print(
        f"Examples used: {stats.get('n_valid', 0)} "
        f"(filtered from {stats.get('n_records', 0)})"
    )
    print(f"Flip_tgt : {pct(stats['flip_target'])} 95% CI {tuple(map(pct, stats['flip_target_ci']))}")
    print(f"Flip_ctrl: {pct(stats['flip_control'])} 95% CI {tuple(map(pct, stats['flip_control_ci']))}")
    print(f"EF       : {pct(stats['ef'])} 95% approx CI {tuple(map(pct, stats['ef_ci']))}")


def main():
    parser = argparse.ArgumentParser(description="Summarize EF logs.")
    parser.add_argument("--log-path", type=str, required=True, help="Path to EF JSONL log.")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for CI estimation.",
    )
    args = parser.parse_args()
    stats = summarize_ef(args.log_path, n_bootstrap=args.n_bootstrap)
    _pretty_print(stats)


if __name__ == "__main__":
    main()
