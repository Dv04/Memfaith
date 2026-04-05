"""Stratified CCS analysis utilities.

Implements label-based and positional stratification to uncover hidden
behavioral biases in the chunk-ablation pipeline, as specified in the
MemFaith evaluation framework.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stratify_by_label(
    records: List[Dict[str, Any]],
    *,
    dataset_filter: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Slice CCS scores by gold answer label.

    For FEVER: SUPPORTS vs REFUTES vs NOT_ENOUGH_INFO.
    For QA: groups by gold_answer (useful for repeated templates).

    Returns a dict keyed by label with CCS stats.
    """
    groups: Dict[str, List[float]] = defaultdict(list)
    counts: Dict[str, int] = defaultdict(int)

    for record in records:
        if dataset_filter and record.get("dataset") != dataset_filter:
            continue
        k = int(record.get("k", 0))
        if k == 0:
            continue
        label = str(record.get("gold_answer", "unknown")).strip().upper()
        ccs = record.get("ccs_example")
        if ccs is not None:
            groups[label].append(float(ccs))
        counts[label] += 1

    result: Dict[str, Dict[str, Any]] = {}
    for label in sorted(groups.keys()):
        values = groups[label]
        result[label] = {
            "label": label,
            "num_examples": counts[label],
            "avg_ccs": round(_safe_mean(values), 6),
            "min_ccs": round(min(values), 6) if values else None,
            "max_ccs": round(max(values), 6) if values else None,
        }
    return result


def stratify_by_position(
    records: List[Dict[str, Any]],
    *,
    bins: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """Map chunk_i to its relative position in the prompt and compute
    causal importance by position bucket.

    Default splits: first 20%, middle 60%, last 20% (approximated with 3 bins
    as first-third, middle-third, last-third for simplicity).
    """
    position_labels = {0: "first", 1: "middle", 2: "last"} if bins == 3 else None
    flip_counts: Dict[str, List[int]] = defaultdict(list)
    total_counts: Dict[str, int] = defaultdict(int)

    for record in records:
        ablations = record.get("ablations") or []
        n_chunks = len(ablations)
        if n_chunks == 0:
            continue
        for abl in ablations:
            chunk_id = int(abl["chunk_id"])
            # Compute position bucket
            if bins == 3 and n_chunks >= 3:
                if chunk_id < n_chunks / 3:
                    bucket = "first"
                elif chunk_id < 2 * n_chunks / 3:
                    bucket = "middle"
                else:
                    bucket = "last"
            else:
                bucket_idx = min(int(chunk_id * bins / n_chunks), bins - 1)
                bucket = position_labels.get(bucket_idx, f"bin_{bucket_idx}") if position_labels else f"bin_{bucket_idx}"

            flipped = int(abl["comparison_to_full"]["flipped"])
            flip_counts[bucket].append(flipped)
            total_counts[bucket] += 1

    result: Dict[str, Dict[str, Any]] = {}
    for bucket in sorted(flip_counts.keys()):
        flips = flip_counts[bucket]
        result[bucket] = {
            "position": bucket,
            "total_chunks": total_counts[bucket],
            "flip_count": sum(flips),
            "flip_rate": round(_safe_mean(flips), 6),
        }
    return result


def stratify_by_dataset(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Comparative CCS across datasets (FEVER vs HotpotQA)."""
    groups: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    counts: Dict[Tuple[str, int], int] = defaultdict(int)

    for record in records:
        ds = record.get("dataset", "unknown")
        k = int(record.get("k", 0))
        if k == 0:
            continue
        ccs = record.get("ccs_example")
        if ccs is not None:
            groups[(ds, k)].append(float(ccs))
        counts[(ds, k)] += 1

    result: Dict[str, Dict[str, Any]] = {}
    for (ds, k) in sorted(groups.keys()):
        values = groups[(ds, k)]
        key = f"{ds}_k{k}"
        result[key] = {
            "dataset": ds,
            "k": k,
            "num_examples": counts[(ds, k)],
            "avg_ccs": round(_safe_mean(values), 6),
            "min_ccs": round(min(values), 6) if values else None,
            "max_ccs": round(max(values), 6) if values else None,
        }
    return result


def stratify_by_gold_coverage(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Analyze flip rate based on whether the removed chunk contains gold evidence."""
    gold_flips: List[int] = []
    non_gold_flips: List[int] = []

    for record in records:
        for abl in record.get("ablations") or []:
            flipped = int(abl["comparison_to_full"]["flipped"])
            has_gold = bool(abl.get("gold_segment_ids"))
            if has_gold:
                gold_flips.append(flipped)
            else:
                non_gold_flips.append(flipped)

    return {
        "gold_chunks": {
            "total": len(gold_flips),
            "flip_count": sum(gold_flips),
            "flip_rate": round(_safe_mean(gold_flips), 6),
        },
        "non_gold_chunks": {
            "total": len(non_gold_flips),
            "flip_count": sum(non_gold_flips),
            "flip_rate": round(_safe_mean(non_gold_flips), 6),
        },
    }
