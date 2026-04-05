"""Multi-chunk dependency analysis for HotpotQA-style multi-hop tasks.

This module proves Hypothesis 2: multi-hop tasks exhibit *distributed*
causal patterns — i.e., removing any one of several chunks independently
causes an answer flip, demonstrating that the model requires multiple
pieces of evidence simultaneously.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import json
from pathlib import Path


def compute_chunk_flip_vector(record: Dict[str, Any]) -> List[int]:
    """Return a binary vector of length K indicating which chunk removals flipped."""
    return [
        int(abl["comparison_to_full"]["flipped"])
        for abl in (record.get("ablations") or [])
    ]


def compute_multi_chunk_dependency(record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single record for distributed causal necessity.

    Returns a dict with:
        - ``flip_vector``: binary list showing which chunks caused flips
        - ``num_causal_chunks``: how many chunks independently cause a flip
        - ``is_distributed``: True if ≥ 2 chunks are independently causal
        - ``causal_chunk_ids``: list of chunk_ids that caused flips
        - ``gold_chunk_ids``: chunk_ids that contain gold evidence
        - ``causal_gold_overlap``: chunk_ids that are both causal AND gold
    """
    ablations = record.get("ablations") or []
    if not ablations:
        return {
            "example_id": record.get("example_id"),
            "flip_vector": [],
            "num_causal_chunks": 0,
            "is_distributed": False,
            "causal_chunk_ids": [],
            "gold_chunk_ids": [],
            "causal_gold_overlap": [],
        }

    flip_vector = compute_chunk_flip_vector(record)
    causal_chunk_ids = [
        abl["chunk_id"]
        for abl, flipped in zip(ablations, flip_vector)
        if flipped
    ]
    gold_chunk_ids = [
        abl["chunk_id"]
        for abl in ablations
        if abl.get("gold_segment_ids")
    ]
    causal_gold_overlap = sorted(set(causal_chunk_ids) & set(gold_chunk_ids))

    return {
        "example_id": record.get("example_id"),
        "flip_vector": flip_vector,
        "num_causal_chunks": sum(flip_vector),
        "is_distributed": sum(flip_vector) >= 2,
        "causal_chunk_ids": causal_chunk_ids,
        "gold_chunk_ids": gold_chunk_ids,
        "causal_gold_overlap": causal_gold_overlap,
    }


def compute_distributed_causal_score(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metric: fraction of examples with distributed causal necessity.

    Only considers records with K > 0 and task_type == 'qa' (multi-hop).
    """
    eligible = [
        r for r in records
        if r.get("task_type") == "qa" and int(r.get("k", 0)) > 0
    ]
    if not eligible:
        return {
            "total_eligible": 0,
            "distributed_count": 0,
            "distributed_fraction": 0.0,
            "avg_causal_chunks": 0.0,
        }

    dependencies = [compute_multi_chunk_dependency(r) for r in eligible]
    distributed_count = sum(1 for d in dependencies if d["is_distributed"])
    avg_causal = (
        sum(d["num_causal_chunks"] for d in dependencies) / len(dependencies)
        if dependencies
        else 0.0
    )

    return {
        "total_eligible": len(eligible),
        "distributed_count": distributed_count,
        "distributed_fraction": round(distributed_count / len(eligible), 4),
        "avg_causal_chunks": round(avg_causal, 4),
    }


def build_dependency_matrix(
    records: List[Dict[str, Any]],
    *,
    dataset_filter: Optional[str] = None,
    k_filter: Optional[int] = None,
) -> Tuple[List[str], List[List[int]]]:
    """Build an N×K binary matrix of flip patterns.

    Returns (example_ids, matrix) where matrix[i][j] == 1 means
    removing chunk j from example i caused an answer flip.
    """
    filtered = records
    if dataset_filter:
        filtered = [r for r in filtered if r.get("dataset") == dataset_filter]
    if k_filter is not None:
        filtered = [r for r in filtered if int(r.get("k", 0)) == k_filter]
    # Exclude baseline rows
    filtered = [r for r in filtered if int(r.get("k", 0)) > 0]

    example_ids: List[str] = []
    matrix: List[List[int]] = []
    for record in filtered:
        example_ids.append(str(record.get("example_id", "")))
        matrix.append(compute_chunk_flip_vector(record))

    return example_ids, matrix


def summarize_dependency_analysis(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Produce a full dependency summary across all records.

    Groups by (dataset, k) and computes dependency statistics for each group.
    """
    groups: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for r in records:
        k = int(r.get("k", 0))
        if k == 0:
            continue
        groups[(r.get("dataset", "unknown"), k)].append(r)

    summary: Dict[str, Any] = {}
    for (dataset, k), group_records in sorted(groups.items()):
        deps = [compute_multi_chunk_dependency(r) for r in group_records]
        n = len(deps)
        distributed = sum(1 for d in deps if d["is_distributed"])
        avg_causal = sum(d["num_causal_chunks"] for d in deps) / n if n else 0.0
        avg_gold_overlap = (
            sum(len(d["causal_gold_overlap"]) for d in deps) / n if n else 0.0
        )
        summary[f"{dataset}_k{k}"] = {
            "dataset": dataset,
            "k": k,
            "num_examples": n,
            "distributed_count": distributed,
            "distributed_fraction": round(distributed / n, 4) if n else 0.0,
            "avg_causal_chunks": round(avg_causal, 4),
            "avg_causal_gold_overlap": round(avg_gold_overlap, 4),
        }
    return summary
