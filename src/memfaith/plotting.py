"""Publication-ready plots for MemFaith CCS experiments.

All functions produce self-contained figures saved to disk.  Designed
to be called from ``scripts/`` runners or the ``run_eval.sh`` script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for scripts
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------
_STYLE = {
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 7,
}

_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CCS Degradation Curve  (CCS vs K)
# ---------------------------------------------------------------------------
def plot_ccs_degradation_curve(
    summary_rows: List[Dict[str, Any]],
    out_path: str,
    *,
    title: str = "CCS Degradation Curve",
) -> None:
    """Plot CCS(K) degradation curve — the primary output of the project.

    ``summary_rows`` should come from ``metrics.aggregate_records()``.
    Each row must have keys: dataset, k, avg_ccs.
    """
    _apply_style()

    datasets: Dict[str, Dict[int, float]] = {}
    for row in summary_rows:
        k = int(row["k"])
        if k == 0:
            continue
        ds = row["dataset"]
        ccs = row.get("avg_ccs")
        if ccs == "" or ccs is None:
            continue
        datasets.setdefault(ds, {})[k] = float(ccs)

    fig, ax = plt.subplots()
    for idx, (ds, k_ccs) in enumerate(sorted(datasets.items())):
        ks = sorted(k_ccs.keys())
        vals = [k_ccs[k] for k in ks]
        color = _PALETTE[idx % len(_PALETTE)]
        ax.plot(ks, vals, marker="o", label=ds.upper(), color=color)

    ax.set_xlabel("Segmentation Depth (K)")
    ax.set_ylabel("Average CCS")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CCS by Gold Label
# ---------------------------------------------------------------------------
def plot_ccs_by_label(
    stratified_data: Dict[str, Dict[str, Any]],
    out_path: str,
    *,
    title: str = "CCS by Gold Label",
) -> None:
    """Bar chart of average CCS broken down by gold label."""
    _apply_style()

    labels = sorted(stratified_data.keys())
    avg_vals = [stratified_data[lbl]["avg_ccs"] for lbl in labels]
    counts = [stratified_data[lbl]["num_examples"] for lbl in labels]

    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    bars = ax.bar(x, avg_vals, color=_PALETTE[: len(labels)], edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average CCS")
    ax.set_title(title)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Positional Flip-Rate  Bar Chart
# ---------------------------------------------------------------------------
def plot_positional_flip_rate(
    stratified_data: Dict[str, Dict[str, Any]],
    out_path: str,
    *,
    title: str = "Chunk Flip Rate by Position",
) -> None:
    """Bar chart of flip rate by chunk position (first / middle / last)."""
    _apply_style()

    buckets = ["first", "middle", "last"]
    present = [b for b in buckets if b in stratified_data]
    if not present:
        present = sorted(stratified_data.keys())

    flip_rates = [stratified_data[b]["flip_rate"] for b in present]
    totals = [stratified_data[b]["total_chunks"] for b in present]

    fig, ax = plt.subplots()
    x = np.arange(len(present))
    bars = ax.bar(x, flip_rates, color=_PALETTE[: len(present)], edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in present])
    ax.set_ylabel("Flip Rate")
    ax.set_title(title)
    for bar, total in zip(bars, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={total}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gold vs Non-Gold  Flip-Rate Comparison
# ---------------------------------------------------------------------------
def plot_gold_vs_nongold_flip(
    gold_strat: Dict[str, Dict[str, Any]],
    out_path: str,
    *,
    title: str = "Flip Rate: Gold vs Non-Gold Chunks",
) -> None:
    """Side-by-side comparison of flip rates for gold vs non-gold chunks."""
    _apply_style()

    categories = ["Gold Evidence\nChunks", "Non-Gold\nChunks"]
    rates = [
        gold_strat["gold_chunks"]["flip_rate"],
        gold_strat["non_gold_chunks"]["flip_rate"],
    ]
    totals = [
        gold_strat["gold_chunks"]["total"],
        gold_strat["non_gold_chunks"]["total"],
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(categories))
    bars = ax.bar(x, rates, color=[_PALETTE[0], _PALETTE[1]], edgecolor="white", width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Flip Rate")
    ax.set_title(title)
    for bar, total in zip(bars, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={total}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-hop Dependency  Heatmap
# ---------------------------------------------------------------------------
def plot_dependency_heatmap(
    example_ids: List[str],
    matrix: List[List[int]],
    out_path: str,
    *,
    title: str = "Chunk Causal Dependency Matrix",
    max_examples: int = 30,
) -> None:
    """Heatmap of the N×K flip matrix from the multi-hop dependency analysis."""
    _apply_style()

    if not matrix:
        return

    # Truncate for readability
    ids = example_ids[:max_examples]
    data = matrix[:max_examples]

    max_k = max(len(row) for row in data)
    padded = [row + [0] * (max_k - len(row)) for row in data]
    arr = np.array(padded, dtype=float)

    fig, ax = plt.subplots(figsize=(max(6, max_k * 0.8), max(4, len(ids) * 0.35)))
    cax = ax.imshow(arr, aspect="auto", cmap="RdYlGn_r", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xlabel("Chunk ID")
    ax.set_ylabel("Example")
    ax.set_title(title)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=7)
    ax.set_xticks(range(max_k))
    fig.colorbar(cax, ax=ax, label="Flipped", shrink=0.8)
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dataset Comparison  Bar Chart
# ---------------------------------------------------------------------------
def plot_dataset_comparison(
    stratified_data: Dict[str, Dict[str, Any]],
    out_path: str,
    *,
    title: str = "CCS Comparison Across Datasets",
) -> None:
    """Grouped bar chart comparing CCS across datasets at each K."""
    _apply_style()

    # Group by dataset
    ds_data: Dict[str, Dict[int, float]] = {}
    for key, stats in stratified_data.items():
        ds = stats["dataset"]
        k = stats["k"]
        ds_data.setdefault(ds, {})[k] = stats["avg_ccs"]

    all_ks = sorted({k for d in ds_data.values() for k in d})
    datasets = sorted(ds_data.keys())
    n_ds = len(datasets)
    width = 0.7 / n_ds

    fig, ax = plt.subplots()
    x = np.arange(len(all_ks))
    for idx, ds in enumerate(datasets):
        vals = [ds_data[ds].get(k, 0) for k in all_ks]
        offset = (idx - n_ds / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=ds.upper(), color=_PALETTE[idx % len(_PALETTE)])

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in all_ks])
    ax.set_ylabel("Average CCS")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
