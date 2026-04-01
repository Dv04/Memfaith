"""Basic diagnostic plots for EF metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_flip_rates_by_dataset(ef_summaries: Dict[str, Dict[str, float]], out_path: str) -> None:
    labels = list(ef_summaries.keys())
    flip_tgt = [ef_summaries[name]["flip_target"] for name in labels]
    flip_ctrl = [ef_summaries[name]["flip_control"] for name in labels]
    ef_vals = [ef_summaries[name]["ef"] for name in labels]

    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(8, 4))
    plt.bar(x - width, flip_tgt, width, label="Flip target")
    plt.bar(x, flip_ctrl, width, label="Flip control")
    plt.bar(x + width, ef_vals, width, label="EF")
    plt.xticks(x, labels)
    plt.ylabel("Probability")
    plt.title("Editability–Faithfulness by dataset")
    plt.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_ef_vs_rationale_length(log_records: List[Dict], out_path: str) -> None:
    lengths = []
    flips = []
    for rec in log_records:
        rationale = rec.get("orig_rationale") or ""
        length = len(rationale.split())
        post_answer = rec.get("post_edit_answer_target")
        orig_answer = rec.get("orig_answer")
        if post_answer is None or orig_answer is None:
            continue
        lengths.append(length)
        flips.append(int(post_answer != orig_answer))

    if not lengths:
        return

    plt.figure(figsize=(6, 4))
    plt.scatter(lengths, flips, alpha=0.5)
    plt.xlabel("Rationale length (tokens)")
    plt.ylabel("Flip (target edit)")
    plt.title("Flip probability vs rationale length")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_ef_vs_edit_efficacy(log_records: List[Dict], out_path: str) -> None:
    efficacies = []
    flips = []
    for rec in log_records:
        stats = rec.get("edit_stats_target") or {}
        eff = stats.get("edit_efficacy")
        post_answer = rec.get("post_edit_answer_target")
        orig_answer = rec.get("orig_answer")
        if eff is None or post_answer is None or orig_answer is None:
            continue
        efficacies.append(eff)
        flips.append(int(post_answer != orig_answer))

    if not efficacies:
        return

    plt.figure(figsize=(6, 4))
    bins = np.linspace(0, 1, 6)
    digitized = np.digitize(efficacies, bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    flip_rates = []
    for i in range(1, len(bins)):
        mask = [idx for idx, d in enumerate(digitized) if d == i]
        if not mask:
            flip_rates.append(0)
            continue
        flip_rates.append(sum(flips[idx] for idx in mask) / len(mask))

    plt.plot(bin_centers, flip_rates, marker="o")
    plt.xlabel("Edit efficacy")
    plt.ylabel("Flip probability (target)")
    plt.title("Flip probability vs edit efficacy")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _load_log_records(log_path: Union[str, Path]) -> List[Dict]:
    with Path(log_path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_flip_vs_rationale_length(
    log_path: Union[str, Path],
    out_path: Union[str, Path],
    bins: int = 10,
) -> None:
    records = _load_log_records(log_path)
    lengths: List[int] = []
    flips: List[int] = []

    for rec in records:
        rationale = rec.get("rationale_pre") or rec.get("orig_rationale") or ""
        pre = rec.get("answer_pre") or rec.get("orig_answer")
        post = rec.get("answer_post_target") or rec.get("post_edit_answer_target")
        if pre is None or post is None:
            continue
        lengths.append(len(rationale.split()))
        flips.append(int(post != pre))

    if not lengths:
        raise ValueError("No valid records found to plot flip vs rationale length.")

    lengths_arr = np.array(lengths)
    flips_arr = np.array(flips)
    min_len, max_len = lengths_arr.min(), lengths_arr.max()
    if max_len == min_len:
        bin_edges = np.array([min_len - 0.5, max_len + 0.5])
    else:
        bin_edges = np.linspace(min_len, max_len, bins + 1)
    bin_ids = np.digitize(lengths_arr, bin_edges, right=False) - 1

    bin_centers: List[float] = []
    flip_rates: List[float] = []
    for idx in range(len(bin_edges) - 1):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        bin_centers.append((bin_edges[idx] + bin_edges[idx + 1]) / 2)
        flip_rates.append(float(flips_arr[mask].mean()))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(bin_centers, flip_rates, marker="o")
    plt.xlabel("Rationale length (tokens)")
    plt.ylabel("Flip probability (target edit)")
    plt.title("Flip probability vs rationale length")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot EF flip rate vs rationale length.")
    parser.add_argument("--log-path", type=str, required=True)
    parser.add_argument(
        "--out-path",
        type=str,
        default="experiments/plots/ef_vs_rationale_length.png",
    )
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    plot_flip_vs_rationale_length(args.log_path, args.out_path, bins=args.bins)


if __name__ == "__main__":
    main()
