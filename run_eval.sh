#!/usr/bin/env bash
set -euo pipefail

# MemFaith CCS Full Evaluation Pipeline
# This script executes the entire week 3-4 workflow including:
# 1. Dataset generation (synthetic)
# 2. CCS inference (FEVER & Hotpot)
# 3. Label export
# 4. Stratified analysis & plotting
# 5. Case study extraction
# 6. Report preparation

export PYTHONPATH=".:${PYTHONPATH:-}"

mkdir -p outputs/memfaith/plots

echo "--- 1. Generating Synthetic Data ---"
python3 scripts/generate_synthetic_data.py

echo "--- 2. Running FEVER CCS (Synthetic) ---"
python3 scripts/run_fever_ccs.py \
  --dataset-path data/memfaith/fever_synthetic.jsonl \
  --output-path outputs/memfaith/fever_synth_ccs.jsonl \
  --summary-path outputs/memfaith/fever_synth_summary.csv \
  --cache-path outputs/memfaith/fever_synth_cache.sqlite \
  --backend transformers \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --k-values "0,2,4,8" \
  --max-examples 20

echo "--- 3. Running HotpotQA CCS (Synthetic) ---"
python3 scripts/run_hotpotqa_ccs.py \
  --dataset-path data/memfaith/hotpot_synthetic.jsonl \
  --output-path outputs/memfaith/hotpot_synth_ccs.jsonl \
  --summary-path outputs/memfaith/hotpot_synth_summary.csv \
  --cache-path outputs/memfaith/hotpot_synth_cache.sqlite \
  --backend transformers \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --k-values "0,2,4,8" \
  --max-examples 20

echo "--- 4. Exporting Combined Chunk Labels ---"
python3 scripts/export_combined_labels.py \
  --fever-log outputs/memfaith/fever_synth_ccs.jsonl \
  --hotpot-log outputs/memfaith/hotpot_synth_ccs.jsonl \
  --output-path outputs/memfaith/combined_chunk_labels.csv

echo "--- 5. Plotting Results ---"
python3 -c "
import json
from src.memfaith import load_experiment_log, aggregate_records
from src.memfaith.plotting import *
from src.memfaith.stratification import *
from src.memfaith.multi_hop_analysis import *

fever_log = load_experiment_log('outputs/memfaith/fever_synth_ccs.jsonl')
hotpot_log = load_experiment_log('outputs/memfaith/hotpot_synth_ccs.jsonl')
all_log = fever_log + hotpot_log

# Degradation Curve
summary = aggregate_records(all_log)
plot_ccs_degradation_curve(summary, 'outputs/memfaith/plots/ccs_degradation_curve.png')

# Label Stratification (FEVER)
label_strat = stratify_by_label(fever_log)
plot_ccs_by_label(label_strat, 'outputs/memfaith/plots/ccs_by_label.png')

# Positional Stratification
pos_strat = stratify_by_position(all_log)
plot_positional_flip_rate(pos_strat, 'outputs/memfaith/plots/positional_flip_rate.png')

# Gold vs Non-Gold
gold_strat = stratify_by_gold_coverage(all_log)
plot_gold_vs_nongold_flip(gold_strat, 'outputs/memfaith/plots/gold_vs_nongold_flip.png')

# Dependency Heatmap (Hotpot)
ids, matrix = build_dependency_matrix(hotpot_log, k_filter=4)
plot_dependency_heatmap(ids, matrix, 'outputs/memfaith/plots/dependency_heatmap.png')

# Dataset Comparison
ds_strat = stratify_by_dataset(all_log)
plot_dataset_comparison(ds_strat, 'outputs/memfaith/plots/dataset_comparison.png')
"

echo "--- 6. Extracting Case Studies ---"
python3 scripts/extract_case_studies.py \
  --fever-log outputs/memfaith/fever_synth_ccs.jsonl \
  --hotpot-log outputs/memfaith/hotpot_synth_ccs.jsonl \
  --output-path outputs/memfaith/case_studies.md

echo "--- 7. Reviewing HotpotQA Flips ---"
python3 scripts/review_hotpot_flips.py \
  --log-path outputs/memfaith/hotpot_synth_ccs.jsonl \
  --output-path outputs/memfaith/hotpot_flip_review.md

echo "Pipeline complete. Outputs in outputs/memfaith/"
