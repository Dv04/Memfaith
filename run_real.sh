#!/usr/bin/env bash
set -euo pipefail

# MemFaith CCS Full Evaluation Pipeline (REAL DATA)
# This script executes the entire workflow over real data:
# 1. Downloads Wikipedia passages and builds contexts w/ BM25 distractors
#    (Using 1000 examples from each dataset for a statistically significant sample)
# 2. Runs CCS ablation pipeline on the real contexts
# 3. Aggregates results and plots.

export PYTHONPATH=".:${PYTHONPATH:-}"

mkdir -p outputs/memfaith/plots
mkdir -p data/memfaith

echo "--- 1. Building Real FEVER Contexts (Downloading Wikipedia ~20GB, please wait) ---"
python3 scripts/build_fever_contexts.py \
  --output data/memfaith/fever_prepared.jsonl \
  --max-examples 1000 \
  --n-distractors 10 \
  --wiki-passages 250000

echo "--- 2. Building Real HotpotQA Contexts ---"
python3 scripts/build_hotpotqa_contexts.py \
  --output data/memfaith/hotpotqa_prepared.jsonl \
  --max-examples 1000 \
  --extra-distractors 5 \
  --wiki-passages 250000

echo "--- 3. Running FEVER Ablation on Real Contexts ---"
python3 scripts/run_fever_ccs.py \
  --dataset-path data/memfaith/fever_prepared.jsonl \
  --output-path outputs/memfaith/fever_real_ccs.jsonl \
  --summary-path outputs/memfaith/fever_real_summary.csv \
  --cache-path outputs/memfaith/fever_real_cache.sqlite \
  --k-values "0,2,4,8" \
  --max-examples 1000

echo "--- 4. Running HotpotQA Ablation on Real Contexts ---"
python3 scripts/run_hotpotqa_ccs.py \
  --dataset-path data/memfaith/hotpotqa_prepared.jsonl \
  --output-path outputs/memfaith/hotpot_real_ccs.jsonl \
  --summary-path outputs/memfaith/hotpot_real_summary.csv \
  --cache-path outputs/memfaith/hotpot_real_cache.sqlite \
  --k-values "0,2,4,8" \
  --max-examples 1000

echo "--- 5. Exporting Final Combined Label CSV for Discriminator Training ---"
python3 scripts/export_combined_labels.py \
  --fever-log outputs/memfaith/fever_real_ccs.jsonl \
  --hotpot-log outputs/memfaith/hotpot_real_ccs.jsonl \
  --output-path outputs/memfaith/combined_chunk_labels.csv

echo "--- 6. Extracting Case Studies ---"
python3 scripts/extract_case_studies.py \
  --fever-log outputs/memfaith/fever_real_ccs.jsonl \
  --hotpot-log outputs/memfaith/hotpot_real_ccs.jsonl \
  --output-path outputs/memfaith/case_studies.md

echo "Pipeline complete. The final CSV is ready in outputs/memfaith/combined_chunk_labels.csv"
