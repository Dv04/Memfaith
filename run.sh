#!/usr/bin/env bash
set -euo pipefail

# Ensure local packages are found
export PYTHONPATH=".:external/unified_editing:${PYTHONPATH:-}"

mkdir -p outputs experiments/plots

# Seeds and logs
python experiments/run_ef_fever.py \
  --model-path models/gpt2 \
  --split dev \
  --max-examples 10 \
  --log-path outputs/ef_fever_gpt2_rome_seed0.jsonl \
  --hparams-path external/unified_editing/hparams/ROME/gpt2.json \
  --device mps \
  --seed 0

python experiments/run_ef_fever.py \
  --model-path models/gpt2 \
  --split dev \
  --max-examples 10 \
  --log-path outputs/ef_fever_gpt2_rome_seed1.jsonl \
  --hparams-path external/unified_editing/hparams/ROME/gpt2.json \
  --device mps \
  --seed 1

python experiments/run_ef_fever.py \
  --model-path models/gpt2 \
  --split dev \
  --max-examples 10 \
  --log-path outputs/ef_fever_gpt2_rome_seed2.jsonl \
  --hparams-path external/unified_editing/hparams/ROME/gpt2.json \
  --device mps \
  --seed 2

# Merge logs
cat outputs/ef_fever_gpt2_rome_seed{0,1,2}.jsonl > outputs/ef_fever_gpt2_rome_all.jsonl

# Evaluate EF
python -m src.ef_eval --log-path outputs/ef_fever_gpt2_rome_all.jsonl
python -m src.ef_eval --log-path outputs/ef_fever_gpt2_rome_seed0.jsonl
python -m src.ef_eval --log-path outputs/ef_fever_gpt2_rome_seed1.jsonl
python -m src.ef_eval --log-path outputs/ef_fever_gpt2_rome_seed2.jsonl

# Plot flip vs rationale length
python -m src.analysis.plots \
  --log-path outputs/ef_fever_gpt2_rome_all.jsonl \
  --out-path experiments/plots/ef_vs_rationale_length.png
