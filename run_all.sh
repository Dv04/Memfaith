#!/bin/bash
set -e

# MemFaith Execution Orchestrator
# Executes massive ablation runs across FEVER and HotpotQA tasks.

MODEL_PATH=${1:-"models/gpt2"}
BACKEND="vllm"
DEVICE="cuda"
MAX_NEW_TOKENS=24
TENSOR_PARALLEL_SIZE=1 # Default scalar tensor parallel. Adjust for multi-GPU if needed.
K_VALUES="0,2,4,8,16" # 0 provides the necessary full-context baseline

echo "========================================"
echo "Starting MemFaith Ablation Suite"
echo "Model Path: ${MODEL_PATH}"
echo "Backend: ${BACKEND}"
echo "K Values: ${K_VALUES}"
echo "========================================"

echo ""
echo "=== 1. Starting FEVER Runs ==="
python scripts/run_fever_ccs.py \
    --backend "${BACKEND}" \
    --model-path "${MODEL_PATH}" \
    --device "${DEVICE}" \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --k-values "${K_VALUES}" \
    --dataset-path "data/memfaith/fever_smoke.jsonl" \
    --output-path "outputs/memfaith/fever_final_ccs.jsonl" \
    --summary-path "outputs/memfaith/fever_final_summary.csv" \
    --cache-path "outputs/memfaith/fever_final_cache.sqlite" \
    --seed 0

echo "FEVER generation completed successfully."

echo ""
echo "=== 2. Starting HotpotQA Runs ==="
python scripts/run_hotpotqa_ccs.py \
    --backend "${BACKEND}" \
    --model-path "${MODEL_PATH}" \
    --device "${DEVICE}" \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --k-values "${K_VALUES}" \
    --dataset-path "data/memfaith/hotpot_smoke.jsonl" \
    --output-path "outputs/memfaith/hotpot_final_ccs.jsonl" \
    --summary-path "outputs/memfaith/hotpot_final_summary.csv" \
    --cache-path "outputs/memfaith/hotpot_final_cache.sqlite" \
    --seed 0

echo "HotpotQA generation completed successfully."

echo ""
echo "All tasks executed. Review CSV summaries in outputs/memfaith/."
