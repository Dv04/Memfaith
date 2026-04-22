"""
MemFaith Colab Execution Script
================================
Copy each section below into a separate Colab cell and run them in order.
Make sure you've set Runtime > Change runtime type > T4 GPU first!
"""

# ===========================================================================
# CELL 1: Mount Google Drive & Upload Project
# ===========================================================================
# Upload Memfaith_colab.zip to the root of your Google Drive first,
# then run this cell.

from google.colab import drive
drive.mount('/content/drive')

# ===========================================================================
# CELL 2: Unzip & Navigate
# ===========================================================================
# !cp /content/drive/MyDrive/Memfaith_colab.zip /content/
# !cd /content && unzip -qo Memfaith_colab.zip
# %cd /content/Memfaith

# ===========================================================================
# CELL 3: Install Dependencies
# ===========================================================================
# !pip install -q vllm transformers datasets spacy rich einops tqdm matplotlib pandas scipy scikit-learn accelerate sentencepiece
# !python -m spacy download en_core_web_sm

# ===========================================================================
# CELL 4: Verify GPU
# ===========================================================================
# import torch
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"GPU: {torch.cuda.get_device_name(0)}")
# print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ===========================================================================
# CELL 5: Download GPT2-XL model (primary baseline)
# ===========================================================================
# from transformers import AutoTokenizer, AutoModelForCausalLM
# model_name = "gpt2-xl"
# print(f"Downloading {model_name}...")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer.save_pretrained("models/gpt2-xl")
# model.save_pretrained("models/gpt2-xl")
# del model, tokenizer  # free memory before vLLM takes over
# import gc; gc.collect()
# import torch; torch.cuda.empty_cache()
# print("Model saved to models/gpt2-xl")

# ===========================================================================
# CELL 6: Run FEVER Ablation Sweep
# ===========================================================================
# !python scripts/run_fever_ccs.py \
#     --backend vllm \
#     --model-path models/gpt2-xl \
#     --max-new-tokens 24 \
#     --k-values "0,2,4,8,16" \
#     --dataset-path data/memfaith/fever_smoke.jsonl \
#     --output-path outputs/memfaith/fever_final_ccs.jsonl \
#     --summary-path outputs/memfaith/fever_final_summary.csv \
#     --cache-path outputs/memfaith/fever_final_cache.sqlite \
#     --seed 0

# ===========================================================================
# CELL 7: Run HotpotQA Ablation Sweep
# ===========================================================================
# !python scripts/run_hotpotqa_ccs.py \
#     --backend vllm \
#     --model-path models/gpt2-xl \
#     --max-new-tokens 24 \
#     --k-values "0,2,4,8,16" \
#     --dataset-path data/memfaith/hotpot_smoke.jsonl \
#     --output-path outputs/memfaith/hotpot_final_ccs.jsonl \
#     --summary-path outputs/memfaith/hotpot_final_summary.csv \
#     --cache-path outputs/memfaith/hotpot_final_cache.sqlite \
#     --seed 0

# ===========================================================================
# CELL 8: Save Results Back to Google Drive
# ===========================================================================
# import shutil
# shutil.copytree("outputs/memfaith", "/content/drive/MyDrive/memfaith_results", dirs_exist_ok=True)
# print("Results saved to Google Drive under memfaith_results/")
# print("Hand the following files to Dev:")
# !ls -lh outputs/memfaith/
