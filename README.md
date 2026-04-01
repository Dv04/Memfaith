# Editability–Faithfulness (EF) Pipeline

This repository evaluates **Editability–Faithfulness (EF)** for free-text rationales. EF asks whether a model’s answer flips more often when we surgically edit internal knowledge related to a rationale-cited fact (target) compared with a matched but uncited control fact. We focus on FEVER claims, GPT-2 XL rationales, and ROME edits applied via the unified-model-editing toolkit.

---

## Prerequisites

- Linux server with at least one CUDA-capable GPU (ROME + GPT-2 XL is impractical on CPU).
- Python 3.11 (tested) and a working `conda` or `venv`.
- FEVER and StrategyQA data under `data/` (see layout below).
- Local GPT-2 XL weights under `models/gpt2`.
- `external/unified_editing` checkout of the [unified-model-editing](https://github.com/scalable-model-editing/unified-model-editing) repo.

---

## Environment Setup

```bash
conda create -n ef_explain python=3.11
conda activate ef_explain

# Core dependencies (adjust CUDA wheel if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate sentencepiece
pip install numpy scipy pandas scikit-learn tqdm rich einops matplotlib

# Unified model editing dependencies
pip install -r external/unified_editing/requirements.txt
```

> Optional: install logging/tracking stacks such as `wandb` or `mlflow` if desired.

---

## Unified Model Editing Checkout

```bash
cd ef_editability
mkdir -p external
cd external
git clone https://github.com/scalable-model-editing/unified-model-editing.git unified_editing
cd ..
```

The ROME hparams referenced below live under `external/unified_editing/hparams/ROME/`.

---

## GPT-2 XL Weights

Place GPT-2 XL files under `models/gpt2`. Two options:

1. **Auto-download via Transformers**
   ```bash
   python - <<'PY'
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
   tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
   model.save_pretrained("models/gpt2")
   tokenizer.save_pretrained("models/gpt2")
   PY
   ```
2. **Manual copy** of `config.json`, `model.safetensors`/`pytorch_model.bin`, `tokenizer.json`, `vocab.json`, `merges.txt`, etc. into `models/gpt2`.

All scripts in this repo accept `--model-path` so you can point at a different directory if needed.

---

## Data Layout

```
data/
  fever/
    train.jsonl
    paper_dev.jsonl
    paper_test.jsonl
    ...
  strategyqa/
    strategyqa_train.json
    strategyqa_test.json
    ...
```

Place additional FEVER splits (shared task) in the same folder if you plan to swap splits.

---

## Sanity Check: Single ROME Edit

Before running EF, verify that ROME can edit your local GPT-2 XL weights:

```bash
python scripts/test_rome_single_edit.py \
  --model-path models/gpt2 \
  --hparams-path external/unified_editing/hparams/ROME/gpt2.json \
  --device cuda
```

This prints the completion to “LeBron James plays the sport of” before and after editing the fact to “football,” along with basic edit metadata. Run this on a GPU box; CPU will be extremely slow.

---

## Running EF on FEVER

Execute real EF runs with:

```bash
python experiments/run_ef_fever.py \
  --model-path models/gpt2 \
  --split dev \
  --max-examples 200 \
  --log-path outputs/ef_fever_gpt2xl_rome_seed0.jsonl \
  --hparams-path external/unified_editing/hparams/ROME/gpt2.json \
  --device cuda \
  --max-new-tokens 96
```

What happens:

1. GPT-2 XL generates FEVER answers + rationales.
2. Heuristic triple extraction selects a target triple cited in the rationale; a global pool provides a matched control triple.
3. For each triple, `apply_rome_edit` flips the fact (simple heuristics for a counterfactual object).
4. The edited model answers the same claim; pre/post answers plus edit metadata are logged as JSONL records (one per example).

`outputs/ef_fever_gpt2xl_rome_seed0.jsonl` becomes the canonical log for later evaluation, plotting, and case studies.

---

## Computing EF Metrics

```bash
python -m src.ef_eval \
  --log-path outputs/ef_fever_gpt2xl_rome_seed0.jsonl \
  --n-bootstrap 2000
```

The CLI prints Flip_tgt, Flip_ctrl, EF, and 95% bootstrap confidence intervals along with the number of usable records.

---

## Plotting Flip Probability vs Rationale Length

```bash
python -m src.analysis.plots \
  --log-path outputs/ef_fever_gpt2xl_rome_seed0.jsonl \
  --out-path experiments/plots/ef_vs_rationale_length.png \
  --bins 12
```

The resulting PNG (saved under `experiments/plots/`) visualizes whether longer rationales correlate with higher flip rates, serving as a Goodhart-style sanity check.

---

## Inspecting Case Studies

1. Launch Jupyter (or VS Code) with the `ef_explain` environment.
2. Open `notebooks/02_ef_case_studies.ipynb`.
3. Set the log path inside the notebook to the EF run you care about (e.g., `outputs/ef_fever_gpt2xl_rome_seed0.jsonl`).
4. Use the provided cells to filter examples (target flip only, both flip, etc.), review pre/post rationales, and add manual annotations about whether the rationale truly used the edited fact.

Notebook `01_preview_rationales.ipynb` remains a lightweight sanity check on rationale quality before running expensive edits.

---

## Limitations & Notes

- **Heuristic triple extraction:** the `(subject, relation, object)` parser is intentionally simple; EF scores may underestimate causal faithfulness if we miss important facts.
- **Counterfactual object selection:** currently uses hand-written swaps (e.g., basketball ↔ football, yes ↔ no, “not X”). Refine this if you need domain-specific edits.
- **ROME resource cost:** editing GPT-2 XL requires significant GPU memory and time because we deep-copy the model per edit to keep the base untouched.
- **Research code:** expect to tweak prompts, thresholds, or logging for your experiments. Please test on small slices before scaling up.

---

## Modified Files in This Update

- `src/editing_wrapper.py`
- `src/rationale_model.py`
- `src/analysis/plots.py`
- `src/ef_eval.py`
- `experiments/run_ef_fever.py`
- `experiments/__init__.py`
- `scripts/test_rome_single_edit.py`
- `README.md`
