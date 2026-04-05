PYTHON ?= python

prepare_data:
	@echo "Assuming FEVER and StrategyQA data already placed under data/."

run_memfaith_smoke:
	$(PYTHON) scripts/run_memfaith_smoke.py

run_memfaith_fever:
	$(PYTHON) scripts/run_fever_ccs.py

run_memfaith_hotpot:
	$(PYTHON) scripts/run_hotpotqa_ccs.py

# --- Week 3 & 4 Targets ---

synthetic_data:
	$(PYTHON) scripts/generate_synthetic_data.py

full_eval:
	bash run_eval.sh

plots:
	$(PYTHON) -c "import json; from src.memfaith import load_experiment_log, aggregate_records; from src.memfaith.plotting import plot_ccs_degradation_curve; log=load_experiment_log('outputs/memfaith/fever_synth_ccs.jsonl'); summary=aggregate_records(log); plot_ccs_degradation_curve(summary, 'outputs/memfaith/plots/ccs_degradation_curve.png')"

case_studies:
	$(PYTHON) scripts/extract_case_studies.py

label_export:
	$(PYTHON) scripts/export_combined_labels.py

# --- Legacy Targets ---

run_ef_fever:
	$(PYTHON) -m experiments.run_ef_fever --config configs/fever_gpt2xl_rome.yaml

run_ef_strategyqa:
	$(PYTHON) -m experiments.run_ef_strategyqa --config configs/strategyqa_gpt2xl_rome.yaml

run_baselines:
	@echo "Baseline evaluations are placeholders; implement ERASER/CC-SHAP to enable this target."
