PYTHON ?= python

prepare_data:
	@echo "Assuming FEVER and StrategyQA data already placed under data/."

run_memfaith_smoke:
	$(PYTHON) scripts/run_memfaith_smoke.py

run_memfaith_fever:
	$(PYTHON) scripts/run_fever_ccs.py

run_memfaith_hotpot:
	$(PYTHON) scripts/run_hotpotqa_ccs.py

run_ef_fever:
	$(PYTHON) -m experiments.run_ef_fever --config configs/fever_gpt2xl_rome.yaml

run_ef_strategyqa:
	$(PYTHON) -m experiments.run_ef_strategyqa --config configs/strategyqa_gpt2xl_rome.yaml

run_baselines:
	@echo "Baseline evaluations are placeholders; implement ERASER/CC-SHAP to enable this target."
