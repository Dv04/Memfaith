# MemFaith Week 1-2 Delivery Map

This is the concrete "through week 2" package now present in the repo.

## Dev Sanghvi

Implemented / drafted:

- canonical scope lock in `Docs/implementation/project_scope.md`
- experiment contract in `Docs/implementation/experiment_contract.md`
- answer comparison policy in `Docs/implementation/answer_comparison_policy.md`
- executable comparator and metrics code in `src/memfaith/comparators.py` and `src/memfaith/metrics.py`

## Ansh Dabral

Implemented:

- deterministic context builder in `src/memfaith/context_builder.py`
- deterministic chunker in `src/memfaith/chunking.py`
- resumable SQLite cache in `src/memfaith/cache.py`
- CCS runner in `src/memfaith/runner.py`
- runnable FEVER and Hotpot smoke scripts in `scripts/run_fever_ccs.py` and `scripts/run_hotpotqa_ccs.py`

## Mohamad Kreidieh

Implemented:

- extension design doc in `Docs/implementation/extension_design.md`
- chunk-label export utility in `src/memfaith/label_builders.py`
- CLI export path in `scripts/export_chunk_labels.py`

## Jade Yan

Implemented / drafted:

- dataset protocol memo in `Docs/implementation/dataset_protocols.md`
- answer-comparison policy doc
- smoke datasets for table/debug generation
- summary CSV generation in `src/memfaith/metrics.py`

## Cross-Team Integration

Added:

- unified `src/memfaith` package
- smoke datasets for local contract validation
- `scripts/run_memfaith_smoke.py` for a one-command integration pass
- unit tests in `tests/test_memfaith_core.py`
