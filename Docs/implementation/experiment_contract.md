# MemFaith Experiment Contract

Every MemFaith run must materialize the following objects for each example and each `K`.

## Required Inputs

- `query`
- `gold_answer`
- `task_type`: `classification` or `qa`
- `evidence_segments`
- `distractor_segments`
- deterministic seed

## Required Intermediate Objects

- long context string with stable segment ordering
- chunk list with stable `chunk_id`
- full-context prediction
- one ablated prediction per chunk when `K > 0`

## Required Outputs Per Record

- `dataset`
- `example_id`
- `k`
- `context.context_id`
- `chunks`
- `full_context.prediction`
- `full_context.is_correct`
- `ablations[*].prediction`
- `ablations[*].comparison_to_full`
- `ccs_example`

## Invariants

- decoding must be deterministic for real-model runs
- identical prompts must hit cache instead of recomputing
- chunk boundaries must be sentence-aware, not raw token cuts
- FEVER-style tasks must compare normalized labels exactly
- QA tasks must compare by exact-normalized match first, then token F1

## Phase Gate for the Extension

Mohamad's scorer work only starts after these are true:

1. FEVER smoke and prepared runs complete end-to-end.
2. The JSONL schema is frozen.
3. Chunk-label export succeeds on the canonical log.
