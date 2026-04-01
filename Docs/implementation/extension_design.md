# MemFaith Extension Design

## Goal

Convert expensive chunk-ablation outputs into a cheap chunk scorer that can rank or prune context before inference.

## Label Formats

Implemented now:

- binary causal label from `comparison_to_full.flipped`

Planned next:

- scalar causal weight from repeated ablations or confidence deltas
- rank labels within each example

## Current Repo Support

- `scripts/export_chunk_labels.py`
- `src/memfaith/label_builders.py`

These emit `(query, chunk_text, causal_label)` rows directly from the canonical JSONL logs.

## Week-2 Readiness

By the end of week 2, Mohamad's branch should have:

- stable chunk-label CSV export
- frozen field names
- positive/negative label counts from FEVER logs

That is sufficient to start model selection without forcing the extension to become the main project.

## Recommended First Scorer

- lightweight encoder classifier
- query + chunk as input
- binary output
- class-imbalance-aware loss

This should stay in a separate extension path and must not block CCS delivery.
