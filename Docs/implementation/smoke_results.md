# MemFaith Smoke Results

These are **integration results**, not report-grade research findings. They exist to prove that the new CCS stack runs end to end and produces the expected artifacts.

## FEVER-Style Smoke

Source outputs:

- `outputs/memfaith/fever_smoke_ccs.jsonl`
- `outputs/memfaith/fever_smoke_summary.csv`

Summary:

| K | Examples | Full Accuracy | Avg CCS | Total Ablations |
|---|---:|---:|---:|---:|
| 0 | 3 | 1.00 | baseline only | 0 |
| 2 | 3 | 1.00 | 0.333333 | 6 |
| 4 | 3 | 1.00 | 0.194444 | 10 |

## HotpotQA-Style Smoke

Source outputs:

- `outputs/memfaith/hotpot_smoke_ccs.jsonl`
- `outputs/memfaith/hotpot_smoke_summary.csv`

Summary:

| K | Examples | Full Accuracy | Avg CCS | Total Ablations |
|---|---:|---:|---:|---:|
| 0 | 2 | 1.00 | baseline only | 0 |
| 2 | 2 | 1.00 | 0.75 | 4 |
| 4 | 2 | 1.00 | 0.50 | 8 |

## Extension-Track Artifact

Chunk-label export:

- `outputs/memfaith/fever_chunk_labels.csv`

Current label count:

- 16 chunk-level rows exported from the FEVER smoke log

## Interpretation

The important result here is not the numeric value itself. The important result is that the repo now produces:

- stable JSONL experiment records
- summary tables
- chunk-level labels for the pruning extension
- deterministic reruns through one command

That is the correct week-1/week-2 target for the full team.
