# MemFaith Answer Comparison Policy

## FEVER / Classification

Normalization rules:

- uppercase
- map `NEI` and `NOT ENOUGH INFO` to `NOT_ENOUGH_INFO`
- map singular forms like `REFUTE` to `REFUTES`

Flip rule:

- flip if normalized full-context label and normalized ablated label differ

Correctness rule:

- correct if normalized prediction equals normalized gold label

## QA / Short Answer

Comparison order:

1. normalized exact match
2. token-level F1

Default threshold:

- F1 `< 0.6` counts as a flip
- F1 `>= 0.6` is treated as semantically preserved in the current local implementation

Future hook:

- ambiguous QA comparisons can be routed to an LLM judge later, but the current core implementation intentionally keeps the comparator local and deterministic

## Why This Policy Exists

The current repo had a real risk of metric drift between proposal language and code. This document and `src/memfaith/comparators.py` now define the canonical behavior in one place.
