# MemFaith Dataset Protocols

## FEVER Mainline

Target task:

- fact verification with labels `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`

Prepared example requirements:

- claim text
- gold label
- evidence segment text, not only FEVER sentence IDs
- distractor passages from Wikipedia or a prepared retrieval pool

Important local constraint:

The repo contains FEVER JSONL splits, but those records mostly carry evidence pointers rather than sentence text. For real FEVER CCS runs, the team still needs a prepared evidence-text stage or a Wikipedia lookup stage before running `src/memfaith`.

## HotpotQA Mainline

Target task:

- short-answer multi-hop QA

Prepared example requirements:

- question text
- gold answer
- supporting paragraphs
- non-supporting paragraphs used as distractors

The codebase supports prepared Hotpot-style JSONL immediately and a standard Hotpot JSON adapter if raw context paragraphs are available.

## StrategyQA Local Fallback

StrategyQA is not the proposal's final secondary dataset, but it remains useful locally because the repo already contains facts and decompositions. The adapter is included so the team can debug the CCS stack without waiting on Hotpot preprocessing.

## Smoke Datasets Added Here

- `data/memfaith/fever_smoke.jsonl`
- `data/memfaith/hotpot_smoke.jsonl`

These are intentionally tiny, explicit, and deterministic. They are for integration and contract validation, not for reporting final research claims.
