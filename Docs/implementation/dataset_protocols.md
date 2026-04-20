# MemFaith Dataset Protocols

## Overview

This document describes the data preparation pipeline that transforms raw
benchmark datasets into prepared JSONL files for the CCS runner.  The pipeline
uses BM25 distractor retrieval from Wikipedia to create realistic long-context
evaluation environments.

## FEVER Mainline

**Target task:** Fact verification with labels `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`.

**Preparation steps:**

1. Load FEVER v1.0 from HuggingFace (`fever`, `v1.0`, split `train`).
2. Extract claim text, gold label, and evidence page titles from each record.
3. Resolve evidence titles against a Wikipedia corpus to obtain actual passage text.
   Raw FEVER records carry page-title pointers, not sentence text — this resolution
   step is required before running CCS.
4. Retrieve BM25-ranked distractor passages from Wikipedia.  Distractors share
   vocabulary with the claim (topically similar) but do not contain the causal
   facts needed to verify it.  Gold evidence titles are excluded from retrieval
   to prevent answer leakage.
5. Output prepared JSONL with explicit `evidence_segments` and `distractor_segments`.

**Script:** `scripts/build_fever_contexts.py`

**Key parameters:**
- `--max-examples`: Number of examples to prepare (default: 500)
- `--n-distractors`: BM25 distractors per example (default: 10)
- `--wiki-passages`: Wikipedia corpus size for BM25 index (default: 50,000)
- `--label-filter`: Optional filter for SUPPORTS / REFUTES / NOT_ENOUGH_INFO

## HotpotQA Mainline

**Target task:** Short-answer multi-hop QA requiring reasoning across two or more passages.

**Preparation steps:**

1. Load HotpotQA from HuggingFace (`hotpot_qa`, `distractor`, split `train`).
2. Separate gold evidence (supporting paragraphs) from built-in context distractors
   using the `supporting_facts` field.
3. Optionally retrieve additional BM25 distractors from Wikipedia to increase memory
   pressure at higher K values.  Existing context distractors are preserved;
   BM25 distractors are appended.
4. Output prepared JSONL in the same schema as FEVER.

**Script:** `scripts/build_hotpotqa_contexts.py`

**Key parameters:**
- `--max-examples`: Number of examples (default: 500)
- `--extra-distractors`: Additional BM25 distractors beyond built-in context (default: 5)
- `--difficulty`: Optional filter for easy / medium / hard

## Counterfactual Dataset (Parametric-Memory-Free Evaluation)

**Problem:** FEVER and HotpotQA use Wikipedia facts that LLMs already know from
pre-training.  Removing gold evidence may not flip the model's answer because it
recalls the fact from parametric memory.  This corrupts the CCS causal labels.

**Solution:** Generate examples with purely fictional entities and facts that no
LLM could have seen during training.  If removing a chunk flips the answer, it is
genuinely causal — the model cannot fall back on memorized knowledge.

**Script:** `scripts/generate_counterfactual_data.py`

**How it works:**

1. `FictionalWorldBuilder` procedurally generates a self-consistent world of
   fictional researchers, institutions, discoveries, and awards from phoneme
   combination tables.  All generation is deterministic (seeded RNG).
2. `CounterfactualFEVERGenerator` creates SUPPORTS / REFUTES / NOT_ENOUGH_INFO
   claims from the fictional facts.
3. `CounterfactualHotpotQAGenerator` creates multi-hop QA questions from linked
   entity pairs (bridge and comparison question types).
4. Distractors are drawn from the same fictional world via BM25, ensuring they
   are topically similar but causally irrelevant.

**Key parameters:**
- `--seed`: RNG seed for deterministic generation (default: 42)
- `--n-entities`: Number of fictional entities to create (default: 20)
- `--fever`: Number of FEVER examples (default: 60)
- `--hotpot`: Number of HotpotQA examples (default: 50)
- `--n-distractors`: Distractors per example (default: 5)

**Why this matters for the discriminator:** Training a chunk scorer on Wikipedia-
contaminated labels teaches it that gold evidence is "non-causal" (because the LLM
answered from memory anyway).  Counterfactual labels are clean — a causal label
means the chunk genuinely drove the model's output.

## StrategyQA Local Fallback

StrategyQA is not the proposal's final secondary dataset, but it remains useful
locally because the repo already contains facts and decompositions.  The adapter
(`load_strategyqa_split`) is included for debugging the CCS stack without waiting
on FEVER/HotpotQA preprocessing.

## BM25 Distractor Retrieval

The retrieval engine (`src/memfaith/distractor_retrieval.py`) implements a
self-contained BM25 Okapi scorer with no external dependencies.

**Why BM25 distractors?**  Random text would allow the model to find gold evidence
by simple topic clustering.  BM25 retrieves passages that share vocabulary with the
query, forcing the model to actually parse and reason about the context rather than
pattern-match.

**Gold exclusion:**  Evidence titles are passed as `exclude_titles` to the retriever
so that gold passages are never returned as distractors.

**Corpus sources:**
- Local JSONL file (`load_wikipedia_corpus`) — each line is `{"title": ..., "text": ...}`
- HuggingFace Datasets (`load_wikipedia_from_huggingface`) — downloads Wikipedia directly
- Fictional world corpus (`FictionalWorldBuilder.all_fact_texts()`) — for counterfactual mode

## Deterministic Chunking

The chunker (`src/memfaith/chunking.py`) splits prepared contexts into K chunks
at sentence boundaries.

**Two modes:**
- `use_spacy=False` (default): Regex-based splitting at `.` / `!` / `?` boundaries.
  Fast but fails on abbreviations like "Dr." or "U.S."
- `use_spacy=True`: Uses spaCy `en_core_web_sm` for accurate sentence boundary
  detection.  Handles abbreviations and edge cases correctly.

**Why sentence boundaries matter:**  Splitting mid-sentence creates out-of-distribution
syntactic artifacts that can cause answer flips from grammar failure rather than
genuine causal ablation — producing false positives in the CCS metric.

## Prepared JSONL Schema

All preparation scripts output files in this format (one JSON object per line):

```json
{
  "dataset": "fever",
  "example_id": "fever-75",
  "query": "The Eiffel Tower is located in Paris.",
  "gold_answer": "SUPPORTS",
  "task_type": "classification",
  "evidence_segments": [
    {"segment_id": 0, "title": "Eiffel Tower", "text": "...", "source_type": "wikipedia_evidence"}
  ],
  "distractor_segments": [
    {"segment_id": 1000, "title": "Colosseum", "text": "...", "source_type": "bm25_distractor",
     "metadata": {"bm25_score": 1.23}}
  ],
  "metadata": {"counterfactual": false, "required_segment_ids": [0]}
}
```

This schema is consumed directly by `load_prepared_examples()` and the CCS runner.

## Smoke Datasets

- `data/memfaith/fever_smoke.jsonl` (3 examples)
- `data/memfaith/hotpot_smoke.jsonl` (2 examples)

These are intentionally tiny and deterministic — for integration testing and contract
validation only, not for reporting research results.
