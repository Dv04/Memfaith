# MemFaith — Master Implementation Blueprint
**Project:** Measuring Causal Faithfulness of Memory Agents in Long-Context LLMs  
**Team:** Dev Sanghvi, Ansh Dabral, Mohamad Kreidieh, Jade Yan  
**Planning Window:** Week of March 30, 2026 through week of April 20, 2026  
**Document Type:** Implementation-first project master plan

---

## 1. Project identity

This project asks a narrow but important question: when a long-context language system produces an answer, which retained chunks of context actually drove that answer, and which chunks were present but causally useless? The short-term implementation goal is to build a reproducible chunk-ablation framework that measures chunk-level causal influence with **Causal Chunk Score (CCS)**. The optional extension is to use those chunk-level causal labels to train a lightweight chunk scorer or discriminator for cheaper context pruning later.

This is not a generic RAG project and not a general summarization project. The core output is a **causal signal over chunks**, not just a better answer. The minimal successful system for this semester is:

1. construct long contexts,
2. split them into K chunks,
3. run full-context inference,
4. remove one chunk at a time,
5. measure answer changes,
6. aggregate those answer changes into CCS,
7. analyze how CCS changes as memory pressure / segmentation depth increases.

---

## 2. Final project framing

### 2.1 Main research question
Can a system identify which chunks in a long context are **causally necessary** for an answer, rather than merely semantically related to it?

### 2.2 Operational question for this semester
As context is segmented more aggressively, does the **average causal contribution of individual chunks** decrease, and can we later learn a cheap model to approximate that causal signal?

### 2.3 Deliverable split
The project should be treated in two layers.

**Layer A — required semester core**
- long-context construction
- chunk segmentation
- full-context baseline inference
- leave-one-chunk-out ablation
- CCS computation
- FEVER + one secondary dataset evaluation
- plots and case studies

**Layer B — optional extension if time remains**
- train a lightweight discriminator on chunk-level causal labels
- use it to prune low-value chunks before inference
- compare full context vs pruned context

If the team runs short on time, Layer A is enough for a complete proposal-to-prototype story.

---

## 3. Core terminology

### 3.1 Long context
A single example whose evidence and distractors are combined into one large text block that is too large to treat as a trivial short-context prompt.

### 3.2 Chunk
A contiguous segment of the long context. Each chunk is treated as one unit of intervention.

### 3.3 Segmentation depth K
The number of chunks into which the same long context is split.
- K = 0 or baseline mode means direct full-context evaluation without chunking as the reference.
- K in {2, 4, 8} means the same context is partitioned more finely.

### 3.4 Full-context baseline
The model answer using the complete long context.
- Symbol: `A_full`

### 3.5 Ablated answer
The model answer after removing chunk `i` from the context.
- Symbol: `A_-i`

### 3.6 Causal Chunk Score (CCS)
Average answer-flip probability when individual chunks are removed:

`CCS(K) = (1 / K) * sum_i Pr[A_-i != A_full]`

Interpretation:
- higher CCS means chunks have stronger average causal influence,
- lower CCS means individual chunks matter less on average.

---

## 4. Final scope decisions

### 4.1 Primary dataset
**FEVER dev**
- task: fact verification
- labels: SUPPORTS / REFUTES / NEI
- use case: controlled single-hop benchmark

### 4.2 Secondary dataset
**HotpotQA**
- task: multi-hop question answering
- use case: test whether chunk-level causal behavior changes when reasoning spans multiple paragraphs

### 4.3 Primary model
**GPT2-XL (1.5B)**
- chosen because it aligns with the prior EF project and enables direct continuity in framing and comparisons

### 4.4 Optional extension model
**Qwen2.5-1.5B**
- stronger compact baseline if time permits

### 4.5 What we are not doing in the core scope
- no full-scale ROME editing loop as the main semester system
- no real MemAgent reproduction with RL overwrite training
- no full teacher-student pruning pipeline unless Layer A stabilizes early

---

## 5. System architecture

### 5.1 Core pipeline

```text
Dataset example
  -> long-context constructor
  -> segment into K chunks
  -> full-context inference
  -> leave-one-chunk-out runs
  -> answer comparison
  -> chunk-level causal labels
  -> CCS aggregation
  -> plots, stratified analysis, case studies
```

### 5.2 Module breakdown

#### Module 1 — Data adapter
Responsibilities:
- load FEVER and HotpotQA examples
- normalize input format
- expose label fields and evidence fields

Outputs:
- `example_id`
- `question_or_claim`
- `gold_label`
- `evidence_text`
- `metadata`

#### Module 2 — Long-context constructor
Responsibilities:
- build a long context from gold evidence plus distractors
- support deterministic construction from seeds
- store exact text used for every run

Outputs:
- `context_text`
- `context_length_tokens`
- `source_segments`

#### Module 3 — Chunker
Responsibilities:
- split a fixed long context into K chunks
- preserve chunk boundaries and indices
- optionally support sentence-aware or paragraph-aware segmentation later

Outputs:
- `chunks = [chunk_1, chunk_2, ..., chunk_K]`

#### Module 4 — Inference engine
Responsibilities:
- run the chosen base model on full context
- rerun on ablated contexts
- log answer, confidence, and generation metadata

Outputs:
- `A_full`
- `A_-i`
- optional logits / confidence / explanation text

#### Module 5 — Causal comparator
Responsibilities:
- compare `A_-i` with `A_full`
- determine whether chunk `i` caused an answer flip
- aggregate flips into per-example and per-K statistics

Outputs:
- chunk flip flags
- per-example CCS contribution
- aggregated CCS

#### Module 6 — Evaluation harness
Responsibilities:
- compute CCS by K
- stratify by label / hop type
- compute full-context accuracy
- compute efficiency measurements
- generate plots and tables

Outputs:
- CCS tables
- degradation curves
- accuracy tables
- case-study bundles

#### Module 7 — Optional discriminator extension
Responsibilities:
- train a small model on chunk-level causal labels
- score chunks before inference
- evaluate whether pruning preserves answer quality

Outputs:
- chunk scorer
- pruning evaluation
- speed/quality tradeoff report

---

## 6. Data design

### 6.1 FEVER data plan
Each FEVER example should be expanded into a long context containing:
- evidence passages relevant to the claim,
- distractor passages from Wikipedia or the same retrieval pool,
- gold label,
- record of which segments are gold-evidence-bearing.

This allows both causal analysis and post-hoc case studies such as:
- a chunk was causally important and actually gold,
- a chunk was causally important but distractor-like,
- a gold chunk was present but had no causal effect.

### 6.2 HotpotQA data plan
Each HotpotQA example should retain:
- question text,
- supporting paragraphs,
- answer span / gold answer,
- paragraph titles,
- bridge or comparison style metadata if accessible.

Goal:
- compare chunk-level causality in single-hop FEVER vs multi-hop HotpotQA.

### 6.3 Unified internal schema

```json
{
  "example_id": "string",
  "dataset": "fever|hotpotqa",
  "query": "claim or question",
  "gold_label": "supports|refutes|nei|answer_text",
  "context_text": "full long context",
  "chunks": ["chunk_1", "chunk_2", "..."],
  "k": 4,
  "a_full": "model answer",
  "ablations": [
    {"chunk_id": 0, "a_minus_i": "answer", "flipped": true},
    {"chunk_id": 1, "a_minus_i": "answer", "flipped": false}
  ],
  "ccs_example": 0.5,
  "metadata": {}
}
```

---

## 7. Evaluation design

### 7.1 Primary metric
**Causal Chunk Score (CCS)**
- computed for each K
- aggregated across examples
- plotted as CCS vs K

### 7.2 Secondary metrics

#### Full-context task accuracy
Before doing causal analysis, the full-context model still needs to be reasonably functional.
- FEVER: label accuracy
- HotpotQA: exact match / F1 or simple answer-match score

#### Stratified CCS
Compute CCS by:
- FEVER label: SUPPORTS / REFUTES / NEI
- HotpotQA subtype if practical
- context length tertile
- chunk position (early / middle / late)

#### Efficiency
Track:
- number of inference calls per example,
- average runtime for K,
- context size reduction under any optional pruning experiment.

### 7.3 Core hypothesis
As K increases, the average causal contribution of any one chunk should decrease because the same context is broken into smaller units.

### 7.4 Important caveat
This does **not** mean overall answer quality must improve or worsen monotonically. It only means the average per-chunk influence can become more diffuse.

---

## 8. What counts as success by April 20

By the week of April 20, the team should be able to show:

1. a working FEVER long-context pipeline,
2. deterministic chunking for K in {2, 4, 8},
3. full-context answers and leave-one-chunk-out answers,
4. CCS numbers for FEVER,
5. at least one secondary-dataset pilot (HotpotQA),
6. a CCS vs K plot,
7. at least 6 high-quality case studies,
8. a clean project report / presentation package,
9. optional early discriminator experiment if Layer A finishes ahead of schedule.

---

## 9. Four-week sprint plan

### Week of March 30
**Theme:** freeze scope and build the first working end-to-end loop

Deliverables:
- finalized data schema
- FEVER adapter
- long-context constructor v1
- chunker v1
- GPT2-XL inference wrapper
- first `A_full` logging run

### Week of April 6
**Theme:** chunk-ablation infrastructure and CCS computation

Deliverables:
- leave-one-chunk-out runner
- answer comparison logic
- CCS calculator
- FEVER first-pass CCS results
- first accuracy sanity-check table

### Week of April 13
**Theme:** secondary dataset + analysis layer

Deliverables:
- HotpotQA adapter or pilot subset
- stratified CCS analysis
- context-length / chunk-position analysis
- first complete plots
- first curated qualitative cases

### Week of April 20
**Theme:** stabilize, document, and optionally extend

Deliverables:
- cleaned final figures and tables
- final work-division outputs completed
- integrated documentation pack
- optional lightweight discriminator prototype if core pipeline is stable

---

## 10. Risks and mitigation

### Risk 1 — Long-context construction is noisy
Mitigation:
- make construction deterministic,
- preserve source passage IDs,
- cap maximum distractor count for first runs.

### Risk 2 — Model output comparison becomes brittle
Mitigation:
- normalize answers before flip comparison,
- use label-mapping rules for FEVER,
- keep semantic matching rules for HotpotQA simple and explicit.

### Risk 3 — Too many ablation runs become expensive
Mitigation:
- start on small subsets,
- cache outputs aggressively,
- parallelize by chunk.

### Risk 4 — Slide/deck confusion between EF and CCS
Mitigation:
- explicitly frame EF as the prior project,
- frame CCS as the semester project core,
- mention discriminator only as extension unless implemented.

### Risk 5 — The team drifts into two projects
Mitigation:
- lock the main delivery to chunk-ablation CCS,
- treat learned pruning as optional.

---

## 11. Reproducibility checklist

- pinned environment file
- deterministic seeds
- deterministic context-construction script
- saved JSONL of all full-context and ablation runs
- saved chunk boundaries for every example
- one script to regenerate CCS tables
- one script to regenerate plots
- one folder for qualitative examples with raw text and predictions

---

## 12. Final recommendation

The best clean story for this semester is:

**Primary project:** chunk-ablation-based causal faithfulness analysis for long-context models  
**Optional extension:** lightweight chunk discriminator trained on chunk-level causal labels

That gives you one coherent implementation path, avoids the earlier EF-vs-CCS deck confusion, and still preserves a strong future extension direction.
