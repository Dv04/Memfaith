# MemFaith — Exhaustive Research & Build Plan
**Project:** Measuring Causal Faithfulness of Memory Agents in Long-Context LLMs  
**Team:** Dev Sanghvi, Ansh Dabral, Mohamad Kreidieh, Jade Yan  
**Course:** Graduate NLP Course Project Proposal  
**Planning window:** Week of March 30, 2026 through the week of April 20, 2026  
**Document type:** Implementation-first research plan and execution blueprint  

---

## What this document is

This document is the **large master planning artifact** for MemFaith. It is written in the same spirit as the ML-with-graphs planning document you shared: not as a short proposal summary, and not as a literature survey, but as an **execution-grade research plan** that answers:

1. **What exactly is the project really about?**
2. **What are the main and secondary technical tracks?**
3. **What datasets, models, and experimental protocols are worth using?**
4. **What will each week produce?**
5. **What decisions need to be locked early so the team does not drift?**
6. **What are the main risks, and what is the fallback route if something breaks?**

The goal is that after reading this document, the team should be able to start implementation without ambiguity, scope drift, or duplicated effort.

---

## One-sentence project statement

**MemFaith studies whether the context retained by a long-context memory-style LLM actually causally influences its final answer, using chunk ablation as the primary intervention and Causal Chunk Score (CCS) as the main measurement, with an optional extension that learns to prune low-value chunks using supervision derived from those causal signals.**

---

## The most important scope clarification

The current project materials contain **two related but not identical ideas**:

### Core project (must-do)
- build long-context examples
- segment them into chunks
- run full-context inference
- remove one chunk at a time
- recompute the answer
- measure whether the answer changed
- aggregate that into **Causal Chunk Score (CCS)**

### Extension project (nice-to-have)
- use chunk-level causal signals as training labels
- train a lightweight discriminator or chunk scorer
- use that scorer to prune context at inference time

### Important consequence
The **mainline project by April 20 should be the CCS / chunk-ablation pipeline**.  
The discriminator/pruning system should be framed as **Extension Track A**, not as the core deliverable, unless the mainline stabilizes early.

This matters because a four-week schedule cannot safely assume all of the following will succeed at once:
- long-context construction,
- multi-dataset adaptation,
- chunk ablation at scale,
- clean metrics,
- and a trained student/discriminator.

So the project should be managed as:

```text
Core: causal chunk analysis
Extension: learned pruning
Bonus: EF / ROME teacher signal integration
```

---

# Outputs of the work by April 20

By the end of this schedule, the team should aim to have the following concrete outputs.

## Output 1 — Reproducible CCS pipeline
A working pipeline that:
- loads examples from FEVER and at least one harder secondary dataset,
- constructs long contexts,
- segments each context into K chunks,
- runs full-context inference,
- runs leave-one-chunk-out ablations,
- computes CCS,
- stores all runs in structured logs,
- and produces tables and plots.

## Output 2 — Multi-dataset comparison
At minimum:
- **FEVER** as the controlled single-hop benchmark,
- **HotpotQA** as the multi-hop benchmark.

Stretch options:
- MuSiQue,
- StrategyQA,
- long-form RAG-style synthetic benchmark,
- or one internal synthetic long-context set.

## Output 3 — Evaluation suite
A reusable evaluation suite that computes:
- CCS,
- CCS vs K,
- stratified CCS by label / reasoning type,
- task accuracy before ablation,
- efficiency metrics,
- optional semantic agreement using an evaluator model.

## Output 4 — Analysis report
A results document that answers:
- whether CCS meaningfully differs across K,
- whether single-hop vs multi-hop tasks behave differently,
- whether some labels or reasoning types are more memory-faithful than others,
- and whether chunk-level causal signals are sparse enough to justify pruning.

## Output 5 — Optional chunk scorer prototype
If the mainline is stable early:
- build chunk-level labels from the ablation pipeline,
- train a simple discriminator/chunk scorer,
- evaluate whether it can preserve performance while reducing context.

---

# Phase 0 — Lock the project in its cleanest form

This phase is conceptual, but it prevents most downstream mistakes.

## 0.1 Canonical input-output contract

### Input
One example consists of:
- a task instance (claim or question),
- a long context built from evidence + distractors or supporting documents,
- a segmentation parameter K,
- a base model,
- and optionally a pruning module.

### Output
For the core project:
- full-context answer `A_full`,
- ablated answers `A_-i` for each chunk i,
- chunk-level flip indicators,
- CCS(K),
- supporting metadata (token counts, label, reasoning type, latency).

For the extension:
- chunk importance labels,
- chunk scorer predictions,
- performance-preservation results after pruning.

## 0.2 Non-negotiable experiment contract

Every experiment must expose these objects:

1. **Long context**  
2. **Segmentation into K chunks**  
3. **A baseline full-context run**  
4. **Ablation runs with one chunk removed**  
5. **A comparison function between answers**  
6. **A stored per-example result record**

If any of these is skipped, the experiment becomes vague and hard to compare.

## 0.3 Project identity in one line

When anyone asks what the team is building, the answer should be:

> We are building a chunk-ablation-based causal analysis pipeline for long-context LLMs, and then optionally using the resulting chunk-level supervision to learn efficient context pruning.

This wording keeps the project coherent even if the pruning extension is incomplete.

---

# Phase 1 — Problem framing and research questions

## 1.1 Main research question

> Do long-context LLMs actually rely on the specific memory chunks they retain, or do many retained chunks fail to causally influence the final answer?

## 1.2 Secondary research question

> As context is segmented more finely, does the average causal contribution of any single chunk become weaker?

## 1.3 Extension research question

> Can chunk-level causal signals be used to train a lightweight model that identifies and prunes irrelevant context while preserving downstream performance?

## 1.4 Derived hypotheses

### H1 — Dilution under finer segmentation
As K increases, average per-chunk causal influence decreases because the same information is distributed across more, smaller chunks.

### H2 — Harder reasoning creates different causal patterns
Multi-hop tasks will exhibit more distributed causal structure than single-hop tasks, meaning CCS patterns will differ across datasets.

### H3 — Only a minority of chunks matter strongly
For many examples, only a small subset of chunks should produce answer flips under ablation.

### H4 — If CCS labels are informative, pruning is possible
A chunk scorer trained on causal labels should preserve most answer quality while reducing context length.

---

# Phase 2 — Variables, definitions, and notation

## 2.1 Long context
A single context assembled from:
- gold evidence,
- retrieved or curated support passages,
- distractor passages,
- and optionally order perturbations.

## 2.2 Segmentation parameter K
K controls how the same long context is partitioned.

Recommended settings:
- `K = 0` → no chunking / full-context baseline
- `K = 2`
- `K = 4`
- `K = 8`
- optional `K = 16` only if runtime allows

## 2.3 Full-context answer
`A_full` is the answer obtained when the model sees the entire long context.

## 2.4 Ablated answer
`A_-i` is the answer obtained after removing chunk i.

## 2.5 Chunk-level causal contribution
Chunk i is considered causally influential if:

`A_-i != A_full`

In practice, the comparison function depends on task type:
- exact label change for FEVER,
- exact match / semantic match change for QA,
- optional evaluator-model agreement for free-form generation.

## 2.6 Causal Chunk Score (CCS)
CCS is the average answer-flip rate under leave-one-chunk-out ablation.

For a given K:

`CCS(K) = (1 / K) * sum_i Pr[A_-i != A_full]`

Interpretation:
- high CCS → chunks individually matter more
- low CCS → individual chunks matter less, or reasoning is more distributed / redundant

## 2.7 Memory pressure
Use a simple operational definition:

`memory_pressure = total_document_tokens / max_context_budget`

This helps stratify examples by how hard the long-context regime is.

## 2.8 Efficiency
Measure:
- ablation runtime per example,
- tokens processed,
- total number of model calls,
- and, for the extension, performance retained at a given pruning ratio.

---

# Phase 3 — Project architecture options

This project has multiple possible technical realizations. The team should lock a primary architecture and keep the others as extensions.

## Option A — Measurement-only CCS pipeline (recommended mainline)

### Description
- build long contexts
- split into K chunks
- run full-context inference
- ablate one chunk at a time
- compute CCS
- analyze by dataset and label

### Why this should be the mainline
- directly aligned with slides 4–7 of the proposal deck
- conceptually clean
- measurable even if the pruning extension slips
- strongest chance of finishing on time

### Deliverables
- CCS implementation
- experiment logs
- plots and analysis

## Option B — CCS pipeline + lightweight chunk scorer (recommended extension)

### Description
- generate chunk-level labels from the measurement pipeline
- train a lightweight discriminator / chunk scorer
- use it to predict which chunks are safe to prune

### Why this is worth doing
- turns measurement into a system contribution
- aligns with the “causal-driven context pruning” story already present in the deck
- more interesting than analysis alone if it stabilizes

### Why it is an extension, not the core
- needs chunk-level labels first
- requires feature design and training
- doubles the evaluation burden

## Option C — EF / ROME-assisted teacher signal (bonus only)

### Description
- use fact-level edits as a richer causal teacher signal
- connect prior EF work to chunk-level labels

### Why it is not in the critical path
- expensive
- complicated
- your own earlier work already showed this is noisy and engineering-heavy

### Best use
- bonus section in writeup
- future work
- or one tiny pilot experiment

---

# Phase 4 — Datasets and protocol map

The deck currently commits to FEVER + HotpotQA, which is a good primary pair. The research plan should support that pair cleanly while leaving room for one optional backup dataset if something breaks.

## 4.1 Primary dataset: FEVER

### Why FEVER stays
- simple label space
- controlled fact-verification setting
- easy to score answer changes
- aligns with prior EF work
- good for chunk-level causal analysis because labels are clean

### Core task
Claim verification with labels:
- SUPPORTS
- REFUTES
- NEI

### Long-context construction plan
For each example:
- include gold evidence passages,
- append Wikipedia distractors,
- randomize distractor order under fixed seed,
- segment into K chunks.

### What FEVER gives you
- strong single-hop benchmark
- clear answer comparison function
- low ambiguity for CCS definition

### What FEVER does not give you
- multi-hop reasoning
- rich narrative chains
- realistic long-document memory by default

## 4.2 Secondary dataset: HotpotQA

### Why HotpotQA is the best secondary choice
- multi-hop reasoning
- multiple supporting facts across paragraphs
- naturally tests distributed evidence use

### Role in the project
HotpotQA tells you whether chunk-level causal analysis behaves differently when answers depend on multiple evidence pieces rather than one local fact.

### Long-context construction plan
For each example:
- include supporting paragraphs,
- add distractor paragraphs,
- keep hop difficulty metadata if available,
- segment into K chunks,
- use EM / F1 or evaluator-based answer comparison.

## 4.3 Optional backup datasets

### MuSiQue
Use if you want compositional multi-hop QA with stronger reasoning separation than HotpotQA.

### StrategyQA
Use if you want question-level binary decisions with reasoning, but it is less naturally long-context than HotpotQA.

### Synthetic memory-stress benchmark
Construct a small internal benchmark where:
- one chunk contains decisive evidence,
- several chunks are distractors,
- and some examples contain redundant or conflicting support.

This is useful for debugging the pipeline before expensive full runs.

## 4.4 Unified internal sample schema

Every processed example should become one record like:

```json
{
  "id": "example_id",
  "dataset": "fever|hotpotqa|...",
  "query": "claim_or_question",
  "gold_label": "...",
  "context_chunks": ["chunk_1", "chunk_2", "chunk_3"],
  "k": 4,
  "full_answer": "...",
  "ablated_answers": ["...", "...", "..."],
  "flip_mask": [1, 0, 1, 0],
  "ccs": 0.5,
  "token_counts": {...},
  "latency": {...},
  "metadata": {...}
}
```

That schema should be stable across datasets.

---

# Phase 5 — Base model and model roster

The proposal currently names GPT2-XL as the primary model and Qwen2.5-1.5B as an extension. That is a reasonable practical pair.

## 5.1 Model selection principles

Pick models that satisfy at least one of the following:
- continuity with prior EF baseline,
- compact enough to run many ablations,
- instruction-following or QA-friendly behavior,
- local deployment feasible.

## 5.2 Primary model: GPT2-XL

### Why use it
- continuity with earlier EF project
- direct comparison story is easy
- known model-editing ecosystem if needed later

### Risks
- old model
- weaker instruction following
- may underperform on QA compared to newer compact instruct models

## 5.3 Secondary model: Qwen2.5-1.5B

### Why use it
- stronger small-model baseline
- more realistic modern compact LLM
- better chance of stable long-context reasoning than GPT2-XL on some prompts

### Risks
- results may not be directly comparable with the old EF setup
- may require different prompt handling

## 5.4 Optional evaluator model

Use a separate model only for answer comparison on free-form outputs, not as a central system component.

Possible use cases:
- semantic agreement judgment
- answer-correctness scoring when exact match is too brittle

This should remain a support tool, not part of the main architecture.

---

# Phase 6 — Experimental protocols

This is where the project stops being a general idea and becomes a concrete plan.

## 6.1 Protocol A — CCS computation

For each dataset example and each K:
1. build long context
2. segment into K chunks
3. run model on full context → `A_full`
4. for each chunk i:
   - remove chunk i
   - rerun model → `A_-i`
   - compare with `A_full`
5. compute per-example CCS
6. aggregate across dataset

## 6.2 Protocol B — CCS vs K curve

For each K in `{0, 2, 4, 8}`:
- compute aggregate CCS,
- plot CCS as a function of K,
- stratify by dataset and label.

## 6.3 Protocol C — Stratified analysis

### FEVER stratification
- SUPPORTS
- REFUTES
- NEI

### HotpotQA stratification
- reasoning type if available
- hop difficulty if available
- answer length bucket if useful

### Shared stratifications
- short / medium / long context
- low / medium / high memory pressure
- high vs low redundancy in evidence

## 6.4 Protocol D — Pruning extension

If extension is activated:
1. use chunk ablation results to create labels such as:
   - causal chunk
   - non-causal chunk
   - or scalar causal weight
2. train a lightweight scorer
3. prune low-score chunks
4. rerun the base model
5. compare:
   - accuracy retained,
   - context reduced,
   - latency saved,
   - and whether CCS-like behavior is preserved

## 6.5 Protocol E — Optional EF-aligned teacher signal

Use only if time remains.  
Try a very small pilot where fact-level EF information is mapped to chunk-level supervision, but do not make this a critical dependency.

---

# Phase 7 — Evaluation harness

The evaluation harness should exist early, before all models are finalized.

## 7.1 Primary metric

### Causal Chunk Score (CCS)
Main causal-faithfulness metric for the proposal.

## 7.2 Secondary metrics

### Task correctness before ablation
Need a sanity-check baseline that confirms the model is not failing even before interventions.

For FEVER:
- classification accuracy

For HotpotQA:
- exact match (EM)
- token-level F1 if implemented
- or evaluator-based semantic correctness if needed

### CCS vs K degradation curve
This is not just a plot — it is one of the central results.

### Context efficiency
Measure:
- number of chunks removed,
- tokens removed,
- runtime savings,
- model calls required.

### Optional pruning performance
If the chunk scorer is built:
- task performance retained at different pruning ratios
- CCS retained at different pruning ratios
- latency improvements

## 7.3 Answer comparison function

This must be decided carefully because the whole project depends on deciding whether answers changed.

### FEVER comparison
Very simple:
- compare predicted label before and after ablation.

### HotpotQA comparison options
#### Option 1 — exact match only
Simple but brittle.

#### Option 2 — EM + token F1
Standard QA metric; better than EM alone.

#### Option 3 — evaluator model
Use only when output style is variable and exact metrics are too noisy.

Recommended plan:
- use standard metrics first,
- keep evaluator model only as backup support.

## 7.4 Analysis outputs that must exist

The project is not done unless these are produced:
- aggregate CCS by K
- per-dataset CCS table
- per-label CCS table
- example-level case studies
- error cases where chunks expected to matter do not flip the answer
- and examples where unexpected chunks flip the answer

---

# Phase 8 — Implementation architecture

This project is mostly experimental NLP, but the engineering must still be clean.

## 8.1 Required software components

### Data builders
- FEVER long-context builder
- HotpotQA long-context builder
- distractor appender
- chunker

### Inference runners
- full-context runner
- ablation runner
- batch experiment runner

### Comparison functions
- FEVER label comparison
- HotpotQA answer comparison
- optional evaluator-model wrapper

### Storage
- JSONL or parquet experiment logs
- cached prompts and responses
- cached chunked contexts

### Analysis utilities
- CCS calculator
- stratification scripts
- plotting scripts
- error case extractor

### Optional learning extension
- chunk-level label builder
- lightweight discriminator trainer
- pruning evaluator

## 8.2 Cache everything that is expensive

Because the schedule is short, do not recompute expensive things unnecessarily.

Cache at least:
- chunked contexts
- full-context answers
- ablated answers
- experiment metadata
- token counts
- evaluator judgments if used

## 8.3 A minimal experiment database schema

One row per `(example_id, dataset, K, chunk_i)` containing:
- full answer
- ablated answer
- flip flag
- latency
- token counts
- gold label
- chunk position
- chunk length
- dataset metadata

This makes later analysis much easier than parsing logs.

---

# Phase 9 — Four-week aggressive schedule

Because the remaining window is only through the week of April 20, the schedule must be designed as a **compressed, decision-driven plan**.

## Week of March 30 — Lock scope, build core pipeline

### Deliverables
- final scope lock: CCS pipeline is core, discriminator is extension
- FEVER long-context builder
- HotpotQA long-context builder
- stable chunking utilities
- baseline inference runner for full-context runs
- result schema finalized
- first end-to-end smoke test on a tiny sample

### Decision checkpoint
By the end of this week the team must decide:
- exact K values
- exact answer comparison function for HotpotQA
- whether Qwen2.5-1.5B is definitely in scope or only backup

## Week of April 6 — Main ablation runs on FEVER

### Deliverables
- leave-one-chunk-out ablation runner
- FEVER experiments across all K values
- CCS computation scripts
- first CCS tables and plots
- first failure-case analysis

### Goal
Have one complete, reproducible experiment on FEVER.

## Week of April 13 — Secondary dataset + deeper analysis

### Deliverables
- HotpotQA adaptation stabilized
- CCS on HotpotQA across selected K values
- stratified analysis by label / reasoning type
- comparison between FEVER and HotpotQA
- evaluation suite finalized

### Optional extension trigger
If the pipeline is stable and the team is ahead, begin the chunk scorer / discriminator extension now.

## Week of April 20 — Extension, polish, writeup, presentation pack

### Deliverables
Core deliverables:
- final tables and plots
- polished case studies
- reproducibility scripts
- final proposal-report text
- final slide support materials

Optional extension deliverables:
- prototype chunk scorer
- pruning-vs-performance curve
- token reduction and runtime benefits

### Rule for this week
Do not open new research directions. Only polish, stabilize, write, and package.

---

# Phase 10 — Work division for four people

This section is intentionally high-level; the dedicated work-division document should go into more detail.

## Dev Sanghvi
### Primary ownership
- overall technical lead
- CCS experiment logic
- long-context experiment design
- alignment with prior EF work
- final analysis and interpretation

### Must own
- correctness of the core causal-faithfulness logic
- answer comparison strategy decisions
- final hypothesis framing

## Ansh Dabral
### Primary ownership
- implementation support for the core pipeline
- experiment orchestration
- ablation execution at scale
- logging, storage, and runtime hygiene

### Must own
- stable automated runs
- cache format
- run reproducibility

## Mohamad Kreidieh
### Primary ownership
- extension track
- chunk-level label generation utilities
- lightweight discriminator / scorer design
- pruning experiments if time allows

### Must own
- extension only after core stabilizes
- no drift into replacing the core method

## Jade Yan
### Primary ownership
- dataset curation and protocol documentation
- FEVER / HotpotQA adaptation details
- evaluation tables
- writeup structure and presentation consistency

### Must own
- keeping FEVER and HotpotQA pipelines comparable
- making the proposal/report text match the actual experiment

---

# Phase 11 — Risks and contingency routes

## Risk 1 — HotpotQA answer comparison becomes too messy
### Symptom
Exact match is too brittle, or answer forms vary too much.

### Mitigation
- use EM + F1 first
- fall back to evaluator-model agreement only if necessary
- if still unstable, reduce HotpotQA to a smaller clean subset

## Risk 2 — Full ablation is too expensive
### Symptom
Too many model calls, slow runs.

### Mitigation
- limit K values initially
- sample subset first
- cache aggressively
- restrict to smaller evaluation slices before scaling

## Risk 3 — Discriminator extension steals too much time
### Symptom
Main CCS pipeline not stable, but extension work starts anyway.

### Mitigation
- hard rule: extension starts only after FEVER CCS pipeline is complete
- keep extension branch separate from mainline branch

## Risk 4 — GPT2-XL is too weak or awkward for the current setup
### Symptom
Poor answer quality or unstable prompting.

### Mitigation
- preserve GPT2-XL for continuity experiments only
- move Qwen2.5-1.5B into the mainline if it is clearly better and still lightweight enough

## Risk 5 — Proposal materials drift away from implementation
### Symptom
Slides describe discriminator/EF training while experiments are actually ablation-only.

### Mitigation
- maintain one explicit statement in every doc:
  - core = CCS pipeline
  - extension = learned pruning

---

# Phase 12 — Reproducibility checklist

By the end of the project, you should have:

- a fixed environment file
- deterministic data splits
- a single config for each experimental family
- a stored schema for all full-context and ablation results
- scripts to regenerate tables and plots
- one script to reproduce FEVER experiments
- one script to reproduce HotpotQA experiments
- one optional script for the discriminator extension
- a small hand-checked case-study set for presentation

---

# Phase 13 — Decision checklist before implementation starts

These questions must be answered explicitly.

## 13.1 K settings
- keep K in `{0,2,4,8}` only?
- add 16 or not?

## 13.2 Base model policy
- GPT2-XL only?
- GPT2-XL + Qwen?
- or Qwen as mainline and GPT2-XL only for continuity?

## 13.3 HotpotQA comparison policy
- exact match only?
- EM + F1?
- evaluator fallback?

## 13.4 Extension gate
What conditions must be satisfied before the discriminator extension starts?

Recommended answer:
- FEVER CCS complete
- HotpotQA CCS pilot complete
- analysis scripts stable

## 13.5 Final deliverable priority
If time runs out, what matters most?

Recommended order:
1. FEVER CCS pipeline
2. HotpotQA comparison
3. analysis and writeup
4. chunk scorer extension
5. EF/ROME bonus integration

---

# Phase 14 — Final recommended execution spine

If I reduce the whole plan to the cleanest possible execution sequence, it is this:

1. Build FEVER long-context + chunking pipeline  
2. Build HotpotQA long-context + chunking pipeline  
3. Run full-context inference  
4. Run leave-one-chunk-out ablations  
5. Compute CCS  
6. Plot CCS vs K  
7. Stratify by label / reasoning type  
8. Write findings  
9. If stable, train lightweight chunk scorer  

That is the shortest route from proposal to credible result.

---

# The one-line summary for the team

> The project is not “train a discriminator.” The project is “measure chunk-level causal faithfulness in long-context LLMs using CCS,” and if that works early enough, then train a discriminator on top of those causal labels.

