# MemFaith — Project Analysis and Direction Lock
**Team:** Dev Sanghvi, Ansh Dabral, Mohamad Kreidieh, Jade Yan  
**Window Covered:** through week of April 20, 2026  
**Document Type:** analysis and decision memo

---

## 1. Why this project is worth doing

MemFaith is a strong project because it sits in a useful middle ground:
- it is more technically serious than a chatbot wrapper,
- it is smaller and more buildable than training a brand-new long-context model,
- it has a direct bridge to Dev’s prior EF work,
- and it can produce both a measurable metric and a visible system.

It also has a built-in narrative that professors understand quickly:
- long contexts are expensive,
- not every chunk really matters,
- we want causal evidence, not just semantic relevance.

---

## 2. The two competing versions that have appeared so far

During planning, two versions of the project appeared.

### Version A — EF / ROME / memory-faithfulness evaluation
This is the direct descendant of the earlier EF project.
It asks:
- if we modify the model’s internal belief, does the answer follow the edit or the retained memory?

Strength:
- conceptually clean and causally strong

Weakness:
- expensive,
- hard to scale,
- difficult to finish as the main semester implementation in the remaining time.

### Version B — chunk ablation / CCS / optional pruning extension
This is the current practical version.
It asks:
- if we remove one chunk from the long context, does the answer change?

Strength:
- much more buildable,
- easier to scale across many examples,
- easier to connect to future pruning.

Weakness:
- weaker intervention than full parametric editing,
- needs careful explanation so it does not sound like plain RAG.

---

## 3. Recommended final scope

Use **Version B as the main implementation**.
Use **Version A only as historical motivation**.

That means the final cleaned project should read as:

> We use chunk ablation to estimate chunk-level causal contribution in long-context QA / fact verification, quantify it with CCS, and optionally learn a lightweight model to approximate that expensive signal.

This makes the project coherent, finishable, and still connected to the prior EF work.

---

## 4. Why the chunk-ablation version is enough

A chunk-ablation system can still answer meaningful questions:
- which chunks matter at all,
- whether causal importance changes with segmentation depth,
- whether single-hop and multi-hop tasks behave differently,
- whether chunk-level causal labels can support efficient pruning.

It also gives visible outputs:
- CCS tables,
- CCS vs K curves,
- per-example flip maps,
- qualitative examples,
- optional pruned-context results.

That is enough for a semester-scale implementation and proposal.

---

## 5. The central ambiguity you must keep under control

The biggest structural danger is mixing these three ideas without clear boundaries:
1. memory agents,
2. chunk ablation,
3. trained chunk discriminator.

The correct hierarchy is:

- **memory agents** = motivation and long-context framing
- **chunk ablation** = semester core method
- **discriminator** = optional extension

If those are presented as if they are all equally central, the project will sound blurry.

---

## 6. What makes the project strong academically

### 6.1 Clear intervention
Removing a chunk is a direct intervention on available evidence.

### 6.2 Clean metric
CCS is simple enough to explain and measurable enough to compare across settings.

### 6.3 Good benchmark choice
FEVER gives clean labels and controlled claims.
HotpotQA introduces harder multi-hop structure.

### 6.4 Built-in analysis depth
The project naturally supports:
- label stratification,
- context-length analysis,
- chunk-position analysis,
- single-hop vs multi-hop comparison.

---

## 7. What could make the project weak

### 7.1 If answer-comparison is sloppy
If answer flips are measured badly, CCS becomes meaningless.

### 7.2 If chunking is arbitrary and unstable
Then differences in CCS might reflect segmentation artifacts, not causal structure.

### 7.3 If the team overcommits to the discriminator
That could dilute the core result and leave both the main method and extension unfinished.

### 7.4 If the slides keep mixing EF and CCS
Then the audience will not know what the actual semester project is.

---

## 8. What should absolutely be finished by April 20

Minimum complete project:
- FEVER adapter
- long-context constructor
- chunker
- full-context inference
- leave-one-chunk-out runner
- CCS computation
- CCS tables and plots
- at least one HotpotQA pilot comparison
- written analysis and case studies

Everything beyond that is extra.

---

## 9. Best extension if extra time remains

The best extension is not more theory. It is:

**a lightweight chunk scorer trained on chunk-level causal labels**

Reason:
- directly connected to the project’s long-term value,
- easy to explain,
- gives an efficiency story,
- does not require abandoning the core CCS pipeline.

---

## 10. What to say if asked “is this just RAG?”

No.
RAG stores or retrieves evidence by semantic relevance.
This project measures **causal necessity** of chunks by intervention.

The important distinction is:
- RAG asks what text looks relevant,
- MemFaith asks what text was actually necessary for the answer.

---

## 11. Final decision memo

### Keep
- FEVER
- HotpotQA
- GPT2-XL as primary baseline
- CCS as main metric
- chunk ablation as core method

### Demote to optional
- Qwen extension
- discriminator training
- any EF/ROME reimplementation inside the semester core

### Remove from main story
- terminology that makes it sound like a teacher-student system unless that extension is actually built
- slide text that treats EF as the main current metric

---

## 12. One-sentence final framing

**MemFaith is a chunk-ablation-based framework for measuring whether long-context language models rely on genuinely causal evidence, with an optional extension toward learned context pruning.**
