# MemFaith: Measuring Causal Faithfulness of Memory Agents
**Final Presentation Storyboard & Content Outline (Expanded for 7-Minute Duration)**
*Course: COMP 640 – Explainable NLP*

---

## 🛑 Structural Note for the Team
*This outline has been explicitly expanded to fill a 7-minute academic presentation. It now includes deep-dives into related work, the mathematical definition of your causality metric (CCS), qualitative examples of the synthetic dataset, and explicit talking points to help pace your delivery. You should aim to spend roughly 30-45 seconds per slide.*

---

## Slide 1: Title Slide (15 seconds)
* **Title:** MemFaith: Measuring Causal Faithfulness of Memory Agents in Long-Context LLMs
* **Subtitle:** From Context Overload to Causal Pruning via Behavioral Distillation
* **Team:** Dev Sanghvi, Keyuan (Jade) Yan, Mohamad Kreidieh, Ansh Dabral
* **Course:** COMP 640 – Explainable NLP

---

## Slide 2: The Core Problem & Motivation (45 seconds)
* **The Long-Context Illusion:** Modern memory agents (e.g., MemAgent) process arbitrarily long documents by chunking text into a memory buffer via overwrite strategies. 
* **The Faithfulness Question:** However, a fundamental question remains: *Does the memory the agent retains actually causally drive its downstream answers, or is the model hallucinating/guessing?*
* **The Editability-Faithfulness (EF) Gap:** Prior seminar work using ROME (knowledge editing) on GPT-2-XL revealed that EF is remarkably low (≈ 12.5%). This implies that the vast majority of retained context does *not* causally drive the model's outputs. It is mostly noise.

---

## Slide 3: Related Work & Limitations (30 seconds)
* **Standard Prompt Compression (e.g., LLMLingua):** Relies on token-level perplexity. *Limitation:* It doesn't understand high-level semantic causality; it just removes "surprising" tokens.
* **Retrieval-Augmented Generation (RAG):** Relies on semantic similarity (Cosine similarity). *Limitation:* A text chunk can be semantically similar to the question but completely causally irrelevant to the answer.
* **Our Gap:** We need a metric and a pruning method that optimizes strictly for *causality*, not just similarity or perplexity.

---

## Slide 4: The Research Question & Approach (45 seconds)
* **Research Question:** *"Can a learned discriminator accurately identify and prune irrelevant context chunks based on their true causal impact, reducing context length while preserving—or enhancing—downstream performance?"*
* **Our 4-Phase Pipeline:**
  1. **Generate** a highly contaminated, long-context environment.
  2. **Ablate** chunks systematically using a Leave-One-Chunk-Out (LOCO) strategy.
  3. **Train** a discriminator via behavioral distillation to predict causal importance.
  4. **Prune** the context at inference time using the trained discriminator.

---

## Slide 5: Data Engineering: Preventing Contamination (Jade) (45 seconds)
* **The Challenge of Parametric Leakage:** If we ask an LLM "When was Barack Obama born?", it might use its pre-training weights rather than the provided context. We cannot measure causal context reliance if the model already knows the answer.
* **Counterfactual Synthetic Generation:** Developed a bespoke dataset generator utilizing fully fictional entities and facts. 
  * *Example Fact:* "Zarkov Lenn was a spectral dynamics researcher."
* **Strict Boundary Control:** Utilized spaCy for robust sentence chunking, ensuring precise segment boundaries for the ablation runner.

---

## Slide 6: High-Density Distractor Retrieval (Jade) (30 seconds)
* **The Retrieval Pipeline:** Implemented a BM25-based pipeline to retrieve 29 highly confusing, structurally similar false segments for every 1 piece of gold evidence.
* **Why BM25?:** Lexical overlap forces the LLM to do actual reasoning rather than simple keyword matching. The model must differentiate between "Zarkov Lenn" and "Zarkov Phen" in the same context window.

---

## Slide 7: Defining the Causal Contribution Score (CCS) (Dev) (45 seconds)
* **Moving Beyond EF:** Instead of editing parametric weights (ROME), we edit the *prompt* at inference time via chunk ablation.
* **The Mathematics of LOCO:**
  * Let $P(y \mid C)$ be the baseline correctness with full context $C$.
  * Let $P(y \mid C \setminus c_i)$ be the correctness when chunk $c_i$ is removed.
  * **CCS Calculation:** A chunk is labeled as strictly causal ($CCS = 1$) if its removal flips the model's prediction from correct to incorrect.
* **The Benefit:** This provides a deterministic, ground-truth label for how important a specific sentence was to the model's final reasoning step.

---

## Slide 8: Infrastructure: Scaling the Ablation (Ansh) (45 seconds)
* **The Compute Bottleneck:** Calculating CCS is incredibly expensive. For a dataset of $N$ questions and $K$ chunks, we must execute $O(K \times N)$ full-context model generations. For 11,200 chunks, this is a massive compute wall.
* **High-Throughput Engineering:**
  * **vLLM Integration:** Replaced standard HuggingFace pipelines with a highly optimized `vLLM` backend for massive parallel batching.
  * **BatchCCSRunner:** Redesigned the evaluation loop to handle full-context predictions and ablation branches in asynchronous batches.
  * **HPC Orchestration:** Engineered automated SLURM deployment scripts (`run_all.sh`) for hands-off execution across compute clusters.

---

## Slide 9: Evaluation Pipeline: "Context Overload" (Dev) (45 seconds)
* **The Experimental Setup:** We explicitly designed the experiment to overwhelm the model's context window. We loaded exactly 30 segments to hit a strict 1024-token limit.
* **Agglomerative Chunk Balancing:** Fixed standard greedy chunk-loading by introducing Agglomerative Merging, guaranteeing perfectly balanced $K$-chunk distribution.
* **Robust Label Extraction:** Engineered robust JSON-schema prompts and regex fallback parsers to forcefully extract labels (`SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`) despite model hallucinations.

---

## Slide 10: Baseline Proof: The Failure of Small Models (Dev) (45 seconds)
* **Hypothesis Verified:** Heavy distractor context completely destroys the reasoning capabilities of small-parameter models.
* **The GPT-2 Failure:** Base models (GPT-2 Small & XL) failed entirely at the formatting task, producing a 100% No-Label rate even with few-shot prompting.
* **The Qwen2.5-0.5B-Instruct Baseline:** 
  * Perfectly followed instructions (0% No-Label Rate).
  * **Accuracy Plunged to Exactly 33.1%.** 
  * **Significance:** Because this is a 3-choice classification task, 33.1% represents *pure random guessing*. The 29 distractors successfully drowned out the 1 piece of true evidence. 

---

## Slide 11: The Discriminator Solution (Mohamad) (45 seconds)
* *(Mohamad to detail the architecture)*
* **The Handoff:** We successfully generated and passed `combined_chunk_labels.csv` containing over **11,200 labeled chunk rows** (30.6% Causal, 69.4% Non-Causal) to train the discriminator.
* **Objective:** Use Behavioral Distillation to teach a smaller discriminator model to predict the CCS score of a chunk *before* it is passed to the generator.

---

## Slide 12: Post-Pruning Results (Mohamad) (45 seconds)
* *(Mohamad to show training loss graphs and evaluation accuracy)*
* **Expected Outcome:** Show the accuracy recovery curve. Demonstrate that when the Discriminator filters the 30 segments down to just the causally relevant chunks, accuracy rises from the 33.1% baseline back up to optimal levels.

---

## Slide 13: Qualitative Case Study (Mohamad or Dev) (30 seconds)
* *(Insert a real example from the dataset showing a prompt with distractors, and showing exactly which chunk the Discriminator successfully pruned away).*

---

## Slide 14: Conclusion & Future Work (30 seconds)
* **Summary:** Successfully built an end-to-end causal measurement pipeline that isolates context reliance from parametric memory.
* **Validation:** Proved that long-context distractors degrade small models to random chance.
* **Resolution:** Demonstrated that behavioral distillation (the Discriminator) effectively prunes context and restores downstream faithfulness.
* **Future Work:** Applying this framework to massive context windows (1M+ tokens) in models like Gemini 1.5 Pro, and exploring its viability as an alternative to RAG for dynamic memory agents.

---
## Slide 15: Q&A
* *Open the floor for questions from Professor Chen and the class.*
