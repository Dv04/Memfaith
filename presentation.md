# Graduate Seminar Presentation (COMP 640 – Explainable NLP)

**Course:** COMP 640 (Explainable NLP) – Prof. Hanjie Chen  
**Presenter:** Dev Sanghvi (NetID: ds221)  
**Talk Length:** 5 minutes (excluding title + Q/A)

---

## Slide 1 – Motivation & Problem Setup (≈45s)
- “Explainability” often stops at pretty rationales; we need to know if they reflect the model’s actual knowledge.
- Pose EF as an intervention test: edit what the rationale cites; compare to a matched control edit.
- Emphasize relevance to Prof. Chen’s class: connects textual explanations to internal representations without re-training.
- Mention FEVER + GPT‑2 XL as the concrete playground (fact verification with evidence).

## Slide 2 – EF Pipeline Overview (≈45s)
- Walk through the six steps: load claim → generate rationale → extract triples → pick target/control → edit → record flips.
- Stress that triple extraction is heuristic (copula-based) but interpretable; gives us the fact to edit.
- Highlight unified-model-editing (ROME) and the idea of flipping subject–relation–object associations.
- EF metric equation: remind the audience it’s just a difference in flip probabilities.

## Slide 3 – Engineering Challenges & Solutions (≈60s)
- State the two main blockers: (1) GPT‑2 XL + ROME is memory hungry, (2) unified-editing assumed Wikipedia stats & fp32.
- Summarize the fixes: batching, per-edit restore hooks, fp16→fp32 casting, rebuilding stats on Wikitext, dtype alignment.
- Mention that each run is capped at 30 examples, so we do multiple seeds and merge logs.
- Reassure the audience the pipeline is now reproducible (precomputed stats, deterministic seeding).

## Slide 4 – Quantitative Results (≈60s)
- Show EF table: mention specific numbers (seed EF around 7–24%, merged EF ≈ 12.5% with 88 examples).
- Call out the positive EF on REFUTES and short rationales; note negative EF for SUPPORTS (possible extractor miss).
- Stress uncertainty: small sample sizes → wide CIs, but signal is consistently positive on refutes.
- Mention how ef_eval.py automates filtering (valid labels, non-null post answers) and bootstrap CIs.

## Slide 5 – Qualitative Case Studies (≈45s)
- Describe one target-only flip, one control-only flip, one both/neither example to illustrate behavior.
- Use them to explain what EF is catching vs. where it fails (spurious triples, non-local edits).
- Tie to Professor Chen’s emphasis on human inspection: notebooks support manual annotation.
- Briefly note control-only flips as motivation for improving triple extraction.

## Slide 6 – Discussion & Next Steps (≈45s)
- Recap takeaway: rationales are partially causal; EF quantifies it but there’s variance and noise.
- List limitations: heuristic extraction, single-edit locality, small per-seed batch size, reliance on GPT‑2 XL.
- Mention future work: better IE models, MEMIT batching, extending to StrategyQA/prompted CoT.
- Close with project status: pipeline + analysis done; writing + potential extensions in progress.

---

## Q/A Slide (time permitting)
- Invite questions on EF design, engineering trade-offs, or future extensions.

---

## Project Title & Summary
**Title:** *Editability–Faithfulness: Probing the Causal Grounding of Free-Text Rationales via ROME*

I built an EF evaluation pipeline that measures whether GPT‑2 XL’s free-text rationales correspond to internal factual knowledge. The system generates FEVER rationales, extracts cited triples, edits those facts with ROME, and checks whether answers flip more often than matched control edits. Across three 30-example seeds (88 valid cases), target edits flip answers ~32% of the time vs. ~19% for controls (EF ≈ 12.5%), with especially strong signals on REFUTES claims and short rationales. The project includes end-to-end tooling (data loading, triple extraction, editing wrappers, EF metrics, plots, and notebooks) and demonstrates partial causal grounding of rationales despite noisy extraction and edit locality.

---

## Presenter Notes

### Slide 1
“As we’ve discussed throughout COMP 640, a nice-sounding rationale isn’t automatically faithful. So I wanted to test whether GPT‑2 XL’s explanations actually cite the facts it uses internally. Editability–Faithfulness frames this as an intervention problem: if I rewrite a fact mentioned in the rationale, the answer should flip more often than if I rewrite a matched but unrelated fact. FEVER is the perfect test bed because we have factual claims and evidence, and GPT‑2 XL can produce both a label and a free-text explanation.”

### Slide 2
“The EF loop is straightforward once you see it. Generate the rationale, extract subject–relation–object triples with a simple copula heuristic, pick a target triple from the rationale and a control triple from another example, edit each triple with ROME, and compare how often the answer changes. The EF metric is literally Flip_target minus Flip_control. That difference is what tells us whether the cited fact is more causally important than a matched control.”

### Slide 3
“Executing that loop on GPT‑2 XL was the main engineering lift. ROME wants fp32 weights, plus cached inverse-covariance stats built on Wikipedia, and each edit naively deep copies the whole model. We fixed this by batching generation, adding per-edit restore hooks, forcing the model into fp32 during the edit and back to fp16 afterward, and rebuilding the stats on Wikitext. Even then we cap each run at 30 examples, so we do multiple seeds and merge the logs. The good news is the pipeline is now deterministic: cached stats, deterministic seeding, and shell scripts to reproduce everything.”

### Slide 4
“With the system running, we evaluated three seeds of 30 examples each. Individual seeds give EF between 7% and 24%; the merged log of 88 valid examples yields Flip_target ≈ 32%, Flip_control ≈ 19%, so EF ≈ 12.5%. The confidence interval is wide because 88 is still small, but it’s consistently positive on REFUTES claims and on shorter rationales. So EF is picking up a real signal: when we edit the fact cited in the rationale, the answer changes more often than when we edit a matched control.”

### Slide 5
“We also looked at case studies to sanity-check the metric. For example, the rationale ‘YouTube has been ranked by a Utah-based web traffic analysis company’ leads to a triple that, when edited, flips the answer, whereas the control edit doesn’t. That’s a win. But we also saw control-only flips where the extractor latched onto template text rather than the real fact, and both-flip cases showing that ROME edits can have non-local effects. These examples live in the notebook so you can annotate them yourself.”

### Slide 6
“To wrap up: EF shows that GPT‑2 XL’s rationales are partially causal, but variance is high because we rely on heuristic extraction and single-point edits. Immediate next steps would be swapping in a better relation extractor, trying MEMIT for batched edits, and scaling to other datasets like StrategyQA. For the seminar, the pipeline, metrics, and plots are done; the remaining work is writing up the results and, if time permits, exploring those extensions.”
