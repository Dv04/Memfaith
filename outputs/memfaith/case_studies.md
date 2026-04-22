# MemFaith Qualitative Case Studies

> **NOTE:** These case studies are drawn from **synthetic validation data**
> and serve as structural demonstrations of the analysis pipeline.
> Final case studies should be regenerated from real model outputs.

## Case 1: Distractor Chunk Causing a Flip (False Causal Signal)

*No matching example found in the current logs.*

---

## Case 2: Gold Evidence NOT Causing a Flip (Missed Causal Signal)

**Dataset:** fever | **Example ID:** `fever-synth-0000` | **K:** 2

**Query:** Albert Einstein developed the theory of relativity.

**Gold Answer:** SUPPORTS

**Full-Context Prediction:** SUPPORTS
(Correct: ✅)

**CCS:** 0.0

### Ablated Chunk 0
- **Contains gold evidence:** Yes
- **Flipped:** No
- **Comparison method:** `label_exact`
- **Score:** 1.0000
- **Ablated prediction:** SUPPORTS

**Chunk text (truncated):**
```
[Segment 1]
Title: Distractor 5
Mount Everest is the tallest mountain above sea level at 8,849 meters.

[Segment 2]
Title: Distractor 4
The Mariana Trench reaches a depth of about 36,000 feet below sea level.

[Segment 3]
Title: Distractor 2
The speed of light in a vacuum is approximately 299,792 kilometers per second.

[Segment 4]
Title: Distractor 7
The Amazon River is the largest river by disch
```

**Analysis:**
This case shows a chunk with gold evidence whose removal does not change the answer. The model either has redundant evidence pathways or relies on parametric priors rather than the contextual evidence.

---

## Case 3: Gold Evidence Causing a Flip (Correct Causal Detection)

**Dataset:** fever | **Example ID:** `fever-synth-0001` | **K:** 4

**Query:** Marie Curie never discovered radium and polonium.

**Gold Answer:** REFUTES

**Full-Context Prediction:** REFUTES
(Correct: ✅)

**CCS:** 0.25

### Ablated Chunk 1
- **Contains gold evidence:** Yes
- **Flipped:** Yes ⚡
- **Comparison method:** `label_exact`
- **Score:** 0.0000
- **Ablated prediction:** NOT_ENOUGH_INFO

**Chunk text (truncated):**
```
[Segment 4]
Title: Distractor 1
Mount Everest is the tallest mountain above sea level at 8,849 meters.

[Segment 5]
Title: Marie Curie - Evidence
Marie Curie was a renowned chemist.

Historical records confirm that Marie Curie discovered radium and polonium.

This contribution significantly advanced the field.
```

**Analysis:**
This is the ideal outcome: removing gold evidence causes the model's answer to change, confirming that this chunk is causally necessary for the model's reasoning.

---

## Case 4: Multi-Hop Distributed Causality (≥2 Chunks Flip Independently)

*No matching example found in the current logs.*

---

## Case 5: FEVER REFUTES with High CCS

**Dataset:** fever | **Example ID:** `fever-synth-0004` | **K:** 2

**Query:** Nikola Tesla never developed any alternating current system.

**Gold Answer:** REFUTES

**Full-Context Prediction:** REFUTES
(Correct: ✅)

**CCS:** 0.5

**Analysis:**
REFUTES claims tend to show higher causal dependency. This example demonstrates that the model requires stronger evidence-grounding to reject a false claim compared to confirming a true one.

---

## Case 6: FEVER SUPPORTS with Low CCS

**Dataset:** fever | **Example ID:** `fever-synth-0000` | **K:** 2

**Query:** Albert Einstein developed the theory of relativity.

**Gold Answer:** SUPPORTS

**Full-Context Prediction:** SUPPORTS
(Correct: ✅)

**CCS:** 0.0

**Analysis:**
SUPPORTS claims often exhibit lower causal scores, suggesting the model may confirm true claims using parametric priors or partial evidence rather than requiring full context.

---
