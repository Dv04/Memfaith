# MemFaith Qualitative Case Studies

> **NOTE:** These case studies are drawn from **synthetic validation data**
> and serve as structural demonstrations of the analysis pipeline.
> Final case studies should be regenerated from real model outputs.

## Case 1: Distractor Chunk Causing a Flip (False Causal Signal)

*No matching example found in the current logs.*

---

## Case 2: Gold Evidence NOT Causing a Flip (Missed Causal Signal)

**Dataset:** fever | **Example ID:** `cf-fever-1` | **K:** 2

**Query:** Myreth Trel was nominated for an international distinction in a related field.

**Gold Answer:** NOT_ENOUGH_INFO

**Full-Context Prediction:** NOT_ENOUGH_INFO
(Correct: ✅)

**CCS:** 0.0

### Ablated Chunk 0
- **Contains gold evidence:** Yes
- **Flipped:** No
- **Comparison method:** `label_exact`
- **Score:** 1.0000
- **Ablated prediction:** NOT_ENOUGH_INFO

**Chunk text (truncated):**
```
[Segment 1]
Title: Myreth Trel - Evidence
Myreth Trel was born in 1876 in Vrenneth, Hexland.

[Segment 2]
Title: Distractor 2
Myreth Feld was born in 1916 in Quelside, Pentharos.

[Segment 3]
Title: Distractor 3
Myreth Laine was born in 1966 in Yendrath, Hexland.

[Segment 4]
Title: Distractor 1
Myreth Brek was born in 1938 in Zinthari, Dunmarch.
```

**Analysis:**
This case shows a chunk with gold evidence whose removal does not change the answer. The model either has redundant evidence pathways or relies on parametric priors rather than the contextual evidence.

---

## Case 3: Gold Evidence Causing a Flip (Correct Causal Detection)

**Dataset:** fever | **Example ID:** `cf-fever-0` | **K:** 2

**Query:** Myreth Trel was a dimensional geophysics researcher.

**Gold Answer:** REFUTES

**Full-Context Prediction:** REFUTES
(Correct: ✅)

**CCS:** 0.5

### Ablated Chunk 0
- **Contains gold evidence:** Yes
- **Flipped:** Yes ⚡
- **Comparison method:** `label_exact`
- **Score:** 0.0000
- **Ablated prediction:** NOT_ENOUGH_INFO

**Chunk text (truncated):**
```
[Segment 1]
Title: Distractor 3
Myreth Brek was a renowned dimensional rheology researcher at Telmara University.

[Segment 2]
Title: Distractor 2
Quelax Czek was a renowned dimensional geophysics researcher at Vyrngate University.

[Segment 3]
Title: Distractor 4
Tavova Ford was a renowned dimensional geophysics researcher at Hexworth Centre for Advanced Studies.

[Segment 4]
Title: Myreth Trel -
```

**Analysis:**
This is the ideal outcome: removing gold evidence causes the model's answer to change, confirming that this chunk is causally necessary for the model's reasoning.

---

## Case 4: Multi-Hop Distributed Causality (≥2 Chunks Flip Independently)

**Dataset:** hotpotqa | **Example ID:** `cf-hotpot-0` | **K:** 2

**Query:** Who was born earlier, Dreven Wick or Drevion Voss?

**Gold Answer:** Drevion Voss

**Full-Context Prediction:** Drevion Voss
(Correct: ✅)

**CCS:** 1.0

**Analysis:**
This multi-hop example demonstrates distributed causal necessity: removing multiple different chunks each independently causes an answer flip. This proves the model requires multiple pieces of evidence simultaneously for correct reasoning.

---

## Case 5: FEVER REFUTES with High CCS

**Dataset:** fever | **Example ID:** `cf-fever-0` | **K:** 2

**Query:** Myreth Trel was a dimensional geophysics researcher.

**Gold Answer:** REFUTES

**Full-Context Prediction:** REFUTES
(Correct: ✅)

**CCS:** 0.5

**Analysis:**
REFUTES claims tend to show higher causal dependency. This example demonstrates that the model requires stronger evidence-grounding to reject a false claim compared to confirming a true one.

---

## Case 6: FEVER SUPPORTS with Low CCS

**Dataset:** fever | **Example ID:** `cf-fever-5` | **K:** 8

**Query:** Kethara Czek was born in Quelside, Arkessa.

**Gold Answer:** SUPPORTS

**Full-Context Prediction:** SUPPORTS
(Correct: ✅)

**CCS:** 0.16666666666666666

**Analysis:**
SUPPORTS claims often exhibit lower causal scores, suggesting the model may confirm true claims using parametric priors or partial evidence rather than requiring full context.

---
