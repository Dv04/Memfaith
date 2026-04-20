# HotpotQA Flip Review

Total flips extracted: **20**

> **NOTE:** This review is for qualitative validation of the LLM-as-a-judge
> and Token-F1 comparison logic. Verify that flips represent genuine semantic
> divergence, not formatting artifacts.

## Flip 1: `hotpot-synth-0000` (K=2, Chunk 1) 🟡 **GOLD CHUNK**

**Query:** Who developed the theory that the person who painted the Mona Lisa studied?

**Gold Answer:** Isaac Newton

**Full-Context Prediction:** Isaac Newton

**Ablated Prediction (chunk 1 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 7]
Title: Isaac Newton - Supporting
Isaac Newton was a renowned mathematician.

Historical records confirm that Isaac Newton formulated the laws of motion.

This contribution significantly advanced the field.

[Segment 8]
Title: Leonardo da Vinci - Supporting
Leonardo da Vinci was a renowne
```

---

## Flip 2: `hotpot-synth-0003` (K=2, Chunk 1) 🟡 **GOLD CHUNK**

**Query:** What book did the marine biologist who studied environmental pesticides write?

**Gold Answer:** Silent Spring

**Full-Context Prediction:** Silent Spring

**Ablated Prediction (chunk 1 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 7]
Title: Rachel Carson - Supporting
Rachel Carson was a renowned marine biologist.

Historical records confirm that Rachel Carson wrote Silent Spring.

This contribution significantly advanced the field.

[Segment 8]
Title: Context 3
The speed of light in a vacuum is approximately 299,792 
```

---

## Flip 3: `hotpot-synth-0003` (K=4, Chunk 2) 🟡 **GOLD CHUNK**

**Query:** What book did the marine biologist who studied environmental pesticides write?

**Gold Answer:** Silent Spring

**Full-Context Prediction:** Silent Spring

**Ablated Prediction (chunk 2 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 7]
Title: Rachel Carson - Supporting
Rachel Carson was a renowned marine biologist.

Historical records confirm that Rachel Carson wrote Silent Spring.

This contribution significantly advanced the field.

[Segment 8]
Title: Context 3
The speed of light in a vacuum is approximately 299,792 
```

---

## Flip 4: `hotpot-synth-0004` (K=2, Chunk 0) 🟡 **GOLD CHUNK**

**Query:** Who formulated a principle about measurement uncertainty in quantum mechanics?

**Gold Answer:** Werner Heisenberg

**Full-Context Prediction:** Werner Heisenberg

**Ablated Prediction (chunk 0 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 1]
Title: Context 6
The Sahara Desert is the largest hot desert in the world.

[Segment 2]
Title: Context 3
The International Space Station orbits Earth approximately every 90 minutes.

[Segment 3]
Title: Context 7
The human body contains approximately 206 bones in the adult skeleton.

[Seg
```

---

## Flip 5: `hotpot-synth-0004` (K=4, Chunk 1) 🟡 **GOLD CHUNK**

**Query:** Who formulated a principle about measurement uncertainty in quantum mechanics?

**Gold Answer:** Werner Heisenberg

**Full-Context Prediction:** Werner Heisenberg

**Ablated Prediction (chunk 1 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 5]
Title: Werner Heisenberg - Supporting
Werner Heisenberg was a renowned physicist.

Historical records confirm that Werner Heisenberg formulated the uncertainty principle.

This contribution significantly advanced the field.

[Segment 6]
Title: Context 5
The Pacific Ocean is the largest a
```

---

## Flip 6: `hotpot-synth-0006` (K=2, Chunk 0) 🟡 **GOLD CHUNK**

**Query:** Who studied the behavior of great apes in the wild?

**Gold Answer:** Jane Goodall

**Full-Context Prediction:** Jane Goodall

**Ablated Prediction (chunk 0 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 1]
Title: Context 6
Jupiter is the largest planet in our solar system with a mass of 1.898e27 kg.

[Segment 2]
Title: Context 0
Mount Everest is the tallest mountain above sea level at 8,849 meters.

[Segment 3]
Title: Context 1
The speed of light in a vacuum is approximately 299,792 kilome
```

---

## Flip 7: `hotpot-synth-0006` (K=4, Chunk 1) 🟡 **GOLD CHUNK**

**Query:** Who studied the behavior of great apes in the wild?

**Gold Answer:** Jane Goodall

**Full-Context Prediction:** Jane Goodall

**Ablated Prediction (chunk 1 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 4]
Title: Context 4
The human body contains approximately 206 bones in the adult skeleton.

[Segment 5]
Title: Jane Goodall - Supporting
Jane Goodall was a renowned primatologist.

Historical records confirm that Jane Goodall studied chimpanzee behavior.

This contribution significantly adv
```

---

## Flip 8: `hotpot-synth-0009` (K=2, Chunk 1) 🟡 **GOLD CHUNK**

**Query:** Who originated the foundational theory of quantum mechanics?

**Gold Answer:** Max Planck

**Full-Context Prediction:** Max Planck

**Ablated Prediction (chunk 1 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 6]
Title: Context 5
The Sahara Desert is the largest hot desert in the world.

[Segment 7]
Title: Max Planck - Supporting
Max Planck was a renowned physicist.

Historical records confirm that Max Planck originated quantum theory.

This contribution significantly advanced the field.

[Segmen
```

---

## Flip 9: `hotpot-synth-0010-v1` (K=2, Chunk 1) 🟡 **GOLD CHUNK**

**Query:** Who developed the theory that the person who painted the Mona Lisa studied?

**Gold Answer:** Isaac Newton

**Full-Context Prediction:** Isaac Newton

**Ablated Prediction (chunk 1 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
Historical records confirm that Isaac Newton formulated the laws of motion.

This contribution significantly advanced the field.

[Segment 8]
Title: Context 3
The International Space Station orbits Earth approximately every 90 minutes.

[Segment 9]
Title: Leonardo da Vinci - Supporting
Leonardo da V
```

---

## Flip 10: `hotpot-synth-0010-v1` (K=4, Chunk 2) 🟡 **GOLD CHUNK**

**Query:** Who developed the theory that the person who painted the Mona Lisa studied?

**Gold Answer:** Isaac Newton

**Full-Context Prediction:** Isaac Newton

**Ablated Prediction (chunk 2 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
This contribution significantly advanced the field.

[Segment 8]
Title: Context 3
The International Space Station orbits Earth approximately every 90 minutes.

[Segment 9]
Title: Leonardo da Vinci - Supporting
Leonardo da Vinci was a renowned polymath.

Historical records confirm that Leonardo da Vi
```

---

## Flip 11: `hotpot-synth-0011-v1` (K=2, Chunk 1) 🟡 **GOLD CHUNK**

**Query:** What did the person who discovered radium also discover alongside it?

**Gold Answer:** polonium

**Full-Context Prediction:** polonium

**Ablated Prediction (chunk 1 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 6]
Title: Context 6
The International Space Station orbits Earth approximately every 90 minutes.

[Segment 7]
Title: Context 7
The Pacific Ocean is the largest and deepest ocean on Earth.

[Segment 8]
Title: Context 3
The Mariana Trench reaches a depth of about 36,000 feet below sea level.

```

---

## Flip 12: `hotpot-synth-0012-v1` (K=2, Chunk 1) 🟡 **GOLD CHUNK**

**Query:** Which field did the person who broke the Enigma code work in?

**Gold Answer:** computer science

**Full-Context Prediction:** computer science

**Ablated Prediction (chunk 1 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 6]
Title: Context 5
The periodic table organizes chemical elements by their atomic number.

[Segment 7]
Title: Alan Turing - Supporting
Alan Turing was a renowned computer scientist.

Historical records confirm that Alan Turing broke the Enigma code.

This contribution significantly advance
```

---

## Flip 13: `hotpot-synth-0013-v1` (K=2, Chunk 1) 🟡 **GOLD CHUNK**

**Query:** What book did the marine biologist who studied environmental pesticides write?

**Gold Answer:** Silent Spring

**Full-Context Prediction:** Silent Spring

**Ablated Prediction (chunk 1 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 7]
Title: Context 6
The Renaissance period lasted from the 14th to the 17th century in Europe.

[Segment 8]
Title: Rachel Carson - Supporting
Rachel Carson was a renowned marine biologist.

Historical records confirm that Rachel Carson wrote Silent Spring.

This contribution significantly a
```

---

## Flip 14: `hotpot-synth-0013-v1` (K=4, Chunk 2) 🟡 **GOLD CHUNK**

**Query:** What book did the marine biologist who studied environmental pesticides write?

**Gold Answer:** Silent Spring

**Full-Context Prediction:** Silent Spring

**Ablated Prediction (chunk 2 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 7]
Title: Context 6
The Renaissance period lasted from the 14th to the 17th century in Europe.

[Segment 8]
Title: Rachel Carson - Supporting
Rachel Carson was a renowned marine biologist.

Historical records confirm that Rachel Carson wrote Silent Spring.

This contribution significantly a
```

---

## Flip 15: `hotpot-synth-0014-v1` (K=2, Chunk 1) 🟡 **GOLD CHUNK**

**Query:** Who formulated a principle about measurement uncertainty in quantum mechanics?

**Gold Answer:** Werner Heisenberg

**Full-Context Prediction:** Werner Heisenberg

**Ablated Prediction (chunk 1 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 8]
Title: Werner Heisenberg - Supporting
Werner Heisenberg was a renowned physicist.

Historical records confirm that Werner Heisenberg formulated the uncertainty principle.

This contribution significantly advanced the field.

[Segment 9]
Title: Context 9
Mount Everest is the tallest mount
```

---

## Flip 16: `hotpot-synth-0015-v1` (K=2, Chunk 1) 🟡 **GOLD CHUNK**

**Query:** What did the inventor of the alternating current system develop?

**Gold Answer:** alternating current system

**Full-Context Prediction:** alternating current system

**Ablated Prediction (chunk 1 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 7]
Title: Context 8
Plate tectonics theory explains the movement of Earth's lithospheric plates.

[Segment 8]
Title: Context 2
The Mariana Trench reaches a depth of about 36,000 feet below sea level.

[Segment 9]
Title: Nikola Tesla - Supporting
Nikola Tesla was a renowned inventor.

Histor
```

---

## Flip 17: `hotpot-synth-0017-v1` (K=2, Chunk 1) 🟡 **GOLD CHUNK**

**Query:** What theoretical objects did the physicist in a wheelchair study?

**Gold Answer:** black holes

**Full-Context Prediction:** black holes

**Ablated Prediction (chunk 1 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 6]
Title: Context 3
Jupiter is the largest planet in our solar system with a mass of 1.898e27 kg.

[Segment 7]
Title: Context 2
The Renaissance period lasted from the 14th to the 17th century in Europe.

[Segment 8]
Title: Stephen Hawking - Supporting
Stephen Hawking was a renowned physicis
```

---

## Flip 18: `hotpot-synth-0018-v1` (K=2, Chunk 0) 🟡 **GOLD CHUNK**

**Query:** What genetic phenomenon did Barbara McClintock discover?

**Gold Answer:** genetic transposition

**Full-Context Prediction:** genetic transposition

**Ablated Prediction (chunk 0 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 1]
Title: Context 1
The human body contains approximately 206 bones in the adult skeleton.

[Segment 2]
Title: Context 3
The Sahara Desert is the largest hot desert in the world.

[Segment 3]
Title: Barbara McClintock - Supporting
Barbara McClintock was a renowned geneticist.

Historical re
```

---

## Flip 19: `hotpot-synth-0019-v1` (K=2, Chunk 0) 🟡 **GOLD CHUNK**

**Query:** Who originated the foundational theory of quantum mechanics?

**Gold Answer:** Max Planck

**Full-Context Prediction:** Max Planck

**Ablated Prediction (chunk 0 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 1]
Title: Context 0
The Great Wall of China stretches over 13,000 miles across northern China.

[Segment 2]
Title: Context 2
Photosynthesis converts carbon dioxide and water into glucose and oxygen.

[Segment 3]
Title: Context 1
Jupiter is the largest planet in our solar system with a mass 
```

---

## Flip 20: `hotpot-synth-0019-v1` (K=4, Chunk 1) 🟡 **GOLD CHUNK**

**Query:** Who originated the foundational theory of quantum mechanics?

**Gold Answer:** Max Planck

**Full-Context Prediction:** Max Planck

**Ablated Prediction (chunk 1 removed):** unknown

**Comparison Method:** `qa_token_f1` | **Score:** 0.0000

**Removed Chunk Preview:**
```
[Segment 4]
Title: Max Planck - Supporting
Max Planck was a renowned physicist.

Historical records confirm that Max Planck originated quantum theory.

This contribution significantly advanced the field.

[Segment 5]
Title: Context 8
The Amazon River is the largest river by discharge volume of water
```

---
