"""Prompt templates for MemFaith ablation runs."""

from __future__ import annotations

from .schemas import NormalizedExample


# ── Few-shot examples for base (non-instruction-tuned) models ────────────
# GPT-2 and GPT-2 XL are base LMs that do not follow instructions.
# Providing 3 worked examples teaches them the expected output format
# through in-context pattern matching.

_FEVER_FEW_SHOT = """Example 1:
Claim: Zarkov Lenn was a spectral dynamics researcher.
Context:
[Segment 1]
Title: Zarkov Lenn - Evidence
Zarkov Lenn was a renowned spectral dynamics researcher at Northwell Polytechnical Institute.
Answer: SUPPORTS

Example 2:
Claim: Tava Brek was born in Solkaris.
Context:
[Segment 1]
Title: Tava Brek - Evidence
Tava Brek was born in 1842 in Dunhaven, Ostovia.
Answer: REFUTES

Example 3:
Claim: Holen Wick published a controversial paper on theoretical applications.
Context:
[Segment 1]
Title: Holen Wick - Evidence
Holen Wick was a renowned lattice geophysics researcher at Braxthorn Research Foundation.
Answer: NOT_ENOUGH_INFO

Now answer the following:
"""

_QA_FEW_SHOT = """Example 1:
Question: What field did the researcher who shared an institution with Tava Brek specialize in?
Context:
[Segment 1]
Title: Evidence 0
Zarkov Lenn was a renowned spectral dynamics researcher at Northwell Polytechnical Institute.
[Segment 2]
Title: Evidence 1
Zarkov Lenn and Tava Brek both conducted research at Northwell Polytechnical Institute.
Answer: spectral dynamics

Example 2:
Question: Who was born earlier, Zarkov Lenn or Tava Brek?
Context:
[Segment 1]
Title: Evidence 0
Zarkov Lenn was born in 1830 in Talmera, Ostovia.
[Segment 2]
Title: Evidence 1
Tava Brek was born in 1842 in Dunhaven, Ostovia.
Answer: Zarkov Lenn

Now answer the following:
"""


def build_prompt(example: NormalizedExample, context_text: str) -> str:
    """Build the prompt for a given example."""
    if example.task_type == "classification":
        return "\n\n".join(
            [
                "You are a careful fact verification assistant.",
                "Read the context and answer using exactly one label: SUPPORTS, REFUTES, or NOT_ENOUGH_INFO.",
                f"Claim: {example.query}",
                "Context:",
                context_text,
                "Answer:",
            ]
        )
    return "\n\n".join(
        [
            "You are a careful question answering assistant.",
            "Answer the question in a short phrase using the provided context only.",
            f"Question: {example.query}",
            "Context:",
            context_text,
            "Answer:",
        ]
    )
