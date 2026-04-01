"""Prompt templates for MemFaith ablation runs."""

from __future__ import annotations

from .schemas import NormalizedExample


def build_prompt(example: NormalizedExample, context_text: str) -> str:
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
