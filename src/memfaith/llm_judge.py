"""LLM-as-a-judge fallback for ambiguous QA answer comparisons.

This module provides a judge interface for determining whether two
free-form answers are semantically equivalent.  When Token-F1 falls
into the ambiguous zone (below the configured threshold), the judge
makes a final call.

Current implementation:
    - ``MockLLMJudge``: deterministic heuristic-based judge (offline).
    - ``OpenAILLMJudge``: placeholder for real API calls (not yet wired).

TODO (Dev): Replace the placeholder with a real API call once  a model
endpoint is available (e.g. a local Qwen-0.5B or GPT-4o-mini API).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMJudge(ABC):
    """Abstract base for semantic equivalence judges."""

    @abstractmethod
    def judge(
        self,
        reference: str,
        candidate: str,
        query: Optional[str] = None,
    ) -> bool:
        """Return True if ``candidate`` is semantically equivalent to ``reference``."""
        ...


class MockLLMJudge(BaseLLMJudge):
    """Deterministic heuristic judge for offline / test usage.

    The mock judges equivalence using aggressive normalization and
    substring containment, which is intentionally generous.  It is
    designed to avoid false-positive flips — i.e., it defaults to
    "not flipped" when unsure.
    """

    _NON_ALNUM = re.compile(r"[^a-z0-9 ]+")

    def _normalize(self, text: str) -> str:
        return self._NON_ALNUM.sub("", text.lower()).strip()

    def judge(
        self,
        reference: str,
        candidate: str,
        query: Optional[str] = None,
    ) -> bool:
        ref = self._normalize(reference)
        cand = self._normalize(candidate)
        if not ref or not cand:
            return ref == cand
        # Exact after normalization
        if ref == cand:
            return True
        # One contains the other
        if ref in cand or cand in ref:
            return True
        # Token-set overlap ≥ 60%
        ref_tokens = set(ref.split())
        cand_tokens = set(cand.split())
        if ref_tokens and cand_tokens:
            overlap = len(ref_tokens & cand_tokens)
            union = len(ref_tokens | cand_tokens)
            if overlap / union >= 0.6:
                return True
        return False


class OpenAILLMJudge(BaseLLMJudge):
    """Placeholder for a real LLM-as-a-judge via API.

    .. warning::
        This is an unimplemented placeholder.  Calling :meth:`judge`
        will raise ``NotImplementedError`` until an API is configured.

    To activate, set the ``OPENAI_API_KEY`` environment variable and
    update the model name below.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def judge(
        self,
        reference: str,
        candidate: str,
        query: Optional[str] = None,
    ) -> bool:
        # -----------------------------------------------------------------
        # TODO: Implement real API call.  Example prompt structure:
        #
        #   system: "You are an answer equivalence judge. ..."
        #   user: f"Question: {query}\nReference: {reference}\n
        #           Candidate: {candidate}\nAre they equivalent? YES/NO"
        #
        #   Parse the response for YES / NO.
        # -----------------------------------------------------------------
        raise NotImplementedError(
            "OpenAILLMJudge is a placeholder. "
            "Set OPENAI_API_KEY or replace with a local model endpoint."
        )
