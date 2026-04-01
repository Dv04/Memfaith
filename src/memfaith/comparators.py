"""Answer comparison utilities for MemFaith CCS evaluation."""

from __future__ import annotations

from collections import Counter
from typing import Iterable
import re

from .schemas import AnswerComparison, NormalizedExample, Prediction


_LABEL_ALIASES = {
    "SUPPORT": "SUPPORTS",
    "SUPPORTS": "SUPPORTS",
    "REFUTE": "REFUTES",
    "REFUTES": "REFUTES",
    "NEI": "NOT_ENOUGH_INFO",
    "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
    "NOT ENOUGH INFO": "NOT_ENOUGH_INFO",
}

_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def _normalize_text(value: str) -> str:
    value = value or ""
    return _NON_WORD_RE.sub(" ", value.lower()).strip()


def _normalize_label(value: str) -> str:
    normalized = value.strip().upper().replace("-", "_")
    return _LABEL_ALIASES.get(normalized, normalized.replace(" ", "_"))


def _tokenize(value: str) -> Iterable[str]:
    return [token for token in _normalize_text(value).split() if token]


def token_f1(reference: str, candidate: str) -> float:
    reference_tokens = list(_tokenize(reference))
    candidate_tokens = list(_tokenize(candidate))
    if not reference_tokens and not candidate_tokens:
        return 1.0
    if not reference_tokens or not candidate_tokens:
        return 0.0

    common = Counter(reference_tokens) & Counter(candidate_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


class AnswerComparator:
    """Compare full-context and ablated answers, plus score against gold."""

    def __init__(self, *, qa_f1_threshold: float = 0.6) -> None:
        self.qa_f1_threshold = qa_f1_threshold

    def compare(self, example: NormalizedExample, baseline: Prediction, candidate: Prediction) -> AnswerComparison:
        if example.task_type == "classification":
            baseline_normalized = _normalize_label(baseline.raw_text)
            candidate_normalized = _normalize_label(candidate.raw_text)
            flipped = baseline_normalized != candidate_normalized
            return AnswerComparison(
                flipped=flipped,
                method="label_exact",
                score=1.0 if not flipped else 0.0,
                baseline_normalized=baseline_normalized,
                candidate_normalized=candidate_normalized,
            )

        baseline_normalized = _normalize_text(baseline.raw_text)
        candidate_normalized = _normalize_text(candidate.raw_text)
        if baseline_normalized == candidate_normalized:
            return AnswerComparison(
                flipped=False,
                method="qa_exact_normalized",
                score=1.0,
                baseline_normalized=baseline_normalized,
                candidate_normalized=candidate_normalized,
            )

        score = token_f1(baseline.raw_text, candidate.raw_text)
        return AnswerComparison(
            flipped=score < self.qa_f1_threshold,
            method="qa_token_f1",
            score=score,
            baseline_normalized=baseline_normalized,
            candidate_normalized=candidate_normalized,
        )

    def is_correct(self, example: NormalizedExample, prediction: Prediction) -> bool:
        if example.task_type == "classification":
            return _normalize_label(prediction.raw_text) == _normalize_label(example.gold_answer)
        return token_f1(example.gold_answer, prediction.raw_text) >= self.qa_f1_threshold
