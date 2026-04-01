"""Heuristic triple extraction from free-text rationales."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

from .data_loading import Example

if TYPE_CHECKING:
    from .rationale_model import RationaleOutput


@dataclass
class FactTriple:
    subject: str
    relation: str
    object: str
    score: float = 0.0
    source_span: Optional[Tuple[int, int]] = None
    origin_example_id: Optional[str] = None


SENTENCE_REGEX = re.compile(r"[^.!?]+[.!?]?")
COPULA_REGEX = re.compile(
    r"(?P<subj>[A-Z][^,.;!?]{2,}?)\s+(?P<verb>is|was|are|were|has been|has|had|became|serves as|plays for)\s+(?P<rest>[^.;!?]+)",
    re.IGNORECASE,
)


def _sentence_spans(text: str) -> Sequence[Tuple[int, int, str]]:
    spans = []
    for match in SENTENCE_REGEX.finditer(text):
        start, end = match.span()
        sentence = text[start:end].strip()
        if sentence:
            spans.append((start, end, sentence))
    return spans


def _split_relation_object(verb: str, rest: str) -> Tuple[str, str]:
    rest = rest.strip()
    lowered = rest.lower()
    for prep in [" in ", " at ", " from ", " for ", " to ", " on ", " of ", " with "]:
        if prep in lowered:
            idx = lowered.index(prep)
            rel_suffix = rest[: idx + len(prep)].strip()
            obj = rest[idx + len(prep) :].strip(" .,:;")
            return f"{verb} {rel_suffix}".strip(), obj
    return verb.strip(), rest.strip(" .,:;")


def extract_triples_from_rationale(rationale: Optional[str], max_triples: int = 5, origin_id: Optional[str] = None) -> List[FactTriple]:
    if not rationale:
        return []
    triples: List[FactTriple] = []
    for start, end, sentence in _sentence_spans(rationale):
        for match in COPULA_REGEX.finditer(sentence):
            subj = match.group("subj").strip()
            verb = match.group("verb").strip()
            rest = match.group("rest").strip()
            if len(subj.split()) > 10 or len(rest.split()) < 1:
                continue
            relation, obj = _split_relation_object(verb, rest)
            score = 1.0 + min(len(subj.split()), 5) * 0.1
            triples.append(
                FactTriple(
                    subject=subj,
                    relation=relation,
                    object=obj,
                    score=score,
                    source_span=(start + match.start(), start + match.end()),
                    origin_example_id=origin_id,
                )
            )
    triples.sort(key=lambda t: t.score, reverse=True)
    return triples[:max_triples]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def score_triples_against_example(triples: List[FactTriple], example: Example) -> List[FactTriple]:
    if not triples:
        return []
    if not example.evidence_text:
        triples.sort(key=lambda t: t.score, reverse=True)
        return triples
    evidence_tokens = _tokens(example.evidence_text)
    for triple in triples:
        triple_tokens = _tokens(f"{triple.subject} {triple.relation} {triple.object}")
        overlap = len(triple_tokens & evidence_tokens)
        denom = len(triple_tokens) or 1
        triple.score += overlap / denom
    triples.sort(key=lambda t: t.score, reverse=True)
    return triples


def select_target_triple(triples: List[FactTriple]) -> Optional[FactTriple]:
    return triples[0] if triples else None


def build_global_triple_pool(all_outputs: List["RationaleOutput"]) -> List[FactTriple]:
    pool: List[FactTriple] = []
    for output in all_outputs:
        triples = extract_triples_from_rationale(output.rationale, origin_id=output.example.example_id)
        pool.extend(triples)
    return pool


def _relation_key(relation: str) -> str:
    parts = relation.split()
    return parts[0].lower() if parts else ""


def sample_control_triple(target: FactTriple, pool: List[FactTriple], rationale_text: Optional[str]) -> Optional[FactTriple]:
    if not pool:
        return None
    rationale_lower = (rationale_text or "").lower()
    target_key = _relation_key(target.relation)
    candidates = [
        triple
        for triple in pool
        if triple.origin_example_id != target.origin_example_id
        and target.subject.lower() not in (triple.subject.lower(), triple.object.lower())
    ]
    filtered = [
        triple
        for triple in candidates
        if triple.subject.lower() not in rationale_lower
        and triple.object.lower() not in rationale_lower
        and (_relation_key(triple.relation) == target_key or not target_key)
    ]
    pick_from = filtered or candidates
    if not pick_from:
        return None
    return random.choice(pick_from)
