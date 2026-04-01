"""Dataset adapters for the MemFaith chunk-ablation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json

from .schemas import NormalizedExample, SourceSegment


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _segment_from_dict(record: Dict[str, Any], *, default_id: int, is_gold: bool) -> SourceSegment:
    return SourceSegment(
        segment_id=int(record.get("segment_id", default_id)),
        title=str(record.get("title", "")).strip(),
        text=str(record.get("text", "")).strip(),
        is_gold=is_gold,
        source_type=str(record.get("source_type", "evidence" if is_gold else "distractor")).strip(),
        metadata=dict(record.get("metadata") or {}),
    )


def load_prepared_examples(path: str) -> List[NormalizedExample]:
    """Load a prepared JSONL dataset with explicit evidence and distractors."""

    prepared_path = Path(path)
    if not prepared_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {prepared_path}")

    examples: List[NormalizedExample] = []
    for record in _read_jsonl(prepared_path):
        evidence_segments = [
            _segment_from_dict(segment, default_id=index, is_gold=True)
            for index, segment in enumerate(record.get("evidence_segments") or [])
        ]
        distractor_segments = [
            _segment_from_dict(segment, default_id=1000 + index, is_gold=False)
            for index, segment in enumerate(record.get("distractor_segments") or [])
        ]
        examples.append(
            NormalizedExample(
                dataset=str(record.get("dataset", "prepared")).strip(),
                example_id=str(record.get("example_id")).strip(),
                query=str(record.get("query", "")).strip(),
                gold_answer=str(record.get("gold_answer", "")).strip(),
                task_type=str(record.get("task_type", "classification")).strip(),
                evidence_segments=evidence_segments,
                distractor_segments=distractor_segments,
                metadata=dict(record.get("metadata") or {}),
            )
        )
    return examples


def load_hotpotqa_json(path: str, *, max_examples: Optional[int] = None) -> List[NormalizedExample]:
    """Load a HotpotQA-style JSON file into the unified schema."""

    hotpot_path = Path(path)
    if not hotpot_path.exists():
        raise FileNotFoundError(f"HotpotQA file not found: {hotpot_path}")

    with hotpot_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("examples") or []

    examples: List[NormalizedExample] = []
    for row in payload:
        supporting_titles = {title for title, _ in row.get("supporting_facts", [])}
        evidence_segments: List[SourceSegment] = []
        distractor_segments: List[SourceSegment] = []
        segment_cursor = 0

        for context_title, sentences in row.get("context", []):
            text = " ".join(sentence.strip() for sentence in sentences if sentence).strip()
            if not text:
                continue
            segment = SourceSegment(
                segment_id=segment_cursor,
                title=str(context_title),
                text=text,
                is_gold=context_title in supporting_titles,
                source_type="supporting_context" if context_title in supporting_titles else "context",
            )
            if segment.is_gold:
                evidence_segments.append(segment)
            else:
                distractor_segments.append(segment)
            segment_cursor += 1

        examples.append(
            NormalizedExample(
                dataset="hotpotqa",
                example_id=str(row.get("_id", len(examples))),
                query=str(row.get("question", "")).strip(),
                gold_answer=str(row.get("answer", "")).strip(),
                task_type="qa",
                evidence_segments=evidence_segments,
                distractor_segments=distractor_segments,
                metadata={"supporting_titles": sorted(supporting_titles)},
            )
        )
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def load_strategyqa_split(path: str, *, max_examples: Optional[int] = None) -> List[NormalizedExample]:
    """Load StrategyQA for local smoke or fallback experiments."""

    strategyqa_path = Path(path)
    if not strategyqa_path.exists():
        raise FileNotFoundError(f"StrategyQA file not found: {strategyqa_path}")

    with strategyqa_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    examples: List[NormalizedExample] = []
    for row in payload:
        evidence_segments = [
            SourceSegment(
                segment_id=index,
                title=f"fact_{index}",
                text=str(fact).strip(),
                is_gold=True,
                source_type="fact",
            )
            for index, fact in enumerate(row.get("facts") or [])
            if str(fact).strip()
        ]
        distractor_segments = [
            SourceSegment(
                segment_id=1000 + index,
                title=f"decomposition_{index}",
                text=str(fact).strip(),
                is_gold=False,
                source_type="decomposition",
            )
            for index, fact in enumerate(row.get("decomposition") or [])
            if str(fact).strip()
        ]
        examples.append(
            NormalizedExample(
                dataset="strategyqa",
                example_id=str(row.get("qid", len(examples))),
                query=str(row.get("question", "")).strip(),
                gold_answer="YES" if bool(row.get("answer")) else "NO",
                task_type="classification",
                evidence_segments=evidence_segments,
                distractor_segments=distractor_segments,
                metadata={"term": row.get("term"), "description": row.get("description")},
            )
        )
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples
