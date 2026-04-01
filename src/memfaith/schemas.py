"""Dataclasses for the MemFaith chunk-ablation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


TaskType = Literal["classification", "qa"]


@dataclass
class SourceSegment:
    """One evidence or distractor unit before context assembly."""

    segment_id: int
    title: str
    text: str
    is_gold: bool
    source_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedExample:
    """Unified example used across datasets and runners."""

    dataset: str
    example_id: str
    query: str
    gold_answer: str
    task_type: TaskType
    evidence_segments: List[SourceSegment] = field(default_factory=list)
    distractor_segments: List[SourceSegment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "example_id": self.example_id,
            "query": self.query,
            "gold_answer": self.gold_answer,
            "task_type": self.task_type,
            "evidence_segments": [segment.to_dict() for segment in self.evidence_segments],
            "distractor_segments": [segment.to_dict() for segment in self.distractor_segments],
            "metadata": self.metadata,
        }


@dataclass
class PlacedSegment:
    """A source segment after deterministic ordering into the long context."""

    segment_id: int
    title: str
    text: str
    is_gold: bool
    source_type: str
    order_index: int
    block_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BuiltContext:
    """Long context and the ordered source segments used to build it."""

    example: NormalizedExample
    context_text: str
    ordered_segments: List[PlacedSegment]
    context_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example.example_id,
            "context_id": self.context_id,
            "context_text": self.context_text,
            "ordered_segments": [segment.to_dict() for segment in self.ordered_segments],
        }


@dataclass
class Chunk:
    """One intervention unit in the ablation loop."""

    chunk_id: int
    text: str
    segment_ids: List[int]
    gold_segment_ids: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkedContext:
    """Chunked view of a long context at one segmentation depth K."""

    built_context: BuiltContext
    k: int
    chunks: List[Chunk]

    def active_segment_ids(self, removed_chunk_id: Optional[int] = None) -> List[int]:
        segment_ids = set()
        for chunk in self.chunks:
            if removed_chunk_id is not None and chunk.chunk_id == removed_chunk_id:
                continue
            segment_ids.update(chunk.segment_ids)
        return sorted(segment_ids)

    def active_gold_segment_ids(self, removed_chunk_id: Optional[int] = None) -> List[int]:
        segment_ids = set()
        for chunk in self.chunks:
            if removed_chunk_id is not None and chunk.chunk_id == removed_chunk_id:
                continue
            segment_ids.update(chunk.gold_segment_ids)
        return sorted(segment_ids)

    def render(self, removed_chunk_id: Optional[int] = None) -> str:
        parts = [
            chunk.text.strip()
            for chunk in self.chunks
            if removed_chunk_id is None or chunk.chunk_id != removed_chunk_id
        ]
        return "\n\n".join(part for part in parts if part)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }


@dataclass
class Prediction:
    """Normalized backend prediction."""

    raw_text: str
    normalized_text: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerComparison:
    """Output of comparing the ablated answer with the full-context answer."""

    flipped: bool
    method: str
    score: float
    baseline_normalized: Optional[str] = None
    candidate_normalized: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
