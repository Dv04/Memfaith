"""Deterministic long-context construction for MemFaith."""

from __future__ import annotations

from hashlib import sha256
from random import Random
from typing import List, Optional

from .schemas import BuiltContext, NormalizedExample, PlacedSegment, SourceSegment


class LongContextBuilder:
    """Construct long contexts from evidence and distractor segments."""

    def __init__(self, *, seed: int = 0, max_distractors: Optional[int] = None) -> None:
        self.seed = seed
        self.max_distractors = max_distractors

    def _rng_for(self, example: NormalizedExample) -> Random:
        seed_material = f"{self.seed}:{example.dataset}:{example.example_id}"
        digest = sha256(seed_material.encode("utf-8")).hexdigest()
        return Random(int(digest[:16], 16))

    def _select_segments(self, example: NormalizedExample) -> List[SourceSegment]:
        distractors = list(example.distractor_segments)
        if self.max_distractors is not None:
            distractors = distractors[: self.max_distractors]
        segments = list(example.evidence_segments) + distractors
        if not segments:
            raise ValueError(
                f"Example {example.example_id} has no evidence or distractor segments. "
                "Prepared datasets must include explicit segment text."
            )
        return segments

    @staticmethod
    def _render_block(position: int, segment: SourceSegment) -> str:
        title = segment.title or f"segment_{segment.segment_id}"
        role = "gold-evidence" if segment.is_gold else segment.source_type
        body = segment.text.strip()
        return "\n".join(
            [
                f"[Segment {position}]",
                f"Title: {title}",
                f"Role: {role}",
                body,
            ]
        ).strip()

    def build(self, example: NormalizedExample) -> BuiltContext:
        segments = self._select_segments(example)
        indices = list(range(len(segments)))
        self._rng_for(example).shuffle(indices)

        placed_segments: List[PlacedSegment] = []
        blocks: List[str] = []
        for order_index, source_index in enumerate(indices, start=1):
            source = segments[source_index]
            block_text = self._render_block(order_index, source)
            blocks.append(block_text)
            placed_segments.append(
                PlacedSegment(
                    segment_id=source.segment_id,
                    title=source.title,
                    text=source.text,
                    is_gold=source.is_gold,
                    source_type=source.source_type,
                    order_index=order_index,
                    block_text=block_text,
                    metadata=dict(source.metadata),
                )
            )

        context_text = "\n\n".join(blocks)
        context_id = sha256(context_text.encode("utf-8")).hexdigest()
        return BuiltContext(
            example=example,
            context_text=context_text,
            ordered_segments=placed_segments,
            context_id=context_id,
        )
