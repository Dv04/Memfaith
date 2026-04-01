"""Sentence-aware deterministic chunking for long contexts."""

from __future__ import annotations

from typing import List, Tuple
import re

from .schemas import BuiltContext, Chunk, ChunkedContext


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    return parts or [text]


class DeterministicChunker:
    """Chunk a built context without breaking words or obvious sentence boundaries."""

    def chunk(self, built_context: BuiltContext, k: int) -> ChunkedContext:
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0:
            return ChunkedContext(built_context=built_context, k=0, chunks=[])

        sentence_units: List[Tuple[str, int, bool]] = []
        for placed_segment in built_context.ordered_segments:
            sentences = _split_sentences(placed_segment.text)
            if not sentences:
                sentences = [placed_segment.text.strip()]
            for sentence_index, sentence in enumerate(sentences):
                if sentence_index == 0:
                    prefix = "\n".join(
                        [
                            f"[Segment {placed_segment.order_index}]",
                            f"Title: {placed_segment.title or f'segment_{placed_segment.segment_id}'}",
                            f"Role: {'gold-evidence' if placed_segment.is_gold else placed_segment.source_type}",
                        ]
                    )
                    rendered = f"{prefix}\n{sentence}".strip()
                else:
                    rendered = sentence
                sentence_units.append((rendered, placed_segment.segment_id, placed_segment.is_gold))

        if not sentence_units:
            raise ValueError(f"Context for {built_context.example.example_id} produced no chunkable text")

        target_chars = max(1, sum(len(text) for text, _, _ in sentence_units) // k)
        groups: List[List[Tuple[str, int, bool]]] = []
        current_group: List[Tuple[str, int, bool]] = []
        current_chars = 0

        for index, unit in enumerate(sentence_units):
            remaining_units = len(sentence_units) - index
            remaining_groups_after_current = max(k - len(groups) - 1, 0)
            must_flush = (
                current_group
                and current_chars >= target_chars
                and remaining_units > remaining_groups_after_current
            )
            if must_flush:
                groups.append(current_group)
                current_group = []
                current_chars = 0
            current_group.append(unit)
            current_chars += len(unit[0])

        if current_group:
            groups.append(current_group)

        while len(groups) > k:
            tail = groups.pop()
            groups[-1].extend(tail)

        while len(groups) < k:
            longest_index = max(range(len(groups)), key=lambda idx: len(groups[idx]))
            longest = groups[longest_index]
            if len(longest) < 2:
                break
            midpoint = len(longest) // 2
            groups[longest_index : longest_index + 1] = [longest[:midpoint], longest[midpoint:]]

        chunks: List[Chunk] = []
        for chunk_id, group in enumerate(groups):
            segment_ids = sorted({segment_id for _, segment_id, _ in group})
            gold_segment_ids = sorted({segment_id for _, segment_id, is_gold in group if is_gold})
            text = "\n\n".join(piece for piece, _, _ in group if piece.strip()).strip()
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    segment_ids=segment_ids,
                    gold_segment_ids=gold_segment_ids,
                )
            )

        return ChunkedContext(built_context=built_context, k=k, chunks=chunks)
