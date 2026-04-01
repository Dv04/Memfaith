"""Main MemFaith CCS runner."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional
import json

from .cache import SQLitePredictionCache
from .chunking import DeterministicChunker
from .comparators import AnswerComparator
from .context_builder import LongContextBuilder
from .prompts import build_prompt
from .schemas import ChunkedContext, NormalizedExample, Prediction


class CCSRunner:
    """Run full-context and leave-one-chunk-out MemFaith experiments."""

    def __init__(
        self,
        *,
        backend,
        comparator: Optional[AnswerComparator] = None,
        builder: Optional[LongContextBuilder] = None,
        chunker: Optional[DeterministicChunker] = None,
        cache: Optional[SQLitePredictionCache] = None,
    ) -> None:
        self.backend = backend
        self.comparator = comparator or AnswerComparator()
        self.builder = builder or LongContextBuilder()
        self.chunker = chunker or DeterministicChunker()
        self.cache = cache

    def _cache_key(
        self,
        *,
        example: NormalizedExample,
        k: int,
        removed_chunk_id: Optional[int],
        prompt: str,
        active_segment_ids: List[int],
    ) -> Optional[str]:
        if self.cache is None:
            return None
        return self.cache.build_key(
            {
                "backend": getattr(self.backend, "name", type(self.backend).__name__),
                "dataset": example.dataset,
                "example_id": example.example_id,
                "k": k,
                "removed_chunk_id": removed_chunk_id,
                "active_segment_ids": active_segment_ids,
                "prompt": prompt,
            }
        )

    def _predict(
        self,
        *,
        example: NormalizedExample,
        chunked_context: ChunkedContext,
        removed_chunk_id: Optional[int],
    ) -> Prediction:
        context_text = (
            chunked_context.built_context.context_text
            if chunked_context.k == 0
            else chunked_context.render(removed_chunk_id=removed_chunk_id)
        )
        active_segment_ids = (
            sorted({segment.segment_id for segment in example.evidence_segments + example.distractor_segments})
            if chunked_context.k == 0
            else chunked_context.active_segment_ids(removed_chunk_id=removed_chunk_id)
        )
        active_gold_segment_ids = (
            sorted({segment.segment_id for segment in example.evidence_segments})
            if chunked_context.k == 0
            else chunked_context.active_gold_segment_ids(removed_chunk_id=removed_chunk_id)
        )
        prompt = build_prompt(example, context_text)
        cache_key = self._cache_key(
            example=example,
            k=chunked_context.k,
            removed_chunk_id=removed_chunk_id,
            prompt=prompt,
            active_segment_ids=active_segment_ids,
        )
        if cache_key is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return Prediction(
                    raw_text=cached.get("raw_text", ""),
                    normalized_text=cached.get("normalized_text"),
                    metadata=dict(cached.get("metadata") or {}),
                )

        prediction = self.backend.predict(
            example=example,
            prompt=prompt,
            context_text=context_text,
            metadata={
                "k": chunked_context.k,
                "removed_chunk_id": removed_chunk_id,
                "active_segment_ids": active_segment_ids,
                "active_gold_segment_ids": active_gold_segment_ids,
                "required_segment_ids": example.metadata.get("required_segment_ids"),
            },
        )
        if cache_key is not None:
            self.cache.put(cache_key, prediction.to_dict())
        return prediction

    def run_record(self, example: NormalizedExample, k: int) -> Dict:
        built_context = self.builder.build(example)
        chunked_context = self.chunker.chunk(built_context, k)
        full_prediction = self._predict(
            example=example,
            chunked_context=chunked_context,
            removed_chunk_id=None,
        )
        record = {
            "dataset": example.dataset,
            "example_id": example.example_id,
            "query": example.query,
            "gold_answer": example.gold_answer,
            "task_type": example.task_type,
            "k": k,
            "backend": getattr(self.backend, "name", type(self.backend).__name__),
            "context": built_context.to_dict(),
            "chunks": chunked_context.to_dict()["chunks"],
            "full_context": {
                "prediction": full_prediction.to_dict(),
                "is_correct": self.comparator.is_correct(example, full_prediction),
            },
            "ablations": [],
            "ccs_example": None,
            "metadata": dict(example.metadata),
        }

        if k == 0:
            return record

        flip_flags: List[int] = []
        for chunk in chunked_context.chunks:
            ablated_prediction = self._predict(
                example=example,
                chunked_context=chunked_context,
                removed_chunk_id=chunk.chunk_id,
            )
            comparison = self.comparator.compare(example, full_prediction, ablated_prediction)
            flip_flags.append(int(comparison.flipped))
            record["ablations"].append(
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.text,
                    "segment_ids": chunk.segment_ids,
                    "gold_segment_ids": chunk.gold_segment_ids,
                    "prediction": ablated_prediction.to_dict(),
                    "comparison_to_full": comparison.to_dict(),
                }
            )

        record["ccs_example"] = sum(flip_flags) / len(flip_flags) if flip_flags else 0.0
        return record

    def run(
        self,
        examples: Iterable[NormalizedExample],
        *,
        k_values: List[int],
        output_path: str,
    ) -> List[Dict]:
        records: List[Dict] = []
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", encoding="utf-8") as handle:
            for example in examples:
                for k in k_values:
                    record = self.run_record(example, k)
                    handle.write(json.dumps(record) + "\n")
                    records.append(record)
        return records
