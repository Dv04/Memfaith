"""Batch-optimized MemFaith CCS runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .cache import SQLitePredictionCache
from .chunking import DeterministicChunker
from .comparators import AnswerComparator
from .context_builder import LongContextBuilder
from .prompts import build_prompt
from .schemas import ChunkedContext, NormalizedExample, Prediction


class BatchCCSRunner:
    """Run full-context and leave-one-chunk-out MemFaith experiments in massive batches."""

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
        
        # 1. First pass: collect all necessary prompts that are NOT in cache
        pending_prompts = []
        pending_work_items = []
        
        # Store built contexts to avoid repeating heavy operations
        built_data = []

        for example in examples:
            example_data = {"example": example, "k_runs": []}
            for k in k_values:
                built_context = self.builder.build(example)
                chunked_context = self.chunker.chunk(built_context, k)
                
                k_run_data = {
                    "k": k,
                    "built_context": built_context,
                    "chunked_context": chunked_context,
                    "requests": []
                }
                
                # Setup full prediction request
                context_text_full = built_context.context_text
                active_segment_ids_full = sorted({seg.segment_id for seg in example.evidence_segments + example.distractor_segments})
                active_gold_segment_ids_full = sorted({seg.segment_id for seg in example.evidence_segments})
                prompt_full = build_prompt(example, context_text_full)
                
                cache_key_full = self._cache_key(
                    example=example, k=k, removed_chunk_id=None,
                    prompt=prompt_full, active_segment_ids=active_segment_ids_full
                )
                
                req_full = {
                    "removed_chunk_id": None,
                    "prompt": prompt_full,
                    "cache_key": cache_key_full,
                    "example": example,
                    "context_text": context_text_full,
                    "metadata": {
                        "k": k,
                        "removed_chunk_id": None,
                        "active_segment_ids": active_segment_ids_full,
                        "active_gold_segment_ids": active_gold_segment_ids_full,
                        "required_segment_ids": example.metadata.get("required_segment_ids"),
                    }
                }
                k_run_data["requests"].append(req_full)
                
                if k > 0:
                    for chunk in chunked_context.chunks:
                        context_text_ablated = chunked_context.render(removed_chunk_id=chunk.chunk_id)
                        active_segment_ids_ablated = chunked_context.active_segment_ids(removed_chunk_id=chunk.chunk_id)
                        active_gold_segment_ids_ablated = chunked_context.active_gold_segment_ids(removed_chunk_id=chunk.chunk_id)
                        prompt_ablated = build_prompt(example, context_text_ablated)
                        
                        cache_key_ablated = self._cache_key(
                            example=example, k=k, removed_chunk_id=chunk.chunk_id,
                            prompt=prompt_ablated, active_segment_ids=active_segment_ids_ablated
                        )
                        req_ablated = {
                            "removed_chunk_id": chunk.chunk_id,
                            "prompt": prompt_ablated,
                            "cache_key": cache_key_ablated,
                            "example": example,
                            "context_text": context_text_ablated,
                            "metadata": {
                                "k": k,
                                "removed_chunk_id": chunk.chunk_id,
                                "active_segment_ids": active_segment_ids_ablated,
                                "active_gold_segment_ids": active_gold_segment_ids_ablated,
                                "required_segment_ids": example.metadata.get("required_segment_ids"),
                            }
                        }
                        k_run_data["requests"].append(req_ablated)
                
                example_data["k_runs"].append(k_run_data)
            built_data.append(example_data)

        # 2. Check cache and dispatch
        for example_data in built_data:
            for k_run in example_data["k_runs"]:
                for req in k_run["requests"]:
                    is_cached = False
                    if req["cache_key"] and self.cache:
                        cached = self.cache.get(req["cache_key"])
                        if cached:
                            is_cached = True
                            
                    if not is_cached:
                        pending_prompts.append(req["prompt"])
                        pending_work_items.append(req)

        # 3. Batch predict
        if pending_prompts:
            if hasattr(self.backend, "predict_batch"):
                responses = self.backend.predict_batch(pending_prompts)
            else:
                responses = []
                for p, req in zip(pending_prompts, pending_work_items):
                    # fallback
                    pred = self.backend.predict(example=req["example"], prompt=p, context_text=req["context_text"], metadata=req["metadata"])
                    responses.append(pred.raw_text)

            backend_name = getattr(self.backend, "name", type(self.backend).__name__)
            for req, response_text in zip(pending_work_items, responses):
                prediction = Prediction(
                    raw_text=response_text,
                    normalized_text=response_text,
                    metadata={"backend": backend_name}
                )
                if req["cache_key"] and self.cache:
                    self.cache.put(req["cache_key"], prediction.to_dict())
                req["prediction"] = prediction

        # 4. Construct final records
        with output.open("w", encoding="utf-8") as handle:
            for example_data in built_data:
                example = example_data["example"]
                
                for k_run in example_data["k_runs"]:
                    k = k_run["k"]
                    chunked_context = k_run["chunked_context"]
                    
                    record = {
                        "dataset": example.dataset,
                        "example_id": example.example_id,
                        "query": example.query,
                        "gold_answer": example.gold_answer,
                        "task_type": example.task_type,
                        "k": k,
                        "backend": getattr(self.backend, "name", type(self.backend).__name__),
                        "context": k_run["built_context"].to_dict(),
                        "chunks": chunked_context.to_dict()["chunks"] if k > 0 else [],
                        "full_context": {},
                        "ablations": [],
                        "ccs_example": None,
                        "metadata": dict(example.metadata),
                    }
                    
                    full_prediction = None
                    flip_flags = []
                    
                    for req in k_run["requests"]:
                        if "prediction" in req:
                            pred = req["prediction"]
                        else:
                            cached_dict = self.cache.get(req["cache_key"])
                            pred = Prediction(
                                raw_text=cached_dict.get("raw_text", ""), # type: ignore
                                normalized_text=cached_dict.get("normalized_text"), # type: ignore
                                metadata=dict(cached_dict.get("metadata") or {}) # type: ignore
                            )
                        
                        if req["removed_chunk_id"] is None:
                            full_prediction = pred
                            record["full_context"] = {
                                "prediction": pred.to_dict(),
                                "is_correct": self.comparator.is_correct(example, pred),
                            }
                        else:
                            comparison = self.comparator.compare(example, full_prediction, pred) # type: ignore
                            flip_flags.append(int(comparison.flipped))
                            chunk_id = req["removed_chunk_id"]
                            chunk_obj = next(c for c in chunked_context.chunks if c.chunk_id == chunk_id)
                            record["ablations"].append(
                                {
                                    "chunk_id": chunk_obj.chunk_id,
                                    "chunk_text": chunk_obj.text,
                                    "segment_ids": chunk_obj.segment_ids,
                                    "gold_segment_ids": chunk_obj.gold_segment_ids,
                                    "prediction": pred.to_dict(),
                                    "comparison_to_full": comparison.to_dict(),
                                }
                            )
                    
                    if k > 0:
                        record["ccs_example"] = sum(flip_flags) / len(flip_flags) if flip_flags else 0.0
                        
                    handle.write(json.dumps(record) + "\n")
                    records.append(record)
                    
        return records
