"""Inference backends for MemFaith."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .schemas import NormalizedExample, Prediction


class HeuristicBackend:
    """Deterministic backend for smoke tests and integration validation."""

    def __init__(self, *, name: str = "heuristic-backend") -> None:
        self.name = name

    def predict(
        self,
        example: NormalizedExample,
        prompt: str,
        context_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Prediction:
        del prompt
        del context_text
        metadata = metadata or {}
        required_segment_ids = set(
            metadata.get("required_segment_ids")
            or example.metadata.get("required_segment_ids")
            or [segment.segment_id for segment in example.evidence_segments]
        )
        active_gold_segment_ids = set(metadata.get("active_gold_segment_ids") or [])
        coverage = 1.0 if not required_segment_ids else len(required_segment_ids & active_gold_segment_ids) / len(required_segment_ids)

        if example.task_type == "classification":
            if example.gold_answer.upper() == "NOT_ENOUGH_INFO":
                answer = "NOT_ENOUGH_INFO"
            elif required_segment_ids.issubset(active_gold_segment_ids):
                answer = example.gold_answer
            else:
                answer = example.metadata.get("missing_evidence_answer", "NOT_ENOUGH_INFO")
        else:
            if required_segment_ids.issubset(active_gold_segment_ids):
                answer = example.gold_answer
            else:
                answer = example.metadata.get("qa_fallback_answer", "unknown")

        return Prediction(
            raw_text=str(answer),
            normalized_text=str(answer),
            metadata={
                "backend": self.name,
                "coverage": coverage,
                "active_gold_segment_ids": sorted(active_gold_segment_ids),
            },
        )


class TransformersBackend:
    """Optional Hugging Face backend for real model runs."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        max_new_tokens: int = 32,
        name: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "TransformersBackend requires torch and transformers to be installed"
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.name = name or f"transformers:{model_name_or_path}"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.to(device)
        self.model.eval()

    def predict(
        self,
        example: NormalizedExample,
        prompt: str,
        context_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Prediction:
        del example
        del context_text
        del metadata
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return Prediction(
            raw_text=decoded,
            normalized_text=decoded,
            metadata={"backend": self.name},
        )
