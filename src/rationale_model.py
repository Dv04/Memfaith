"""Prompt utilities for generating answers and rationales with causal LMs."""

from __future__ import annotations

import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data_loading import Example, load_dataset

LABEL_PATTERN = re.compile(
    r"\b(SUPPORTS|REFUTES|NOT[\s_-]*ENOUGH[\s_-]*INFO|YES|NO)\b", re.IGNORECASE
)
LABEL_LINE_PATTERN = re.compile(
    r"^(SUPPORTS|REFUTES|NOT[\s_-]*ENOUGH[\s_-]*INFO|YES|NO)\b", re.IGNORECASE
)


@dataclass
class RationaleOutput:
    example: Example
    model_answer: Optional[str]
    normalized_answer: Optional[str]
    rationale: Optional[str]
    raw_output_text: str


def _canonicalize_label(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    normalized = text.strip().upper().replace("-", "_").replace(" ", "_")
    if normalized in {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "YES", "NO"}:
        return normalized
    return None


class RationaleLM:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_p: float = 0.95,
        batch_size: int = 4,
    ):
        self.model_name = model_name
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self._use_half = device.startswith("cuda") and torch.cuda.is_available()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.float16 if self._use_half else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype
        )
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------
    def _build_prompt_answer_only(self, ex: Example) -> str:
        if ex.dataset == "fever":
            return (
                f"Claim: {ex.input_text}\n"
                "Question: Is this claim SUPPORTED, REFUTED, or NOT ENOUGH INFO?\n"
                "Answer:"
            )
        if ex.dataset == "strategyqa":
            return f"Question: {ex.input_text}\nAnswer YES or NO.\nAnswer:"
        raise ValueError(f"Unknown dataset '{ex.dataset}'")

    def _build_prompt_answer_and_rationale(self, ex: Example) -> str:
        if ex.dataset == "fever":
            return (
                f"Claim: {ex.input_text}\n"
                "Question: Is this claim SUPPORTED, REFUTED, or NOT ENOUGH INFO?\n"
                "First, answer with a label. Then explain your reasoning.\n"
                "Answer:"
            )
        if ex.dataset == "strategyqa":
            return (
                f"Question: {ex.input_text}\n"
                "First answer YES or NO, then explain in one or two sentences.\n"
                "Answer:"
            )
        raise ValueError(f"Unknown dataset '{ex.dataset}'")

    # ------------------------------------------------------------------
    # Core generation helpers
    # ------------------------------------------------------------------
    def _generate(self, prompts: List[str]) -> List[str]:
        """Generate in mini-batches to control VRAM usage."""
        outputs: List[str] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self._use_half
                else nullcontext()
            )
            with torch.no_grad(), autocast_ctx:
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else 1.0,
                    top_p=self.top_p if self.temperature > 0 else 1.0,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend(texts)

            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return outputs

    def _extract_answer(self, text: str) -> Optional[str]:
        snippet = text
        if "Answer:" in text:
            snippet = text.split("Answer:", 1)[1].lstrip()
        first_line = snippet.splitlines()[0] if snippet else ""
        match = LABEL_LINE_PATTERN.match(first_line)
        if not match:
            match = LABEL_PATTERN.search(text)
            if not match:
                return None
        return match.group(1).upper().replace(" ", "_").replace("-", "_")

    def _parse_answer_and_rationale(
        self, text: str
    ) -> tuple[Optional[str], Optional[str]]:
        answer = self._extract_answer(text)
        rationale = None
        if "Explanation:" in text:
            rationale = text.split("Explanation:", 1)[1].strip()
        elif "Reasoning:" in text:
            rationale = text.split("Reasoning:", 1)[1].strip()
        else:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if len(lines) >= 2:
                rationale = " ".join(lines[1:])
        return answer, rationale

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def generate_answer_only(self, examples: List[Example]) -> List[RationaleOutput]:
        prompts = [self._build_prompt_answer_only(ex) for ex in examples]
        generations = self._generate(prompts)
        outputs: List[RationaleOutput] = []
        for ex, gen_text in zip(examples, generations):
            answer = self._extract_answer(gen_text)
            outputs.append(
                RationaleOutput(
                    example=ex,
                    model_answer=answer,
                    normalized_answer=_canonicalize_label(answer),
                    rationale=None,
                    raw_output_text=gen_text,
                )
            )
        return outputs

    def generate_answer_and_rationale(
        self, examples: List[Example]
    ) -> List[RationaleOutput]:
        prompts = [self._build_prompt_answer_and_rationale(ex) for ex in examples]
        generations = self._generate(prompts)
        outputs: List[RationaleOutput] = []
        for ex, gen_text in zip(examples, generations):
            answer, rationale = self._parse_answer_and_rationale(gen_text)
            outputs.append(
                RationaleOutput(
                    example=ex,
                    model_answer=answer,
                    normalized_answer=_canonicalize_label(answer),
                    rationale=rationale,
                    raw_output_text=gen_text,
                )
            )
        return outputs

    def generate_answer_only_with_model(
        self, model, examples: List[Example]
    ) -> List[RationaleOutput]:
        """Reuse tokenizer/prompts but temporarily swap in another model."""
        original_model = self.model
        original_device = self.device
        original_half = self._use_half

        tmp_device = str(next(model.parameters()).device)
        tmp_half = next(model.parameters()).dtype == torch.float16
        self.model = model
        self.device = tmp_device
        self._use_half = tmp_half and tmp_device.startswith("cuda")
        try:
            return self.generate_answer_only(examples)
        finally:
            self.model = original_model
            self.device = original_device
            self._use_half = original_half


def _demo():
    dataset = load_dataset("fever", "train")[:3]
    lm = RationaleLM("openai-community/gpt2", device="cpu", max_new_tokens=64)
    outputs = lm.generate_answer_and_rationale(dataset)
    for out in outputs:
        print(f"Claim: {out.example.input_text}")
        print(f"Answer: {out.model_answer}")
        print(f"Rationale: {out.rationale}")
        print("-" * 40)


if __name__ == "__main__":
    _demo()
