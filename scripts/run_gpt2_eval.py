#!/usr/bin/env python3
"""Run GPT-2 (small or XL) evaluation on the full-context dataset.

Evaluates each example by feeding the complete prompt (with all 30 segments)
to the model and comparing its generated answer against the gold answer.

Usage
-----
    # GPT-2 small
    python scripts/run_gpt2_eval.py --model gpt2 --device mps

    # GPT-2 XL
    python scripts/run_gpt2_eval.py --model gpt2-xl --device mps
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memfaith import (
    AnswerComparator,
    LongContextBuilder,
    load_prepared_examples,
)
from src.memfaith.prompts import build_prompt
from src.memfaith.schemas import NormalizedExample, Prediction


# ── Robust label extraction for base (non-instruction-tuned) models ──────
# GPT-2 is a base model: it doesn't follow instructions.  Instead of
# emitting "SUPPORTS", it continues generating context-like text.
# We scan the raw output for any FEVER-style label keywords.

_FEVER_LABEL_PATTERNS = [
    (re.compile(r"\bnot[\s_-]*enough[\s_-]*info(?:rmation)?\b", re.I), "NOT_ENOUGH_INFO"),
    (re.compile(r"\brefute[sd]?\b", re.I), "REFUTES"),
    (re.compile(r"\bsupport[sed]*\b", re.I), "SUPPORTS"),
]


def extract_json_field(raw_text: str, field_name: str) -> str | None:
    """Attempt to parse a JSON object from the raw text and extract a specific field."""
    # Try to find something that looks like JSON
    match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict) and field_name in data:
            return str(data[field_name]).strip()
    except json.JSONDecodeError:
        pass
    return None

def extract_fever_label(raw_text: str) -> str:
    """Search raw model output for FEVER labels."""
    # First try JSON parsing
    json_val = extract_json_field(raw_text, "label")
    if json_val and json_val.upper() in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"):
        return json_val.upper()
        
    # Fallback to regex scanning
    for pattern, label in _FEVER_LABEL_PATTERNS:
        if pattern.search(raw_text):
            return label
    return "NO_LABEL"


def extract_qa_answer(raw_text: str) -> str:
    """For QA tasks, take the first line / short phrase as the answer."""
    # First try JSON parsing
    json_val = extract_json_field(raw_text, "answer")
    if json_val:
        return json_val
        
    text = raw_text.strip()
    # Take up to the first newline or period
    first_line = text.split("\n")[0].strip()
    # If it still looks like segment continuation, give up
    if first_line.startswith("[Segment") or first_line.startswith("Title:"):
        return "NO_ANSWER"
    return first_line if first_line else "NO_ANSWER"


def normalize_prediction(example: NormalizedExample, raw_text: str) -> str:
    """Extract a meaningful answer from GPT-2's raw continuation output."""
    if example.task_type == "classification":
        return extract_fever_label(raw_text)
    return extract_qa_answer(raw_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation on full context")
    parser.add_argument("--model", type=str, default="gpt2", help="HF model name")
    parser.add_argument("--device", type=str, default="mps", help="Device: cpu, mps, or cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--context-window", type=int, default=1024, help="Context window to enforce for truncation")
    parser.add_argument("--fever-path", default="data/memfaith/counterfactual_fever.jsonl")
    parser.add_argument("--hotpot-path", default="data/memfaith/counterfactual_hotpotqa.jsonl")
    parser.add_argument("--output-path", default=None, help="Output CSV path (auto-named if omitted)")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Auto-name output based on model
    model_tag = args.model.replace("/", "_").replace("-", "_")
    if args.output_path is None:
        args.output_path = f"outputs/memfaith/{model_tag}_full_context_results.csv"

    # Load model
    print(f"Loading {args.model}...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.to(args.device)
    model.eval()
    print(f"Model loaded on {args.device}. Enforcing context window: {args.context_window}")

    # Load data
    examples = []
    for path in [args.fever_path, args.hotpot_path]:
        if Path(path).exists():
            examples.extend(load_prepared_examples(path))
    if args.max_examples:
        examples = examples[:args.max_examples]
    print(f"Loaded {len(examples)} examples")

    builder = LongContextBuilder(seed=args.seed)
    comparator = AnswerComparator()

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    total = 0
    no_label_count = 0

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "example_id", "query", "gold_answer", "task_type",
            "model", "prediction_raw", "prediction_normalized", "is_correct",
            "n_segments", "prompt_tokens", "time_seconds",
        ])

        for i, ex in enumerate(examples):
            built = builder.build(ex)
            prompt = build_prompt(ex, built.context_text)

            # Support chat templates for instruction-tuned models
            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_str = prompt

            # Tokenize and truncate to model's context window
            inputs = tokenizer(prompt_str, return_tensors="pt", truncation=True,
                             max_length=args.context_window - args.max_new_tokens)
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            prompt_tokens = inputs["input_ids"].shape[1]

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            elapsed = time.time() - t0

            generated = outputs[0][inputs["input_ids"].shape[1]:]
            decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()

            # Extract a normalized answer from the raw output
            normalized = normalize_prediction(ex, decoded)

            if normalized in ("NO_LABEL", "NO_ANSWER"):
                no_label_count += 1

            # Create a Prediction with the extracted label
            pred = Prediction(raw_text=normalized, normalized_text=normalized, metadata={})
            correct_flag = comparator.is_correct(ex, pred)
            if correct_flag:
                correct += 1
            total += 1

            n_segs = len(ex.evidence_segments) + len(ex.distractor_segments)

            writer.writerow([
                ex.dataset, ex.example_id, ex.query, ex.gold_answer,
                ex.task_type, args.model, decoded, normalized, correct_flag,
                n_segs, prompt_tokens, f"{elapsed:.2f}",
            ])

            if (i + 1) % 10 == 0 or i == 0:
                acc = correct / total * 100
                no_lbl_pct = no_label_count / total * 100
                print(f"  [{i+1}/{len(examples)}] Acc: {acc:.1f}% | No-label: {no_lbl_pct:.0f}% | Extracted: '{normalized}' | Raw: '{decoded[:50]}...' | {elapsed:.1f}s")

    acc = correct / total * 100 if total else 0
    no_lbl_pct = no_label_count / total * 100 if total else 0
    print(f"\n{'='*60}")
    print(f"FINAL: {args.model} on {total} examples")
    print(f"  Accuracy:         {acc:.1f}% ({correct}/{total})")
    print(f"  No-label outputs: {no_lbl_pct:.1f}% ({no_label_count}/{total})")
    print(f"  Output: {out}")
    print(f"{'='*60}")

    # Also write a summary JSON
    summary = {
        "model": args.model,
        "device": args.device,
        "total_examples": total,
        "correct": correct,
        "accuracy": acc,
        "no_label_count": no_label_count,
        "no_label_pct": no_lbl_pct,
        "output_path": str(out),
    }
    summary_path = out.with_suffix(".summary.json")
    with summary_path.open("w") as sf:
        json.dump(summary, sf, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
