"""Sanity-check script for applying a single ROME edit to GPT-2 XL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.editing_wrapper import apply_rome_edit
from src.triple_extraction import FactTriple


def _generate_completion(model, tokenizer, prompt: str, device: torch.device) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    continuation_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-edit ROME smoke test.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/gpt2",
        help="Path to the local GPT-2 XL weights.",
    )
    parser.add_argument(
        "--hparams-path",
        type=str,
        default=str(
            REPO_ROOT / "external" / "unified_editing" / "hparams" / "ROME" / "gpt2.json"
        ),
        help="ROME hyperparameters JSON file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu"
    )
    if device.type == "cpu":
        print("[WARN] Running on CPU; ROME editing will be slow.", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    model.eval()

    # Use a simple, single-token subject to avoid ROME's "multiple fill-ins" issue.
    triple = FactTriple(
        subject="Paris",
        relation="is the capital of",
        object="France",
        score=1.0,
    )

    # Counterfactual (wrong) capital for the edit
    new_object = "Germany"

    prompt = "Paris is the capital of"

    print("Prompt:", prompt)
    before = _generate_completion(model, tokenizer, prompt, device)
    print("Before edit:", before)

    edited_model, stats, restore = apply_rome_edit(
        base_model=model,
        tokenizer=tokenizer,
        triple=triple,
        new_object=new_object,
        hparams_path=args.hparams_path,
        device=str(device),
    )
    edited_model.eval()
    try:
        after = _generate_completion(edited_model, tokenizer, prompt, device)
    finally:
        restore()
    print("After edit:", after)
    print("Edit stats:", stats.raw)


if __name__ == "__main__":
    main()
