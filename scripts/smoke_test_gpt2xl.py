"""Quick sanity check that the local GPT-2 XL weights can generate FEVER-style answers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path("models/gpt2")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    prompt = (
        "Claim: Michael Jordan plays the sport of basketball.\n"
        "Question: Is this claim SUPPORTED, REFUTED, or NOT ENOUGH INFO?\n"
        "Answer:"
    )

    if not MODEL_DIR.exists():
        print(f"[ERROR] Expected local model directory at {MODEL_DIR}, but it was not found.", file=sys.stderr)
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as exc:  # pragma: no cover - informative failure
        print(f"[ERROR] Failed to load tokenizer from {MODEL_DIR}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    except Exception as exc:  # pragma: no cover - informative failure
        print(f"[ERROR] Failed to load model from {MODEL_DIR}: {exc}", file=sys.stderr)
        sys.exit(1)

    device = pick_device()
    try:
        model.to(device)
    except RuntimeError as exc:
        print(f"[WARN] Could not move model to {device} ({exc}); falling back to CPU.", file=sys.stderr)
        device = torch.device("cpu")
        model.to(device)

    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("Prompt:")
    print(prompt)
    print("\nModel continuation:")
    print(completion.strip())


if __name__ == "__main__":
    main()
