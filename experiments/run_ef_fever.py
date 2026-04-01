"""Run the EF (Editability–Faithfulness) pipeline on FEVER with real ROME edits."""

from __future__ import annotations

import argparse
import json
import sys
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.data_loading import load_dataset
from src.editing_wrapper import apply_rome_edit
from src.rationale_model import RationaleLM
from src.triple_extraction import (
    FactTriple,
    build_global_triple_pool,
    extract_triples_from_rationale,
    sample_control_triple,
    score_triples_against_example,
    select_target_triple,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EF on FEVER using ROME.")
    parser.add_argument("--model-path", type=str, default="models/gpt2")
    parser.add_argument("--split", type=str, default="dev", help="FEVER split to use.")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument(
        "--log-path",
        type=str,
        default="outputs/ef_fever_gpt2xl_rome.jsonl",
        help="Destination JSONL log file.",
    )
    parser.add_argument(
        "--hparams-path",
        type=str,
        default="external/unified_editing/hparams/ROME/gpt2.json",
        help="Path to the ROME hparams JSON.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for generation to control VRAM usage.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling.")
    return parser.parse_args()


NEGATION_MAP = {
    "yes": "no",
    "no": "yes",
    "supported": "refuted",
    "refuted": "supported",
    "supports": "refutes",
    "refutes": "supports",
    "not_enough_info": "supports",
}

SPORT_SWAP = {
    "basketball": "football",
    "football": "basketball",
    "baseball": "football",
    "soccer": "basketball",
}


def choose_counterfactual_object(triple: FactTriple) -> str:
    obj = (triple.object or "").strip()
    low = obj.lower()
    if low in NEGATION_MAP:
        replacement = NEGATION_MAP[low]
    elif low in SPORT_SWAP:
        replacement = SPORT_SWAP[low]
    elif low.startswith("not "):
        replacement = obj[4:]
    else:
        replacement = f"not {obj}".strip()
    if obj and obj[0].isupper():
        replacement = replacement.capitalize()
    return replacement


def apply_edit_and_answer(
    lm: RationaleLM,
    example,
    triple: FactTriple,
    new_object: str,
    args: argparse.Namespace,
) -> Tuple[Optional[str], dict]:
    original_dtype = next(lm.model.parameters()).dtype
    cast_back = False
    if original_dtype == torch.float16:
        lm.model = lm.model.to(torch.float32)
        lm._use_half = False  # pylint: disable=protected-access
        cast_back = True

    edited_model, stats, restore = apply_rome_edit(
        base_model=lm.model,
        tokenizer=lm.tokenizer,
        triple=triple,
        new_object=new_object,
        hparams_path=args.hparams_path,
        device=args.device,
    )
    try:
        outputs = lm.generate_answer_only_with_model(edited_model, [example])
        answer = outputs[0].model_answer
    finally:
        restore()
        if cast_back:
            lm.model = lm.model.to(original_dtype)
            lm._use_half = original_dtype == torch.float16  # pylint: disable=protected-access
    return answer, stats.raw


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    examples = load_dataset("fever", args.split)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(examples)
    examples = examples[: args.max_examples]

    lm = RationaleLM(
        model_name=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=0.95,
        batch_size=args.batch_size,
    )

    original_outputs = []
    for i in range(0, len(examples), args.batch_size):
        batch = examples[i : i + args.batch_size]
        original_outputs.extend(lm.generate_answer_and_rationale(batch))
    triple_pool = build_global_triple_pool(original_outputs)

    kept = 0
    with log_path.open("w", encoding="utf-8") as f:
        for output in original_outputs:
            if not output.rationale:
                continue

            triples = extract_triples_from_rationale(
                output.rationale, origin_id=output.example.example_id
            )
            triples = score_triples_against_example(triples, output.example)
            target_triple = select_target_triple(triples)
            if target_triple is None:
                continue

            control_triple = sample_control_triple(
                target_triple, triple_pool, output.rationale
            )

            try:
                target_new_object = choose_counterfactual_object(target_triple)
                answer_post_target, stats_target = apply_edit_and_answer(
                    lm, output.example, target_triple, target_new_object, args
                )
            except Exception as exc:  # pragma: no cover - debug path
                print(
                    f"[WARN] Target edit failed for example {output.example.example_id}: {exc}",
                    file=sys.stderr,
                )
                continue

            answer_post_control = None
            stats_control = None
            control_new_object = None
            if control_triple is not None:
                try:
                    control_new_object = choose_counterfactual_object(control_triple)
                    answer_post_control, stats_control = apply_edit_and_answer(
                        lm, output.example, control_triple, control_new_object, args
                    )
                except Exception as exc:  # pragma: no cover - debug path
                    print(
                        f"[WARN] Control edit failed for example {output.example.example_id}: {exc}",
                        file=sys.stderr,
                    )
                    answer_post_control = None
                    stats_control = None
                    control_new_object = None

            record = {
                "dataset": "fever",
                "split": args.split,
                "example_id": output.example.example_id,
                "claim": output.example.input_text,
                "label": output.example.gold_label,
                "answer_pre": output.model_answer,
                "rationale_pre": output.rationale,
                "target_triple": {
                    "subject": target_triple.subject,
                    "relation": target_triple.relation,
                    "object": target_triple.object,
                },
                "control_triple": {
                    "subject": control_triple.subject,
                    "relation": control_triple.relation,
                    "object": control_triple.object,
                }
                if control_triple
                else None,
                "target_new_object": target_new_object,
                "control_new_object": control_new_object,
                "answer_post_target": answer_post_target,
                "answer_post_control": answer_post_control,
                "edit_stats_target": stats_target,
                "edit_stats_control": stats_control,
            }

            f.write(json.dumps(record) + "\n")
            kept += 1

    print(f"Wrote {kept} EF records to {log_path}")


if __name__ == "__main__":
    main()
