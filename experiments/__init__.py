"""Experiment helpers for running EF evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.data_loading import Example, load_dataset
from src.editing_wrapper import EditStats, apply_memit_edits, apply_rome_edit
from src.rationale_model import RationaleLM, RationaleOutput
from src.triple_extraction import (
    FactTriple,
    build_global_triple_pool,
    extract_triples_from_rationale,
    sample_control_triple,
    score_triples_against_example,
    select_target_triple,
)


def _resolve_hparams_path(algo: str, editing_cfg: Dict[str, Any]) -> Optional[Path]:
    if editing_cfg.get("hparams_path"):
        return Path(editing_cfg["hparams_path"])
    hparams_fname = editing_cfg.get("hparams_fname")
    if not hparams_fname:
        return None
    root = Path(editing_cfg.get("unified_editing_root", "external/unified_editing"))
    return root / "hparams" / algo.upper() / hparams_fname


def _apply_edit_dispatch(model, tokenizer, triple: FactTriple, editing_cfg: Dict[str, Any]):
    algo = editing_cfg.get("algorithm", "ROME").upper()
    if algo == "MEMIT":
        return apply_memit_edits(model, tokenizer, [triple], editing_cfg)

    new_object = editing_cfg.get("new_object") or editing_cfg.get("edit_object_replacement")
    if not new_object:
        new_object = f"not {triple.object}".strip()
    hparams_path = _resolve_hparams_path("ROME", editing_cfg)

    return apply_rome_edit(
        base_model=model,
        tokenizer=tokenizer,
        triple=triple,
        new_object=new_object,
        hparams_path=hparams_path,
        device=editing_cfg.get("device"),
    )


def _run_with_model(lm: RationaleLM, model, example: Example):
    """Temporarily swap the LM model to reuse prompting utilities."""
    original_model = lm.model
    lm.model = model
    try:
        return lm.generate_answer_and_rationale([example])[0]
    finally:
        lm.model = original_model


def _triple_to_dict(triple: Optional[FactTriple]) -> Optional[Dict[str, Any]]:
    return asdict(triple) if triple else None


def run_ef_experiment(config: Dict[str, Any]) -> Path:
    dataset_name = config["dataset_name"]
    split = config.get("split", "dev")
    num_examples = config.get("num_examples")

    examples = load_dataset(dataset_name, split)
    if num_examples:
        examples = examples[: num_examples]

    model_cfg = config.get("model", {})
    lm = RationaleLM(
        model_name=model_cfg.get("name", "openai-community/gpt2"),
        device=model_cfg.get("device", "cuda"),
        max_new_tokens=model_cfg.get("max_new_tokens", 128),
        temperature=model_cfg.get("temperature", 0.2),
        top_p=model_cfg.get("top_p", 0.95),
    )

    initial_outputs = lm.generate_answer_and_rationale(examples)
    triple_pool = build_global_triple_pool(initial_outputs)
    editing_cfg = config.get("editing", {})

    log_path = Path(config.get("log_path", f"experiments/logs/{dataset_name}_{split}_ef.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_f:
        for output in initial_outputs:
            triples = extract_triples_from_rationale(output.rationale, origin_id=output.example.example_id)
            triples = score_triples_against_example(triples, output.example)
            target_triple = select_target_triple(triples)
            if target_triple is None:
                continue

            control_triple = sample_control_triple(target_triple, triple_pool, output.rationale)

            record = {
                "dataset": dataset_name,
                "example_id": output.example.example_id,
                "input_text": output.example.input_text,
                "gold_label": output.example.gold_label,
                "orig_answer": output.model_answer,
                "orig_rationale": output.rationale,
                "target_triple": _triple_to_dict(target_triple),
                "control_triple": _triple_to_dict(control_triple),
            }

            base_model = lm.model

            edited_model, edit_stats_target, restore_target = _apply_edit_dispatch(base_model, lm.tokenizer, target_triple, editing_cfg)
            try:
                post_target = _run_with_model(lm, edited_model, output.example)
            finally:
                if restore_target:
                    restore_target()

            record["post_edit_answer_target"] = post_target.model_answer
            record["post_edit_rationale_target"] = post_target.rationale
            record["edit_stats_target"] = asdict(edit_stats_target) if edit_stats_target else None

            if control_triple:
                edited_model_ctrl, edit_stats_control, restore_control = _apply_edit_dispatch(base_model, lm.tokenizer, control_triple, editing_cfg)
                try:
                    post_control = _run_with_model(lm, edited_model_ctrl, output.example)
                finally:
                    if restore_control:
                        restore_control()
                record["post_edit_answer_control"] = post_control.model_answer
                record["post_edit_rationale_control"] = post_control.rationale
                record["edit_stats_control"] = asdict(edit_stats_control) if edit_stats_control else None
            else:
                record["post_edit_answer_control"] = None
                record["post_edit_rationale_control"] = None
                record["edit_stats_control"] = None

            log_f.write(json.dumps(record) + "\n")

    return log_path
