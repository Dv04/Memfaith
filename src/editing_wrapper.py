"""Thin wrapper around unified-model-editing for ROME/MEMIT edits."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .triple_extraction import FactTriple

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIFIED_ROOT = PROJECT_ROOT / "external" / "unified_editing"
DEFAULT_HPARAMS_PATH = DEFAULT_UNIFIED_ROOT / "hparams" / "ROME" / "gpt2.json"


@dataclass
class EditStats:
    success: bool
    algo_name: str
    edit_efficacy: Optional[float] = None
    edit_specificity: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def _ensure_unified_editing_on_path(root: Path) -> None:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"Unified-model-editing repo not found at {root}. "
            "Clone it under external/unified_editing as documented in the README."
        )
    if str(root) not in sys.path:
        sys.path.append(str(root))


def _import_rome_components():
    _ensure_unified_editing_on_path(DEFAULT_UNIFIED_ROOT)
    from rome.rome_main import ROMEHyperParams, apply_rome_to_model  # type: ignore

    return ROMEHyperParams, apply_rome_to_model


def _make_restore_fn(
    model: PreTrainedModel, orig_weights: Optional[Dict[str, torch.Tensor]]
) -> Callable[[], None]:
    if not orig_weights:
        return lambda: None

    param_dict = dict(model.named_parameters())
    buffer_dict = dict(model.named_buffers())

    def restore():
        with torch.no_grad():
            for name, tensor in orig_weights.items():
                target = param_dict.get(name)
                if target is None:
                    target = buffer_dict.get(name)
                if target is None:
                    continue
                target.copy_(tensor.to(target.device))

    return restore


def apply_rome_edit(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    triple: FactTriple,
    new_object: str,
    hparams_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    copy_model: bool = False,
) -> Tuple[PreTrainedModel, EditStats, Callable[[], None]]:
    """Apply a single ROME edit using unified-model-editing."""

    if not new_object or not new_object.strip():
        raise ValueError("new_object must be a non-empty string.")

    ROMEHyperParams, apply_rome_to_model = _import_rome_components()

    resolved_hparams = Path(hparams_path or DEFAULT_HPARAMS_PATH)
    if not resolved_hparams.exists():
        raise FileNotFoundError(
            f"ROME hparams file not found at {resolved_hparams}. "
            "Pass --hparams-path pointing to a valid JSON config."
        )

    hparams = ROMEHyperParams.from_json(str(resolved_hparams))

    target_device: torch.device
    if device is not None:
        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
            target_device = torch.device("cpu")
    else:
        target_device = next(base_model.parameters()).device

    base_model.to(target_device)

    relation = (triple.relation or "").strip()
    if relation:
        prompt_template = f"{{}} {relation}"
    else:
        prompt_template = "{}"

    request = {
        "case_id": 0,
        "prompt": prompt_template,
        "subject": triple.subject,
        "target_new": {"str": new_object},
        "target_true": {"str": triple.object},
    }

    LOGGER.info(
        "Applying ROME edit: [%s] -> [%s]",
        request["prompt"],
        request["target_new"]["str"],
    )

    try:
        edited_model, orig_weights, _ = apply_rome_to_model(
            base_model,
            tokenizer,
            requests=[request],
            hparams=hparams,
            copy=copy_model,
            return_orig_weights=not copy_model,
        )
    except Exception as exc:  # pragma: no cover - passthrough for debugging
        LOGGER.exception("ROME edit failed for request %s", request)
        raise RuntimeError(f"ROME edit failed: {exc}") from exc

    edited_model.to(target_device)
    restore_fn = _make_restore_fn(edited_model, None if copy_model else orig_weights)

    stats = EditStats(
        success=True,
        algo_name="ROME",
        raw={
            "request": request,
            "hparams_path": str(resolved_hparams),
            "layers": getattr(hparams, "layers", None),
        },
    )

    return edited_model, stats, restore_fn


def apply_memit_edits(model, tokenizer, triples: List[FactTriple], config: Dict[str, Any]) -> Tuple[Any, EditStats]:
    """Apply MEMIT edits. Currently stubbed."""

    LOGGER.warning("MEMIT integration not implemented yet.")
    return model, EditStats(success=False, algo_name="MEMIT", raw={"reason": "not implemented"})


if __name__ == "__main__":  # pragma: no cover
    print("src.editing_wrapper module ready. Use apply_rome_edit for real edits.")
