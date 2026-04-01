"""Utilities for loading FEVER and StrategyQA into a shared Example schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional
import json


DatasetName = Literal["fever", "strategyqa"]


@dataclass
class Example:
    """Unified record consumed by the EF pipeline."""

    dataset: DatasetName
    example_id: str
    input_text: str
    gold_label: str
    evidence_text: Optional[str]
    meta: Dict[str, Any]


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


FEVER_SPLIT_FILES = {
    "train": "train.jsonl",
    "dev": "paper_dev.jsonl",
    "test": "paper_test.jsonl",
}


def load_fever(split: str, fever_dir: str = "data/fever") -> List[Example]:
    """Load FEVER split into Example objects."""

    if split not in FEVER_SPLIT_FILES:
        raise ValueError(f"Unsupported FEVER split '{split}'. Expected {list(FEVER_SPLIT_FILES)}")
    path = Path(fever_dir) / FEVER_SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"FEVER split file not found: {path}")

    examples: List[Example] = []
    for record in _read_jsonl(path):
        claim = record.get("claim", "").strip()
        label = record.get("label", "").upper().replace(" ", "_")
        if label == "NOT_ENOUGH_INFO":
            label = "NOT_ENOUGH_INFO"
        example_id = str(record.get("id", len(examples)))

        examples.append(
            Example(
                dataset="fever",
                example_id=example_id,
                input_text=claim,
                gold_label=label,
                evidence_text=None,
                meta=record,
            )
        )
    return examples


STRATEGYQA_SPLIT_FILES = {
    "train": "strategyqa_train.json",
    "test": "strategyqa_test.json",
}


def _normalize_sqa_label(answer: Any) -> str:
    if isinstance(answer, bool):
        return "YES" if answer else "NO"
    if isinstance(answer, str):
        answer = answer.strip().lower()
        if answer in {"yes", "y", "true"}:
            return "YES"
        if answer in {"no", "n", "false"}:
            return "NO"
    raise ValueError(f"Unrecognized StrategyQA answer: {answer}")


def load_strategyqa(split: str, sqa_dir: str = "data/strategyqa") -> List[Example]:
    if split not in STRATEGYQA_SPLIT_FILES:
        raise ValueError(f"Unsupported StrategyQA split '{split}'. Expected {list(STRATEGYQA_SPLIT_FILES)}")
    path = Path(sqa_dir) / STRATEGYQA_SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"StrategyQA split file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    examples: List[Example] = []
    for idx, record in enumerate(data):
        question = record.get("question", "").strip()
        answer = _normalize_sqa_label(record.get("answer"))
        evidence_chunks = []
        facts = record.get("facts") or []
        decomposition = record.get("decomposition") or []
        if isinstance(facts, list):
            evidence_chunks.extend(facts)
        if isinstance(decomposition, list):
            evidence_chunks.extend(decomposition)
        evidence_text = " | ".join(evidence_chunks) if evidence_chunks else None

        examples.append(
            Example(
                dataset="strategyqa",
                example_id=str(record.get("qid", idx)),
                input_text=question,
                gold_label=answer,
                evidence_text=evidence_text,
                meta=record,
            )
        )
    return examples


def load_dataset(name: str, split: str) -> List[Example]:
    if name == "fever":
        return load_fever(split)
    if name == "strategyqa":
        return load_strategyqa(split)
    raise ValueError(f"Unknown dataset '{name}'")


def _demo():
    print("FEVER sample:")
    fever_examples = load_fever("train")[:3]
    for ex in fever_examples:
        print(ex)
    print("\nStrategyQA sample:")
    sqa_examples = load_strategyqa("train")[:3]
    for ex in sqa_examples:
        print(ex)


if __name__ == "__main__":
    _demo()
