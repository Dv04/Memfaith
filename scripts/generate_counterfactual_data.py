#!/usr/bin/env python3
"""Generate counterfactual datasets with fictional entities for CCS evaluation.

Unlike the synthetic data generator which uses real scientists (Einstein, Curie),
this script generates entirely fictional entities and facts that no LLM could
have seen during pre-training.  This eliminates the Wikipedia contamination
problem where models answer from parametric memory instead of reading the context.

Usage
-----
    python scripts/generate_counterfactual_data.py
    python scripts/generate_counterfactual_data.py --n-entities 30 --fever 100 --hotpot 80
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memfaith.counterfactual import (
    CounterfactualFEVERGenerator,
    CounterfactualHotpotQAGenerator,
    FictionalWorldBuilder,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate counterfactual datasets with fictional entities"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-entities", type=int, default=20)
    parser.add_argument("--fever", type=int, default=60, help="Number of FEVER examples")
    parser.add_argument("--hotpot", type=int, default=50, help="Number of HotpotQA examples")
    parser.add_argument("--n-distractors", type=int, default=5)
    parser.add_argument("--output-dir", default="data/memfaith")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    world = FictionalWorldBuilder(seed=args.seed, n_entities=args.n_entities)
    logger.info(
        "Built fictional world: %d entities, %d facts",
        len(world.entities),
        len(world.facts),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fever_gen = CounterfactualFEVERGenerator(world, seed=args.seed)
    fever_examples = fever_gen.generate(
        n_examples=args.fever, n_distractors=args.n_distractors,
    )
    fever_path = output_dir / "counterfactual_fever.jsonl"
    with fever_path.open("w", encoding="utf-8") as fh:
        for ex in fever_examples:
            fh.write(json.dumps(ex.to_dict()) + "\n")
    logger.info("Generated %d FEVER examples -> %s", len(fever_examples), fever_path)

    hotpot_gen = CounterfactualHotpotQAGenerator(world, seed=args.seed)
    hotpot_examples = hotpot_gen.generate(
        n_examples=args.hotpot, n_distractors=args.n_distractors,
    )
    hotpot_path = output_dir / "counterfactual_hotpotqa.jsonl"
    with hotpot_path.open("w", encoding="utf-8") as fh:
        for ex in hotpot_examples:
            fh.write(json.dumps(ex.to_dict()) + "\n")
    logger.info("Generated %d HotpotQA examples -> %s", len(hotpot_examples), hotpot_path)


if __name__ == "__main__":
    main()
