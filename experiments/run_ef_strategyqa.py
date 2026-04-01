"""Entry point to run EF on StrategyQA."""

from __future__ import annotations

import argparse
import yaml

from . import run_ef_experiment


def main():
    parser = argparse.ArgumentParser(description="Run EF pipeline on StrategyQA.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/strategyqa_gpt2xl_rome.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.setdefault("dataset_name", "strategyqa")
    log_path = run_ef_experiment(config)
    print(f"Wrote EF log to {log_path}")


if __name__ == "__main__":
    main()
