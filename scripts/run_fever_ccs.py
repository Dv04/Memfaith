"""Run the MemFaith CCS pipeline on a prepared FEVER-style dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memfaith import (
    AnswerComparator,
    CCSRunner,
    HeuristicBackend,
    LongContextBuilder,
    SQLitePredictionCache,
    TransformersBackend,
    aggregate_records,
    load_prepared_examples,
    write_summary_csv,
)
from src.memfaith.batch_runner import BatchCCSRunner
from src.memfaith.backends import VLLMBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MemFaith CCS on a FEVER-style prepared dataset.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/memfaith/fever_smoke.jsonl",
        help="Prepared JSONL with query/evidence/distractor segments.",
    )
    parser.add_argument(
        "--backend",
        choices=["heuristic", "transformers", "vllm"],
        default="heuristic",
        help="Inference backend to use.",
    )
    parser.add_argument("--model-path", type=str, default="models/gpt2", help="HF model path when using transformers backend.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--k-values", type=str, default="0,2,4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-distractors", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-path", type=str, default="outputs/memfaith/fever_smoke_ccs.jsonl")
    parser.add_argument("--summary-path", type=str, default="outputs/memfaith/fever_smoke_summary.csv")
    parser.add_argument("--cache-path", type=str, default="outputs/memfaith/fever_smoke_cache.sqlite")
    return parser.parse_args()


def build_backend(args: argparse.Namespace):
    if args.backend == "heuristic":
        return HeuristicBackend()
    if args.backend == "vllm":
        return VLLMBackend(
            args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_new_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    return TransformersBackend(
        args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )


def main() -> None:
    args = parse_args()
    examples = load_prepared_examples(args.dataset_path)
    examples = [example for example in examples if example.dataset == "fever"]
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError(f"No FEVER examples found in {args.dataset_path}")

    backend = build_backend(args)
    cache = SQLitePredictionCache(args.cache_path)
    runner = BatchCCSRunner(
        backend=backend,
        comparator=AnswerComparator(),
        builder=LongContextBuilder(seed=args.seed, max_distractors=args.max_distractors),
        cache=cache,
    )
    k_values = [int(value.strip()) for value in args.k_values.split(",") if value.strip()]
    records = runner.run(examples, k_values=k_values, output_path=args.output_path)
    summary = aggregate_records(records)
    write_summary_csv(summary, args.summary_path)
    cache.close()

    print(f"Wrote {len(records)} records to {Path(args.output_path)}")
    print(f"Wrote summary to {Path(args.summary_path)}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
