"""Core MemFaith CCS pipeline package."""

from .adapters import load_hotpotqa_json, load_prepared_examples, load_strategyqa_split
from .backends import HeuristicBackend, TransformersBackend
from .cache import SQLitePredictionCache
from .chunking import DeterministicChunker
from .comparators import AnswerComparator
from .context_builder import LongContextBuilder
from .distractor_retrieval import (
    BM25Retriever,
    enrich_examples_with_distractors,
    load_wikipedia_corpus,
    load_wikipedia_from_huggingface,
    retrieve_distractors_for_example,
)
from .label_builders import export_chunk_labels
from .llm_judge import BaseLLMJudge, MockLLMJudge, OpenAILLMJudge
from .metrics import aggregate_records, load_experiment_log, write_summary_csv
from .multi_hop_analysis import (
    build_dependency_matrix,
    compute_distributed_causal_score,
    compute_multi_chunk_dependency,
    summarize_dependency_analysis,
)
from .plotting import (
    plot_ccs_by_label,
    plot_ccs_degradation_curve,
    plot_dataset_comparison,
    plot_dependency_heatmap,
    plot_gold_vs_nongold_flip,
    plot_positional_flip_rate,
)
from .runner import CCSRunner
from .stratification import (
    stratify_by_dataset,
    stratify_by_gold_coverage,
    stratify_by_label,
    stratify_by_position,
)

__all__ = [
    "AnswerComparator",
    "BaseLLMJudge",
    "BM25Retriever",
    "CCSRunner",
    "DeterministicChunker",
    "HeuristicBackend",
    "LongContextBuilder",
    "MockLLMJudge",
    "OpenAILLMJudge",
    "SQLitePredictionCache",
    "TransformersBackend",
    "aggregate_records",
    "build_dependency_matrix",
    "compute_distributed_causal_score",
    "compute_multi_chunk_dependency",
    "enrich_examples_with_distractors",
    "export_chunk_labels",
    "load_experiment_log",
    "load_hotpotqa_json",
    "load_prepared_examples",
    "load_strategyqa_split",
    "load_wikipedia_corpus",
    "load_wikipedia_from_huggingface",
    "plot_ccs_by_label",
    "plot_ccs_degradation_curve",
    "plot_dataset_comparison",
    "plot_dependency_heatmap",
    "plot_gold_vs_nongold_flip",
    "plot_positional_flip_rate",
    "retrieve_distractors_for_example",
    "stratify_by_dataset",
    "stratify_by_gold_coverage",
    "stratify_by_label",
    "stratify_by_position",
    "summarize_dependency_analysis",
    "write_summary_csv",
]
