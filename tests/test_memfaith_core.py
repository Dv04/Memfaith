"""Unit tests for the MemFaith CCS implementation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.memfaith import (
    AnswerComparator,
    CCSRunner,
    HeuristicBackend,
    LongContextBuilder,
    aggregate_records,
    export_chunk_labels,
    load_prepared_examples,
)


class MemFaithCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fever_examples = load_prepared_examples("data/memfaith/fever_smoke.jsonl")
        self.hotpot_examples = load_prepared_examples("data/memfaith/hotpot_smoke.jsonl")

    def test_load_prepared_examples(self) -> None:
        self.assertEqual(len([example for example in self.fever_examples if example.dataset == "fever"]), 3)
        self.assertEqual(len([example for example in self.hotpot_examples if example.dataset == "hotpotqa"]), 2)

    def test_runner_emits_flips_for_fever_support_example(self) -> None:
        backend = HeuristicBackend()
        runner = CCSRunner(
            backend=backend,
            comparator=AnswerComparator(),
            builder=LongContextBuilder(seed=0),
        )
        support_example = next(example for example in self.fever_examples if example.example_id == "fever-smoke-1")
        record = runner.run_record(support_example, 2)
        self.assertEqual(record["dataset"], "fever")
        self.assertGreaterEqual(record["ccs_example"], 0.5)
        self.assertTrue(record["full_context"]["is_correct"])

    def test_aggregate_and_export_chunk_labels(self) -> None:
        backend = HeuristicBackend()
        runner = CCSRunner(backend=backend)
        all_examples = self.fever_examples + self.hotpot_examples
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.jsonl"
            csv_path = Path(tmpdir) / "labels.csv"
            records = runner.run(all_examples, k_values=[0, 2], output_path=str(log_path))
            summary = aggregate_records(records)
            self.assertTrue(any(row["dataset"] == "fever" and row["k"] == 2 for row in summary))
            rows = export_chunk_labels(str(log_path), str(csv_path))
            self.assertTrue(rows)
            self.assertTrue(csv_path.exists())


if __name__ == "__main__":
    unittest.main()
