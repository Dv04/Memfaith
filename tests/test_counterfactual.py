"""Tests for the counterfactual dataset generator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memfaith.counterfactual import (
    CounterfactualFEVERGenerator,
    CounterfactualHotpotQAGenerator,
    FictionalWorldBuilder,
)
from src.memfaith.adapters import load_prepared_examples
from src.memfaith.backends import HeuristicBackend
from src.memfaith.comparators import AnswerComparator
from src.memfaith.runner import CCSRunner


class TestFictionalWorldBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.world = FictionalWorldBuilder(seed=42, n_entities=10)

    def test_generates_correct_entity_count(self) -> None:
        self.assertEqual(len(self.world.entities), 10)

    def test_entities_have_unique_names(self) -> None:
        names = [e.name for e in self.world.entities.values()]
        self.assertEqual(len(names), len(set(names)))

    def test_facts_are_generated(self) -> None:
        self.assertGreater(len(self.world.facts), 0)

    def test_deterministic_with_same_seed(self) -> None:
        world2 = FictionalWorldBuilder(seed=42, n_entities=10)
        names1 = sorted(e.name for e in self.world.entities.values())
        names2 = sorted(e.name for e in world2.entities.values())
        self.assertEqual(names1, names2)

    def test_different_seed_different_world(self) -> None:
        world2 = FictionalWorldBuilder(seed=99, n_entities=10)
        names1 = sorted(e.name for e in self.world.entities.values())
        names2 = sorted(e.name for e in world2.entities.values())
        self.assertNotEqual(names1, names2)

    def test_entity_pairs_exist(self) -> None:
        pairs = self.world.get_entity_pairs()
        self.assertGreater(len(pairs), 0)

    def test_all_fact_texts_returns_corpus(self) -> None:
        corpus = self.world.all_fact_texts()
        self.assertGreater(len(corpus), 0)
        for entry in corpus:
            self.assertIn("title", entry)
            self.assertIn("text", entry)


class TestCounterfactualFEVER(unittest.TestCase):
    def setUp(self) -> None:
        self.world = FictionalWorldBuilder(seed=42, n_entities=10)
        self.gen = CounterfactualFEVERGenerator(self.world, seed=42)

    def test_generates_correct_count(self) -> None:
        examples = self.gen.generate(n_examples=10, n_distractors=3)
        self.assertEqual(len(examples), 10)

    def test_examples_have_correct_schema(self) -> None:
        examples = self.gen.generate(n_examples=5, n_distractors=3)
        for ex in examples:
            self.assertEqual(ex.dataset, "fever")
            self.assertEqual(ex.task_type, "classification")
            self.assertIn(ex.gold_answer, ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"])
            self.assertTrue(ex.evidence_segments)
            self.assertTrue(ex.evidence_segments[0].is_gold)
            self.assertTrue(ex.metadata.get("counterfactual"))
            self.assertIn("required_segment_ids", ex.metadata)

    def test_distractors_are_not_gold(self) -> None:
        examples = self.gen.generate(n_examples=5, n_distractors=3)
        for ex in examples:
            for seg in ex.distractor_segments:
                self.assertFalse(seg.is_gold)

    def test_roundtrip_through_jsonl(self) -> None:
        examples = self.gen.generate(n_examples=5, n_distractors=3)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict()) + "\n")
            path = f.name

        loaded = load_prepared_examples(path)
        self.assertEqual(len(loaded), 5)
        for ex in loaded:
            self.assertEqual(ex.dataset, "fever")
            self.assertTrue(ex.evidence_segments)
        Path(path).unlink()


class TestCounterfactualHotpotQA(unittest.TestCase):
    def setUp(self) -> None:
        self.world = FictionalWorldBuilder(seed=42, n_entities=10)
        self.gen = CounterfactualHotpotQAGenerator(self.world, seed=42)

    def test_generates_examples(self) -> None:
        examples = self.gen.generate(n_examples=5, n_distractors=3)
        self.assertGreater(len(examples), 0)

    def test_examples_have_correct_schema(self) -> None:
        examples = self.gen.generate(n_examples=5, n_distractors=3)
        for ex in examples:
            self.assertEqual(ex.dataset, "hotpotqa")
            self.assertEqual(ex.task_type, "qa")
            self.assertTrue(ex.query)
            self.assertTrue(ex.gold_answer)
            self.assertTrue(ex.evidence_segments)
            self.assertTrue(ex.metadata.get("counterfactual"))
            self.assertIn("hop_type", ex.metadata)

    def test_multi_hop_has_multiple_evidence(self) -> None:
        examples = self.gen.generate(n_examples=5, n_distractors=3)
        multi_evidence = [ex for ex in examples if len(ex.evidence_segments) > 1]
        self.assertGreater(len(multi_evidence), 0)


class TestCounterfactualPipelineIntegration(unittest.TestCase):
    """Verify counterfactual examples run through the full CCS pipeline."""

    def test_fever_runs_through_ccs(self) -> None:
        world = FictionalWorldBuilder(seed=42, n_entities=10)
        gen = CounterfactualFEVERGenerator(world, seed=42)
        examples = gen.generate(n_examples=3, n_distractors=3)

        runner = CCSRunner(
            backend=HeuristicBackend(),
            comparator=AnswerComparator(),
        )
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            records = runner.run(examples, k_values=[2], output_path=f.name)

        self.assertEqual(len(records), 3)
        for r in records:
            self.assertIn("ccs_example", r)
            self.assertIsNotNone(r["ccs_example"])
        Path(f.name).unlink()

    def test_hotpot_runs_through_ccs(self) -> None:
        world = FictionalWorldBuilder(seed=42, n_entities=10)
        gen = CounterfactualHotpotQAGenerator(world, seed=42)
        examples = gen.generate(n_examples=3, n_distractors=3)

        runner = CCSRunner(
            backend=HeuristicBackend(),
            comparator=AnswerComparator(),
        )
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            records = runner.run(examples, k_values=[2], output_path=f.name)

        self.assertGreater(len(records), 0)
        for r in records:
            self.assertIn("ccs_example", r)
        Path(f.name).unlink()


if __name__ == "__main__":
    unittest.main()
