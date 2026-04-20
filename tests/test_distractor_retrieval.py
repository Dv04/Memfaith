"""Tests for BM25 distractor retrieval and spaCy chunking modules."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memfaith.schemas import NormalizedExample, SourceSegment
from src.memfaith.distractor_retrieval import (
    BM25Retriever,
    retrieve_distractors_for_example,
    enrich_examples_with_distractors,
)
from src.memfaith.chunking import DeterministicChunker
from src.memfaith.context_builder import LongContextBuilder
from src.memfaith.adapters import load_prepared_examples


MINI_CORPUS = [
    {"title": "Paris", "text": "Paris is the capital and most populous city of France."},
    {"title": "Berlin", "text": "Berlin is the capital city of Germany."},
    {"title": "Madrid", "text": "Madrid is the capital city of Spain."},
    {"title": "Rome", "text": "Rome is the capital city of Italy and home to the Colosseum."},
    {"title": "Tokyo", "text": "Tokyo is the capital of Japan and one of the most populous cities."},
    {"title": "London", "text": "London is the capital of England and the United Kingdom."},
    {"title": "Eiffel Tower", "text": "The Eiffel Tower is a wrought-iron lattice tower in Paris."},
    {"title": "Great Wall", "text": "The Great Wall of China stretches across northern China."},
    {"title": "Colosseum", "text": "The Colosseum is an oval amphitheatre in Rome, Italy."},
    {"title": "Pyramids", "text": "The Great Pyramids of Giza are ancient monuments in Egypt."},
]


def _make_example(
    example_id: str = "test-1",
    query: str = "The Eiffel Tower is in Paris.",
    gold_answer: str = "SUPPORTS",
    evidence_title: str = "Eiffel Tower",
    evidence_text: str = "The Eiffel Tower is a landmark in Paris, France.",
) -> NormalizedExample:
    return NormalizedExample(
        dataset="fever",
        example_id=example_id,
        query=query,
        gold_answer=gold_answer,
        task_type="classification",
        evidence_segments=[
            SourceSegment(
                segment_id=0,
                title=evidence_title,
                text=evidence_text,
                is_gold=True,
                source_type="evidence",
            )
        ],
        distractor_segments=[],
        metadata={"required_segment_ids": [0]},
    )


class TestBM25Retriever(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = BM25Retriever(MINI_CORPUS)

    def test_retrieve_returns_correct_count(self) -> None:
        results = self.retriever.retrieve("capital of France", n=3)
        self.assertEqual(len(results), 3)

    def test_retrieve_excludes_titles(self) -> None:
        results = self.retriever.retrieve(
            "Eiffel Tower Paris",
            n=5,
            exclude_titles={"Eiffel Tower"},
        )
        titles = {r["title"] for r in results}
        self.assertNotIn("Eiffel Tower", titles)

    def test_retrieve_ranks_relevant_passages_higher(self) -> None:
        results = self.retriever.retrieve("capital of France Paris", n=3)
        self.assertEqual(results[0]["title"], "Paris")

    def test_retrieve_returns_bm25_scores(self) -> None:
        results = self.retriever.retrieve("capital city", n=2)
        for r in results:
            self.assertIn("_bm25_score", r)
            self.assertGreater(r["_bm25_score"], 0.0)


class TestDistractorRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = BM25Retriever(MINI_CORPUS)
        self.example = _make_example()

    def test_retrieve_distractors_produces_source_segments(self) -> None:
        distractors = retrieve_distractors_for_example(
            self.example, self.retriever, n_distractors=3,
        )
        self.assertEqual(len(distractors), 3)
        for seg in distractors:
            self.assertIsInstance(seg, SourceSegment)
            self.assertFalse(seg.is_gold)
            self.assertEqual(seg.source_type, "bm25_distractor")

    def test_distractors_exclude_gold_evidence_title(self) -> None:
        distractors = retrieve_distractors_for_example(
            self.example, self.retriever, n_distractors=8,
        )
        titles = {seg.title for seg in distractors}
        self.assertNotIn("Eiffel Tower", titles)

    def test_enrich_examples_adds_distractors(self) -> None:
        examples = [_make_example()]
        self.assertEqual(len(examples[0].distractor_segments), 0)

        enrich_examples_with_distractors(
            examples, self.retriever, n_distractors=4,
        )
        self.assertEqual(len(examples[0].distractor_segments), 4)

    def test_enrich_replace_existing(self) -> None:
        example = _make_example()
        example.distractor_segments = [
            SourceSegment(
                segment_id=999, title="Old", text="Old distractor",
                is_gold=False, source_type="manual",
            )
        ]
        enrich_examples_with_distractors(
            [example], self.retriever, n_distractors=3, replace_existing=True,
        )
        self.assertEqual(len(example.distractor_segments), 3)
        self.assertTrue(all(s.source_type == "bm25_distractor" for s in example.distractor_segments))


class TestPipelineIntegration(unittest.TestCase):
    """Verify enriched examples integrate with existing context builder and chunker."""

    def test_enriched_example_builds_and_chunks(self) -> None:
        retriever = BM25Retriever(MINI_CORPUS)
        example = _make_example()
        enrich_examples_with_distractors([example], retriever, n_distractors=4)

        builder = LongContextBuilder(seed=42)
        built = builder.build(example)
        self.assertIn("Eiffel Tower", built.context_text)
        self.assertEqual(len(built.ordered_segments), 5)  # 1 evidence + 4 distractors

        chunker = DeterministicChunker()
        chunked = chunker.chunk(built, k=2)
        self.assertEqual(len(chunked.chunks), 2)
        for chunk in chunked.chunks:
            self.assertTrue(chunk.text.strip())

    def test_smoke_data_still_loads(self) -> None:
        """Existing smoke datasets must not be broken by our changes."""
        fever = load_prepared_examples("data/memfaith/fever_smoke.jsonl")
        hotpot = load_prepared_examples("data/memfaith/hotpot_smoke.jsonl")
        self.assertEqual(len(fever), 3)
        self.assertEqual(len(hotpot), 2)


class TestSpacyChunking(unittest.TestCase):
    """Test spaCy-based sentence splitting in the chunker."""

    def setUp(self) -> None:
        try:
            import spacy
            spacy.load("en_core_web_sm")
            self.spacy_available = True
        except (ImportError, OSError):
            self.spacy_available = False

    def test_spacy_chunker_produces_valid_output(self) -> None:
        if not self.spacy_available:
            self.skipTest("spaCy en_core_web_sm not installed")

        example = _make_example()
        example.distractor_segments = [
            SourceSegment(
                segment_id=100, title="Berlin",
                text="Berlin is the capital of Germany. It has a rich history.",
                is_gold=False, source_type="distractor",
            ),
        ]

        builder = LongContextBuilder(seed=0)
        built = builder.build(example)

        chunker = DeterministicChunker(use_spacy=True)
        chunked = chunker.chunk(built, k=2)
        self.assertEqual(len(chunked.chunks), 2)
        for chunk in chunked.chunks:
            self.assertTrue(chunk.text.strip())

    def test_regex_fallback_still_works(self) -> None:
        """Default chunker without spaCy must still work exactly as before."""
        example = _make_example()
        example.distractor_segments = [
            SourceSegment(
                segment_id=100, title="Berlin",
                text="Berlin is the capital of Germany.",
                is_gold=False, source_type="distractor",
            ),
        ]

        builder = LongContextBuilder(seed=0)
        built = builder.build(example)

        chunker = DeterministicChunker(use_spacy=False)
        chunked = chunker.chunk(built, k=2)
        self.assertEqual(len(chunked.chunks), 2)


if __name__ == "__main__":
    unittest.main()
