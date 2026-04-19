#!/usr/bin/env python3
"""Build prepared HotpotQA contexts with BM25 distractors.

HotpotQA already ships with supporting and non-supporting context paragraphs.
This script loads the dataset, separates gold evidence from context distractors,
optionally enriches with additional BM25 distractors from Wikipedia, and outputs
prepared JSONL for the CCS runner.

Usage
-----
    python scripts/build_hotpotqa_contexts.py \
        --output data/memfaith/hotpotqa_prepared.jsonl \
        --max-examples 500 \
        --extra-distractors 5 \
        --wiki-passages 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memfaith.distractor_retrieval import (
    BM25Retriever,
    retrieve_distractors_for_example,
)
from src.memfaith.schemas import NormalizedExample, SourceSegment

logger = logging.getLogger(__name__)


def load_hotpotqa_from_huggingface(
    *,
    split: str = "train",
    max_examples: int = 500,
    difficulty_filter: str | None = None,
) -> list[NormalizedExample]:
    from datasets import load_dataset

    logger.info("Loading HotpotQA from HuggingFace (split=%s)", split)
    ds = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)

    examples: list[NormalizedExample] = []

    for row in ds:
        if difficulty_filter and row.get("level", "") != difficulty_filter:
            continue

        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()
        if not question or not answer:
            continue

        supporting_titles = set()
        for title in row.get("supporting_facts", {}).get("title", []):
            supporting_titles.add(title.strip())

        context_titles = row.get("context", {}).get("title", [])
        context_sentences = row.get("context", {}).get("sentences", [])

        evidence_segments: list[SourceSegment] = []
        distractor_segments: list[SourceSegment] = []
        seg_id = 0

        for title, sentences in zip(context_titles, context_sentences):
            text = " ".join(s.strip() for s in sentences if s.strip()).strip()
            if not text:
                continue

            is_gold = title.strip() in supporting_titles
            segment = SourceSegment(
                segment_id=seg_id,
                title=title.strip(),
                text=text,
                is_gold=is_gold,
                source_type="supporting_context" if is_gold else "context_distractor",
            )

            if is_gold:
                evidence_segments.append(segment)
            else:
                distractor_segments.append(segment)
            seg_id += 1

        if not evidence_segments:
            continue

        examples.append(
            NormalizedExample(
                dataset="hotpotqa",
                example_id=str(row.get("id", len(examples))),
                query=question,
                gold_answer=answer,
                task_type="qa",
                evidence_segments=evidence_segments,
                distractor_segments=distractor_segments,
                metadata={
                    "supporting_titles": sorted(supporting_titles),
                    "required_segment_ids": [s.segment_id for s in evidence_segments],
                    "level": row.get("level", ""),
                    "type": row.get("type", ""),
                },
            )
        )

        if len(examples) >= max_examples:
            break

    logger.info("Loaded %d HotpotQA examples", len(examples))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prepared HotpotQA contexts")
    parser.add_argument("--output", default="data/memfaith/hotpotqa_prepared.jsonl")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--extra-distractors", type=int, default=5,
                        help="Additional BM25 distractors from Wikipedia beyond built-in context")
    parser.add_argument("--wiki-passages", type=int, default=50_000)
    parser.add_argument("--wiki-corpus", default=None, help="Path to local JSONL corpus")
    parser.add_argument("--difficulty", default=None, choices=["easy", "medium", "hard"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    examples = load_hotpotqa_from_huggingface(
        max_examples=args.max_examples,
        difficulty_filter=args.difficulty,
    )

    if args.extra_distractors > 0:
        if args.wiki_corpus:
            from src.memfaith.distractor_retrieval import load_wikipedia_corpus
            wiki_corpus = load_wikipedia_corpus(args.wiki_corpus, max_passages=args.wiki_passages)
        else:
            from src.memfaith.distractor_retrieval import load_wikipedia_from_huggingface
            wiki_corpus = load_wikipedia_from_huggingface(max_passages=args.wiki_passages)

        logger.info("Building BM25 index over %d passages ...", len(wiki_corpus))
        retriever = BM25Retriever(wiki_corpus)

        for example in examples:
            existing_titles = {s.title for s in example.distractor_segments}
            extra = retrieve_distractors_for_example(
                example,
                retriever,
                n_distractors=args.extra_distractors,
                segment_id_offset=1000 + len(example.distractor_segments),
            )
            extra = [s for s in extra if s.title not in existing_titles]
            example.distractor_segments.extend(extra)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example.to_dict()) + "\n")

    logger.info("Wrote %d prepared examples to %s", len(examples), output_path)


if __name__ == "__main__":
    main()
