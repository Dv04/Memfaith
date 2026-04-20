#!/usr/bin/env python3
"""Build prepared FEVER contexts with real evidence and BM25 distractors.

This script loads FEVER claims from HuggingFace, resolves evidence sentence
text from the Wikipedia corpus, retrieves BM25-ranked distractors, and
outputs prepared JSONL files that the CCS runner can consume directly.

Usage
-----
    python scripts/build_fever_contexts.py \
        --output data/memfaith/fever_prepared.jsonl \
        --max-examples 500 \
        --n-distractors 10 \
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


def load_fever_from_huggingface(
    *,
    split: str = "train",
    max_examples: int = 500,
    label_filter: str | None = None,
) -> list[dict]:
    from datasets import load_dataset

    logger.info("Loading FEVER dataset from HuggingFace (split=%s)", split)
    ds = load_dataset("fever", "v1.0", split=split, trust_remote_code=True)

    records = []
    for row in ds:
        label = row.get("label")
        label_str = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO"}.get(
            label, str(label)
        )
        if label_filter and label_str != label_filter:
            continue

        claim = row.get("claim", "")
        evidences = row.get("evidences", [])

        if not claim or not evidences:
            continue

        records.append({
            "id": row.get("id", len(records)),
            "claim": claim,
            "label": label_str,
            "evidences": evidences,
        })

        if len(records) >= max_examples:
            break

    logger.info("Loaded %d FEVER examples", len(records))
    return records


def resolve_fever_evidence(
    fever_records: list[dict],
    wiki_corpus: list[dict],
) -> list[NormalizedExample]:
    """Convert raw FEVER records into NormalizedExample with evidence text.

    When exact Wikipedia page lookups are not available, evidence is
    resolved by title-matching against the loaded corpus passages.
    """
    title_index: dict[str, str] = {}
    for entry in wiki_corpus:
        title = entry.get("title", "").strip().lower()
        text = entry.get("text", "").strip()
        if title and text and title not in title_index:
            title_index[title] = text

    examples: list[NormalizedExample] = []
    skipped = 0

    for record in fever_records:
        evidence_segments: list[SourceSegment] = []
        seen_titles: set[str] = set()
        seg_id = 0

        for evidence_group in record["evidences"]:
            for annotation in evidence_group:
                wiki_title = annotation.get("wikipedia_url", "") or annotation.get("title", "")
                wiki_title = wiki_title.replace("_", " ").strip()

                if not wiki_title or wiki_title.lower() in seen_titles:
                    continue
                seen_titles.add(wiki_title.lower())

                text = title_index.get(wiki_title.lower(), "")
                if not text:
                    continue

                evidence_segments.append(
                    SourceSegment(
                        segment_id=seg_id,
                        title=wiki_title,
                        text=text,
                        is_gold=True,
                        source_type="wikipedia_evidence",
                    )
                )
                seg_id += 1

        if not evidence_segments:
            skipped += 1
            continue

        examples.append(
            NormalizedExample(
                dataset="fever",
                example_id=f"fever-{record['id']}",
                query=record["claim"],
                gold_answer=record["label"],
                task_type="classification",
                evidence_segments=evidence_segments,
                distractor_segments=[],
                metadata={
                    "required_segment_ids": [s.segment_id for s in evidence_segments],
                },
            )
        )

    logger.info(
        "Resolved %d examples with evidence text (%d skipped due to missing pages)",
        len(examples),
        skipped,
    )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prepared FEVER contexts")
    parser.add_argument("--output", default="data/memfaith/fever_prepared.jsonl")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--n-distractors", type=int, default=10)
    parser.add_argument("--wiki-passages", type=int, default=50_000)
    parser.add_argument("--wiki-corpus", default=None, help="Path to local JSONL corpus")
    parser.add_argument("--label-filter", default=None, choices=["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    fever_records = load_fever_from_huggingface(
        max_examples=args.max_examples * 3,
        label_filter=args.label_filter,
    )

    if args.wiki_corpus:
        from src.memfaith.distractor_retrieval import load_wikipedia_corpus
        wiki_corpus = load_wikipedia_corpus(args.wiki_corpus, max_passages=args.wiki_passages)
    else:
        from src.memfaith.distractor_retrieval import load_wikipedia_from_huggingface
        wiki_corpus = load_wikipedia_from_huggingface(max_passages=args.wiki_passages)

    examples = resolve_fever_evidence(fever_records, wiki_corpus)
    examples = examples[: args.max_examples]

    logger.info("Building BM25 index over %d passages ...", len(wiki_corpus))
    retriever = BM25Retriever(wiki_corpus)

    for example in examples:
        distractors = retrieve_distractors_for_example(
            example,
            retriever,
            n_distractors=args.n_distractors,
        )
        example.distractor_segments = distractors

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example.to_dict()) + "\n")

    logger.info("Wrote %d prepared examples to %s", len(examples), output_path)


if __name__ == "__main__":
    main()
