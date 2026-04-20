"""BM25-based distractor retrieval for long-context construction.

Retrieves semantically similar but causally irrelevant Wikipedia passages
to serve as distractors in the LOCO ablation pipeline.

Includes a self-contained BM25 Okapi implementation (no external dependencies
beyond the standard library) following Robertson et al. (1995).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .schemas import NormalizedExample, SourceSegment

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


def _tokenize(text: str) -> List[str]:
    return [t for t in _WHITESPACE_RE.split(text.strip().lower()) if t]


class BM25Retriever:
    """BM25 Okapi retriever over a pre-loaded passage corpus.

    Self-contained implementation following Robertson et al. (1995).
    No external dependencies beyond the Python standard library.

    Parameters
    ----------
    corpus : sequence of dicts with at least ``"text"`` and ``"title"`` keys.
    k1, b : BM25 tuning parameters (Robertson defaults: k1=1.5, b=0.75).
    """

    def __init__(
        self,
        corpus: Sequence[Dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._corpus = list(corpus)
        self._k1 = k1
        self._b = b

        self._doc_tokens: List[List[str]] = []
        self._doc_lens: List[int] = []
        self._df: Counter = Counter()

        for entry in self._corpus:
            tokens = _tokenize(entry.get("text", ""))
            self._doc_tokens.append(tokens)
            self._doc_lens.append(len(tokens))
            for term in set(tokens):
                self._df[term] += 1

        self._n = len(self._corpus)
        self._avgdl = sum(self._doc_lens) / self._n if self._n else 1.0
        logger.info("BM25 index built: %d documents, avgdl=%.1f", self._n, self._avgdl)

    def _score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        doc_tokens = self._doc_tokens[doc_idx]
        dl = self._doc_lens[doc_idx]
        tf = Counter(doc_tokens)
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            term_tf = tf[term]
            term_df = self._df.get(term, 0)
            idf = math.log((self._n - term_df + 0.5) / (term_df + 0.5) + 1.0)
            numerator = term_tf * (self._k1 + 1.0)
            denominator = term_tf + self._k1 * (1.0 - self._b + self._b * dl / self._avgdl)
            score += idf * numerator / denominator
        return score

    def retrieve(
        self,
        query: str,
        *,
        n: int = 10,
        exclude_titles: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Return the top-*n* passages ranked by BM25 relevance to *query*.

        Passages whose title appears in *exclude_titles* are filtered out
        so that gold evidence is never returned as a distractor.
        """
        exclude_titles = {t.lower().strip() for t in (exclude_titles or set())}
        tokens = _tokenize(query)

        scored = []
        for idx in range(self._n):
            title = self._corpus[idx].get("title", "")
            if title.lower().strip() in exclude_titles:
                continue
            score = self._score_document(tokens, idx)
            if score > 0:
                scored.append((idx, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for idx, score in scored[:n]:
            results.append({**self._corpus[idx], "_bm25_score": float(score)})
        return results


def retrieve_distractors_for_example(
    example: NormalizedExample,
    retriever: BM25Retriever,
    *,
    n_distractors: int = 10,
    segment_id_offset: int = 1000,
) -> List[SourceSegment]:
    """Retrieve BM25 distractors for a single example.

    The query used for retrieval is the example's query text combined with
    the evidence segment titles (to improve topical recall).  Gold evidence
    titles are excluded from the results to prevent leaking causal passages.
    """
    gold_titles = {seg.title for seg in example.evidence_segments}

    query_parts = [example.query]
    query_parts.extend(gold_titles)
    combined_query = " ".join(query_parts)

    hits = retriever.retrieve(
        combined_query,
        n=n_distractors,
        exclude_titles=gold_titles,
    )

    segments: List[SourceSegment] = []
    for i, hit in enumerate(hits):
        text = hit.get("text", "").strip()
        if not text:
            continue
        segments.append(
            SourceSegment(
                segment_id=segment_id_offset + i,
                title=str(hit.get("title", f"distractor_{i}")),
                text=text,
                is_gold=False,
                source_type="bm25_distractor",
                metadata={
                    "bm25_score": hit.get("_bm25_score", 0.0),
                    "retrieval_query": combined_query,
                },
            )
        )
    return segments


def enrich_examples_with_distractors(
    examples: List[NormalizedExample],
    retriever: BM25Retriever,
    *,
    n_distractors: int = 10,
    segment_id_offset: int = 1000,
    replace_existing: bool = False,
) -> List[NormalizedExample]:
    """Add BM25 distractors to each example in-place and return the list.

    If *replace_existing* is ``True``, any pre-existing distractor segments
    are removed before retrieval.  Otherwise new distractors are appended.
    """
    for example in examples:
        if replace_existing:
            example.distractor_segments = []

        new_distractors = retrieve_distractors_for_example(
            example,
            retriever,
            n_distractors=n_distractors,
            segment_id_offset=segment_id_offset + len(example.distractor_segments),
        )
        example.distractor_segments.extend(new_distractors)
        logger.debug(
            "Example %s: retrieved %d distractors (total %d)",
            example.example_id,
            len(new_distractors),
            len(example.distractor_segments),
        )
    return examples


def load_wikipedia_corpus(
    path: str,
    *,
    max_passages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load a Wikipedia passage corpus from a JSONL file.

    Each line must be a JSON object with at least ``"title"`` and ``"text"``.
    Compatible with the Wikipedia paragraph dumps used by DPR and KILT.
    """
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Wikipedia corpus not found: {corpus_path}")

    corpus: List[Dict[str, Any]] = []
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("text", "").strip():
                corpus.append(entry)
            if max_passages is not None and len(corpus) >= max_passages:
                break

    logger.info("Loaded %d passages from %s", len(corpus), corpus_path)
    return corpus


def load_wikipedia_from_huggingface(
    *,
    split: str = "train",
    max_passages: Optional[int] = 50_000,
    dataset_name: str = "wikipedia",
    dataset_config: str = "20220301.en",
) -> List[Dict[str, Any]]:
    """Load Wikipedia passages from HuggingFace Datasets.

    Returns a list of ``{"title": ..., "text": ...}`` dicts suitable
    for constructing a ``BM25Retriever``.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets is required for Wikipedia loading. "
            "Install it with: pip install datasets"
        )

    logger.info(
        "Loading Wikipedia from HuggingFace (%s/%s, split=%s, max=%s)",
        dataset_name,
        dataset_config,
        split,
        max_passages,
    )

    if max_passages is not None:
        ds = load_dataset(
            dataset_name,
            dataset_config,
            split=f"{split}[:{max_passages}]",
            trust_remote_code=True,
        )
    else:
        ds = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            trust_remote_code=True,
        )

    corpus: List[Dict[str, Any]] = []
    for row in ds:
        text = str(row.get("text", "")).strip()
        title = str(row.get("title", "")).strip()
        if text and title:
            corpus.append({"title": title, "text": text})

    logger.info("Loaded %d Wikipedia passages from HuggingFace", len(corpus))
    return corpus
