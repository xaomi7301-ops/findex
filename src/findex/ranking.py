"""Lab 03: interchangeable TF-IDF / BM25 ranking with highlighted snippets."""

from __future__ import annotations

import heapq
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from findex.index import PlainPosting, Posting
from findex.object_model import Index
from findex.query import parse
from findex.timing import timed


class Scorer(Protocol):
    """Duck-typed ranking strategy; no inheritance required."""

    def score(self, term: str, posting: Posting | PlainPosting, index: Index) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    """Smoothed IDF and log-scaled frequency (a simple TF-IDF baseline)."""

    def score(self, term: str, posting: Posting | PlainPosting, index: Index) -> float:
        if posting.tf < 1 or index.num_docs == 0:
            return 0.0
        idf = math.log((index.num_docs + 1) / (index.df(term) + 1)) + 1.0
        return (1.0 + math.log(posting.tf)) * idf

    def __call__(self, term: str, posting: Posting | PlainPosting, index: Index) -> float:
        return self.score(term, posting, index)


@dataclass(frozen=True, slots=True)
class BM25:
    """Robertson–Sparck Jones smoothed BM25 with length normalization."""

    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 <= 0 or not 0 <= self.b <= 1:
            raise ValueError("Expected k1 > 0 and 0 <= b <= 1")

    def score(self, term: str, posting: Posting | PlainPosting, index: Index) -> float:
        if posting.tf < 1 or index.num_docs == 0:
            return 0.0
        n = index.num_docs
        df = index.df(term)
        idf = math.log1p((n - df + 0.5) / (df + 0.5))
        avg_length = index.avg_doc_length or 1.0
        normalization = 1.0 - self.b + self.b * index.doc_length(posting.doc_id) / avg_length
        return idf * (posting.tf * (self.k1 + 1)) / (
            posting.tf + self.k1 * normalization
        )

    def __call__(self, term: str, posting: Posting | PlainPosting, index: Index) -> float:
        return self.score(term, posting, index)


@dataclass(frozen=True, order=True, slots=True)
class SearchResult:
    """Results compare by descending score, then ascending doc_id."""

    sort_key: tuple[float, int] = field(init=False, repr=False)
    doc_id: int = field(compare=False)
    score: float = field(compare=False)
    title: str = field(compare=False)
    path: str = field(compare=False)
    snippet: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sort_key", (-self.score, self.doc_id))

    def __str__(self) -> str:
        return f"{self.score:.3f} | {self.path} | {self.title}"


def snippet_for(index: Index, doc_id: int, terms: tuple[str, ...], *, radius: int = 80) -> str:
    """Read a result document on demand and highlight term matches with [[...]]."""
    if index.corpus_root is None:
        return "[snippet unavailable: supply --corpus]"
    relative = index.doc_meta[doc_id].path
    source = index.corpus_root / Path(relative)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return "[snippet unavailable: document not found]"

    unique_terms = sorted({term for term in terms if term}, key=len, reverse=True)
    pattern = (
        re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(term) for term in unique_terms) + r")(?!\w)",
            re.IGNORECASE,
        )
        if unique_terms
        else None
    )
    # Prefer the rarest matching query term instead of simply the earliest hit.
    found = None
    if pattern is not None:
        for term in sorted(unique_terms, key=index.df):
            candidate = re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.IGNORECASE)
            if candidate is not None:
                found = candidate
                break
    at = found.start() if found is not None else 0
    start = max(0, at - radius)
    end = min(len(text), at + radius + (len(found.group()) if found else 0))
    excerpt = text[start:end]
    excerpt = re.sub(r"\s+", " ", excerpt).strip()
    if pattern is not None:
        excerpt = pattern.sub(lambda match: f"[[{match.group()}]]", excerpt)
    return ("…" if start > 0 else "") + excerpt + ("…" if end < len(text) else "")


@timed
def ranked_search(
    index: Index,
    query: str,
    *,
    scorer: Scorer | None = None,
    k: int = 10,
) -> list[SearchResult]:
    """Evaluate the Boolean AST, score matching docs, and heap-select the best k."""
    if k < 0:
        raise ValueError("k must be nonnegative")
    if k == 0:
        return []
    scorer = scorer if scorer is not None else BM25()
    matched = index.matching_ids(query)
    if not matched:
        return []
    positive_terms = tuple(dict.fromkeys(parse(query).positive_terms()))
    scores: dict[int, float] = defaultdict(float)

    for term in positive_terms:
        postings = index.get(term, [])
        from array import array

        if isinstance(postings, array):
            pairs = (Posting(int(postings[i]), int(postings[i + 1]))
                     for i in range(0, len(postings), 2))
        else:
            pairs = iter(postings)
        for posting in pairs:
            if posting.doc_id in matched:
                scores[posting.doc_id] += scorer.score(term, posting, index)

    # NOT-only queries are legitimate: their matching documents receive zero.
    for doc_id in matched:
        scores.setdefault(doc_id, 0.0)
    ranked = heapq.nlargest(k, scores.items(), key=lambda value: (value[1], -value[0]))
    return [
        SearchResult(
            doc_id=doc_id,
            score=score,
            title=index.doc_meta[doc_id].title,
            path=index.doc_meta[doc_id].path,
            snippet=snippet_for(index, doc_id, positive_terms),
        )
        for doc_id, score in ranked
    ]
