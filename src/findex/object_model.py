"""Lab 03: an object-oriented Mapping interface over Lab 02's inverted index.

The wrapped InvertedIndex remains unchanged, so lab-02 pickle/JSON files still load.
For phrase search build a new index with --positions; no rewrite of lab-02 tags.
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from functools import cached_property, lru_cache
from pathlib import Path

from findex.index import InvertedIndex, PostingList, build_index
from findex.store import load
from findex.timing import timed

logger = logging.getLogger(__name__)


class Index(Mapping[str, PostingList]):
    """Read-only, Pythonic view of a stored inverted index.

    len(ix) gives the vocabulary size, not the document count.
    Cache belongs to each Index instance, preventing mixing unrelated corpora.
    """

    def __init__(self, raw: InvertedIndex, corpus_root: Path | None = None) -> None:
        self._raw = raw
        self.corpus_root = Path(corpus_root) if corpus_root is not None else None
        self._closed = False

        @lru_cache(maxsize=256)
        def resolve(query: str) -> frozenset[int]:
            from findex.query import parse

            return frozenset(parse(query).evaluate(self))

        self._resolve = resolve

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Index has been closed")

    def __len__(self) -> int:
        self._ensure_open()
        return len(self._raw.postings)

    def __iter__(self) -> Iterator[str]:
        self._ensure_open()
        return iter(self._raw.postings)

    def __getitem__(self, term: str) -> PostingList:
        self._ensure_open()
        return self._raw.postings[term]

    def __contains__(self, term: object) -> bool:
        self._ensure_open()
        return term in self._raw.postings

    def __repr__(self) -> str:
        if self._closed:
            return "Index(closed=True)"
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"

    @property
    def num_docs(self) -> int:
        self._ensure_open()
        return len(self._raw.doc_lengths)

    @cached_property
    def avg_doc_length(self) -> float:
        self._ensure_open()
        if self.num_docs == 0:
            return 0.0
        return sum(self._raw.doc_lengths.values()) / self.num_docs

    @property
    def has_positions(self) -> bool:
        self._ensure_open()
        return self._raw.with_positions

    @property
    def doc_meta(self):
        self._ensure_open()
        return self._raw.doc_meta

    @property
    def doc_lengths(self):
        self._ensure_open()
        return self._raw.doc_lengths

    def doc_length(self, doc_id: int) -> int:
        self._ensure_open()
        return self._raw.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        """Number of documents containing term (document frequency)."""
        from array import array

        self._ensure_open()
        entries = self._raw.postings.get(term)
        if entries is None:
            return 0
        return len(entries) // 2 if isinstance(entries, array) else len(entries)

    def matching_ids(self, query: str) -> frozenset[int]:
        """Cached parse → Boolean match; log hit/miss for the defense demo."""
        self._ensure_open()
        before = self._resolve.cache_info().hits
        result = self._resolve(query)
        hit = self._resolve.cache_info().hits > before
        logger.info("query-cache %s: %r (%d docs)", "HIT" if hit else "MISS", query, len(result))
        return result

    def cache_info(self):
        self._ensure_open()
        return self._resolve.cache_info()

    def close(self) -> None:
        """Release cached results and owned postings even on exceptions."""
        if self._closed:
            return
        self._resolve.cache_clear()
        self.__dict__.pop("avg_doc_length", None)
        self._raw.postings.clear()
        self._raw.doc_lengths.clear()
        self._raw.doc_meta.clear()
        self._closed = True

    @property
    def raw(self) -> InvertedIndex:
        """Expose the underlying Lab 02 index to serialization code."""
        self._ensure_open()
        return self._raw

    @property
    def closed(self) -> bool:
        return self._closed


@timed
def build_ranked_index(root: Path, *, positions: bool = True) -> Index:
    """Build the Lab-02 index without changing its saved binary representation."""
    raw = build_index(Path(root), storage="slots", with_positions=positions)
    return Index(raw, corpus_root=root)


@timed
def load_ranked_index(path: Path, *, corpus_root: Path | None = None) -> Index:
    """Load a *trusted* own-generated pickle or a JSON index."""
    return Index(load(Path(path)), corpus_root=corpus_root)


@contextmanager
def open_index(path: Path, *, corpus_root: Path | None = None) -> Generator[Index]:
    """Open and reliably close an Index on normal and exceptional exit."""
    ix = load_ranked_index(path, corpus_root=corpus_root)
    try:
        yield ix
    finally:
        ix.close()
