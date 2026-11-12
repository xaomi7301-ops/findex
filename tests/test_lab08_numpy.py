"""Lab 08: compare scalar/vector scores, ordering, and timing on fixtures."""
from __future__ import annotations

import numpy as np
import pytest

from findex.index import build_index
from findex.numpy_ranking import score_numpy, top_k
from findex.object_model import Index, build_ranked_index
from findex.query import parse
from findex.ranking import BM25, TfIdf, ranked_search, score_python


@pytest.mark.parametrize("storage", ["plain", "slots", "array"])
@pytest.mark.parametrize("scorer", [BM25(), TfIdf()])
@pytest.mark.parametrize("query", ["python", "python OR async OR java", "NOT java", "missing"])
@pytest.mark.parametrize("k", [1, 3, 10])
def test_vector_matches_lab03(tiny_corpus, storage, scorer, query, k):
    index = Index(build_index(tiny_corpus, storage=storage))
    try:
        matched = index.matching_ids(query)
        terms = tuple(dict.fromkeys(parse(query).positive_terms()))
        scalar = score_python(index, matched, terms, scorer, k)
        vector = score_numpy(index, matched, terms, scorer, k)
        assert [doc for doc, _ in vector] == [doc for doc, _ in scalar]
        assert [score for _, score in vector] == pytest.approx([score for _, score in scalar])
    finally:
        index.close()


def test_numeric_layout_and_phrase(tiny_corpus):
    with_positions = build_ranked_index(tiny_corpus)
    numeric = with_positions.numpy_postings
    assert numeric is with_positions.numpy_postings
    for values in (numeric.doc_ids, numeric.tfs, numeric.doc_lengths):
        assert values.dtype == np.int32
        assert values.flags.c_contiguous
        assert not values.flags.writeable
    assert len(ranked_search(with_positions, '"event loop"')) == 2
    with_positions.close()


def test_partition_ties_are_deterministic():
    scores = np.array([1.0, 3.0, 3.0, 3.0, 0.0])
    candidates = np.array([4, 3, 2, 1, 0], dtype=np.int32)
    assert top_k(scores, candidates, 2) == [(1, 3.0), (2, 3.0)]
    assert top_k(scores, candidates, 0) == []


def test_benchmark_search(benchmark, tiny_corpus):
    index = build_ranked_index(tiny_corpus)
    _ = index.numpy_postings
    # Warm the query cache; no snippets/file reads in the timed search.
    index.corpus_root = None
    query = "python OR async OR java"
    ranked_search(index, query)
    result = benchmark(ranked_search, index, query)
    assert result
    # A generous regression ceiling, not a claim about production latency.
    assert benchmark.stats.stats.median < 0.050
    index.close()


def test_benchmark_build_index(benchmark, tiny_corpus):
    result = benchmark(build_index, tiny_corpus, with_positions=True)
    assert result.document_count == 4
    assert benchmark.stats.stats.median < 0.250
