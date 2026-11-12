"""Semantic retrieval tests use a deterministic fake encoder, no network/model."""
from __future__ import annotations

import numpy as np
import pytest

from findex.object_model import build_ranked_index
from findex.semantic import SemanticData, SemanticSearch, build_embeddings, chunks, normalize, rrf


class FakeEncoder:
    def encode(self, texts):
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, 0 if "python" in text.lower() else 1] = 1.0
        return vectors


def test_chunk_overlap():
    assert list(chunks("one two three four five", size=3, overlap=1)) == [
        "one two three", "three four five"
    ]


def test_normalization():
    vectors = np.zeros((1, 384), np.float32)
    vectors[0, :2] = [3, 4]
    assert np.linalg.norm(normalize(vectors)[0]) == pytest.approx(1.0)


def test_rrf():
    ranked = rrf([[1, 2, 3], [2, 1, 4]])
    assert [doc for doc, _ in ranked] == [1, 2, 3, 4]


def test_semantic_build_load_and_search(tiny_corpus, tmp_path):
    index = build_ranked_index(tiny_corpus)
    encoder = FakeEncoder()
    path = tmp_path / "embeddings"
    built = build_embeddings(index, tiny_corpus, path, encoder, batch_size=2)
    loaded = SemanticData.load(path, index)
    assert np.array_equal(loaded.matrix, built.matrix)
    assert loaded.matrix.shape == (4, 384)
    assert isinstance(loaded.matrix, np.memmap)
    engine = SemanticSearch(loaded, encoder)
    semantic = engine.search(index, "python", mode="semantic", k=3)
    assert len(semantic) == 3
    assert all(hit.score == pytest.approx(1.0) for hit in semantic)
    hybrid = engine.search(index, "python", mode="hybrid", k=3)
    assert len(hybrid) == 3
    assert len({hit.doc_id for hit in hybrid}) == 3
    index.close()
