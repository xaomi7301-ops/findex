"""Lab 08 web modes use in-memory fake embeddings, never the live model."""

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from findex.object_model import build_ranked_index
from findex.semantic import SemanticSearch, build_embeddings
from findex.web import SemanticRuntime, Settings, create_app, get_index, get_semantic


class FakeEncoder:
    def encode(self, texts):
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, 0 if "python" in text.lower() else 1] = 1.0
        return vectors


def test_modes_and_pagination(tiny_corpus: Path, tmp_path: Path):
    index = build_ranked_index(tiny_corpus)
    encoder = FakeEncoder()
    data = build_embeddings(index, tiny_corpus, tmp_path / "vectors", encoder)
    runtime = SemanticRuntime(SemanticSearch(data, encoder))
    app = create_app(Settings(index_path=tmp_path / "missing.bin"))
    app.dependency_overrides[get_index] = lambda: index
    app.dependency_overrides[get_semantic] = lambda: runtime
    with TestClient(app) as client:
        for mode in ("lexical", "semantic", "hybrid"):
            first = client.get("/search", params={"q": "python", "mode": mode, "k": 1})
            second = client.get("/search", params={"q": "python", "mode": mode, "k": 1, "page": 2})
            assert first.status_code == second.status_code == 200
            assert first.json()["mode"] == mode
            assert first.json()["results"][0]["doc_id"] != second.json()["results"][0]["doc_id"]
        assert client.get("/search", params={"q": "python", "mode": "bad"}).status_code == 422
        response = client.get("/", params={"q": "python", "mode": "hybrid", "k": 1})
        assert response.status_code == 200
        assert "Hybrid · RRF" in response.text
        assert "mode=hybrid" in response.text
    index.close()


def test_semantic_not_configured(tiny_corpus: Path, tmp_path: Path):
    index = build_ranked_index(tiny_corpus)
    app = create_app(Settings(index_path=tmp_path / "missing.bin"))
    app.dependency_overrides[get_index] = lambda: index
    with TestClient(app) as client:
        assert client.get("/search", params={"q": "python", "mode": "semantic"}).status_code == 503
        assert client.get("/search", params={"q": "python"}).status_code == 200
    index.close()
