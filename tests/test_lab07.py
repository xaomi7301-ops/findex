"""Web tests inject an in-memory index, without loading a real saved index."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.index import build_index
from findex.object_model import Index
from findex.web import Settings, create_app, get_index


@pytest.fixture
def client(tiny_corpus: Path, tmp_path: Path):
    application = create_app(Settings(index_path=tmp_path / "missing.bin"))
    index = Index(build_index(tiny_corpus, with_positions=True), corpus_root=tiny_corpus)
    application.dependency_overrides[get_index] = lambda: index
    with TestClient(application) as session:
        yield session
    index.close()


def test_search_and_pages(client: TestClient) -> None:
    one = client.get("/search", params={"q": "python", "k": 1})
    two = client.get("/search", params={"q": "python", "k": 1, "page": 2})
    assert one.status_code == two.status_code == 200
    assert one.json()["total"] == 3
    assert one.json()["results"][0]["doc_id"] != two.json()["results"][0]["doc_id"]
    assert one.headers["X-Request-ID"]


@pytest.mark.parametrize("query", ["", "?q=x&k=0", "?q=x&page=0", "?q=x&scorer=bad", "?q=("])
def test_invalid_queries(client: TestClient, query: str) -> None:
    assert client.get("/search" + query).status_code == 422


def test_document_stats_and_ui(client: TestClient) -> None:
    result = client.get("/search", params={"q": "python"}).json()["results"][0]
    doc_id = result["doc_id"]
    assert client.get(f"/docs/{doc_id}").json()["text"]
    assert client.get(f"/doc/{doc_id}").status_code == 200
    assert client.get("/docs/9999").status_code == 404
    assert client.get("/docs").status_code == 200
    assert client.get("/stats").json()["documents"] == 4
    assert client.get("/health").json() == {"status": "ok"}
    page = client.get("/", params={"q": "python", "k": 1})
    assert page.status_code == 200
    assert "<mark>" in page.text
    assert "Наступна" in page.text
    assert client.get("/static/style.css").status_code == 200


def test_health_missing(tmp_path: Path) -> None:
    application = create_app(Settings(index_path=tmp_path / "missing.bin"))
    with TestClient(application) as session:
        assert session.get("/health").status_code == 503


def test_clean_unexpected_error(tmp_path: Path) -> None:
    application = create_app(Settings(index_path=tmp_path / "missing.bin"))

    def broken() -> Index:
        raise RuntimeError("secret traceback")

    application.dependency_overrides[get_index] = broken
    with TestClient(application, raise_server_exceptions=False) as session:
        response = session.get("/search", params={"q": "python"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert response.headers["X-Request-ID"]
