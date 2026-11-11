"""Lab 02: index construction, hashing, Boolean query, persistence, CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from array import array
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from findex.index import DocMeta, Posting, build_index, posting_doc_ids
from findex.search import merge_and, merge_not, merge_or, search
from findex.store import load, save


@pytest.fixture
def corpus(tmp_path):
    documents = {
        "alpha.txt": "Python Python code\nThis is a test",
        "beta.txt": "Python and search engine",
        "gamma.txt": "Rust and documentation",
        "delta.txt": "search engine tools",
    }
    for name, text in documents.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def names(index, query, engine="merge"):
    return {index.doc_meta[doc_id].path for doc_id in search(index, query, engine=engine)}


def test_posting_and_meta_frozen_hashable():
    posting = Posting(2, 3)
    meta = DocMeta("2.txt", "Document")
    assert {posting} == {Posting(2, 3)}
    assert {meta} == {DocMeta("2.txt", "Document")}
    with pytest.raises(FrozenInstanceError):
        posting.tf = 5


def test_inverted_index_tfs_and_lengths(corpus):
    index = build_index(corpus)
    assert index.document_count == 4
    assert index.doc_lengths.keys() == index.doc_meta.keys()
    python_postings = index.postings["python"]
    assert len(python_postings) == 2
    assert [p.doc_id for p in python_postings] == sorted(p.doc_id for p in python_postings)
    assert {index.doc_meta[p.doc_id].path: p.tf for p in python_postings} == {
        "alpha.txt": 2,
        "beta.txt": 1,
    }
    assert sum(index.doc_lengths.values()) == index.token_count


def test_storage_alternatives_equivalent(corpus):
    data = [build_index(corpus, storage=kind) for kind in ("plain", "slots", "array")]
    assert len({obj.token_count for obj in data}) == 1
    for index in data:
        assert names(index, "python") == {"alpha.txt", "beta.txt"}
        assert names(index, "search OR rust") == {"beta.txt", "gamma.txt", "delta.txt"}
        for posting in index.postings.values():
            assert posting_doc_ids(posting) == sorted(posting_doc_ids(posting))
    assert isinstance(data[-1].postings["python"], array)


def test_positions_switch(corpus):
    index = build_index(corpus, with_positions=True)
    alpha_id = next(k for k, meta in index.doc_meta.items() if meta.path == "alpha.txt")
    post = next(p for p in index.postings["python"] if p.doc_id == alpha_id)
    assert post.tf == 2
    assert post.positions == (0, 1)
    with pytest.raises(ValueError, match="positions"):
        build_index(corpus, storage="array", with_positions=True)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("python", {"alpha.txt", "beta.txt"}),
        ("PYTHON search", {"beta.txt"}),
        ("python AND search", {"beta.txt"}),
        ("python OR rust", {"alpha.txt", "beta.txt", "gamma.txt"}),
        ("python NOT search", {"alpha.txt"}),
        ("NOT python", {"gamma.txt", "delta.txt"}),
        ("python OR NOT search", {"alpha.txt", "beta.txt", "gamma.txt"}),
        ("missingterm", set()),
        ("", set()),
    ],
)
def test_boolean_engines(corpus, query, expected):
    index = build_index(corpus)
    assert names(index, query) == expected
    assert names(index, query, engine="set") == expected


def test_invalid_queries(corpus):
    index = build_index(corpus)
    for query in ("OR python", "python OR", "python AND OR rust", "NOT"):
        with pytest.raises(ValueError):
            search(index, query)
    with pytest.raises(ValueError):
        search(index, "python", engine="bad")


def test_two_pointer_functions():
    assert merge_and([1, 3, 5], [2, 3, 6]) == [3]
    assert merge_or([1, 3, 5], [2, 3, 6]) == [1, 2, 3, 5, 6]
    assert merge_not([1, 3, 5], [2, 3, 6]) == [1, 5]
    assert merge_and([], [2]) == []
    assert merge_or([], [2]) == [2]
    assert merge_not([1], []) == [1]


@pytest.mark.parametrize("storage", ["slots", "plain", "array"])
@pytest.mark.parametrize("format", ["pickle", "json"])
def test_roundtrip(corpus, tmp_path, storage, format):
    index = build_index(corpus, storage=storage)
    filename = tmp_path / ("index.json" if format == "json" else "index.bin")
    save(index, filename, format=format)
    loaded = load(filename, format=format)
    assert loaded.doc_meta == index.doc_meta
    assert loaded.doc_lengths == index.doc_lengths
    assert loaded.vocabulary_size == index.vocabulary_size
    for term in ("python", "search", "missingterm"):
        assert names(loaded, term) == names(index, term)


def test_positions_roundtrip(corpus, tmp_path):
    index = build_index(corpus, with_positions=True)
    for filename in (tmp_path / "p.bin", tmp_path / "p.json"):
        save(index, filename)
        loaded = load(filename)
        assert loaded.with_positions
        assert list(loaded.postings["python"])[0].positions


def test_cli_end_to_end(corpus, tmp_path):
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    saved = tmp_path / "index.bin"
    build = subprocess.run(
        [sys.executable, "-m", "findex.index", str(corpus), "--out", str(saved)],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )
    assert "Documents: 4" in build.stdout
    lookup = subprocess.run(
        [sys.executable, "-m", "findex.search", str(saved), "python OR rust", "--engine", "set"],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )
    assert "Found 3 document(s)" in lookup.stdout
