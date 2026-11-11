"""Lab 03: object model, AST equality, phrase search, scoring and CLI tests."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from findex.index import Posting, build_index
from findex.object_model import Index, build_ranked_index, open_index
from findex.query import And, Not, Or, Phrase, Term, parse
from findex.ranking import BM25, TfIdf, ranked_search, snippet_for
from findex.store import save
from findex.timing import timed


@pytest.fixture
def corpus(tmp_path):
    documents = {
        "alpha.txt": "Python python and asyncio event loop and more notes about python.",
        "beta.txt": "Python async event loop.",
        "gamma.txt": "Java event loop.",
        "delta.txt": "async await event loop python.",
    }
    for name, text in documents.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


@pytest.fixture
def index(corpus):
    result = build_ranked_index(corpus)
    yield result
    result.close()


def test_mapping_dunders_and_properties(index):
    assert isinstance(index, Index)
    assert len(index) > 0
    assert "python" in index
    assert "unfindable" not in index
    assert "python" in list(index)
    assert index["python"]
    with pytest.raises(KeyError):
        index["unfindable"]
    assert repr(index).startswith("Index(terms=")
    assert index.num_docs == 4
    assert index.df("python") == 3
    assert index.doc_length(0) > 0
    assert index.avg_doc_length > 0
    assert "avg_doc_length" in index.__dict__


def test_close_raises_and_is_idempotent(corpus):
    index = build_ranked_index(corpus)
    index.close()
    index.close()
    assert index.closed
    with pytest.raises(RuntimeError, match="closed"):
        len(index)


def test_context_manager_closes_even_on_exception(corpus, tmp_path):
    raw = build_index(corpus, with_positions=True)
    path = tmp_path / "index.bin"
    save(raw, path)
    owned = None
    with pytest.raises(ZeroDivisionError):
        with open_index(path, corpus_root=corpus) as ix:
            owned = ix
            assert ix["python"]
            raise ZeroDivisionError
    assert owned is not None and owned.closed
    with open_index(path, corpus_root=corpus) as ix:
        assert ix.num_docs == 4
        assert ix.has_positions


def test_tree_equality_and_operators():
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))
    assert Term("a") & (Term("b") | ~Term("c")) == And(
        Term("a"), Or(Term("b"), Not(Term("c")))
    )
    with pytest.raises(FrozenInstanceError):
        node = Term("hello")
        node.value = "other"


def test_complex_query_tree_and_results(index):
    expected = And(
        And(
            And(Term("python"), Or(Term("async"), Term("await"))),
            Not(Term("java")),
        ),
        Phrase(("event", "loop")),
    )
    query = 'python AND (async OR await) NOT java "event loop"'
    assert parse(query) == expected
    paths = {index.doc_meta[doc_id].path for doc_id in parse(query).evaluate(index)}
    assert paths == {"beta.txt", "delta.txt"}


def test_phrase_contiguous_repetition_and_negative(index):
    assert len(Phrase(("event", "loop")).evaluate(index)) == 4
    assert not Phrase(("loop", "event")).evaluate(index)
    alpha_id = next(doc_id for doc_id, meta in index.doc_meta.items()
                    if meta.path == "alpha.txt")
    assert Phrase(("python", "python")).evaluate(index) == {alpha_id}
    assert parse('"event loop"').evaluate(index) == set(index.doc_meta)


def test_phrase_requires_positions(corpus):
    ix = Index(build_index(corpus, with_positions=False), corpus_root=corpus)
    try:
        with pytest.raises(ValueError, match="positions"):
            parse('"event loop"').evaluate(ix)
    finally:
        ix.close()


@pytest.mark.parametrize(
    "query",
    [
        "",
        "AND python",
        "python OR",
        "a OR OR b",
        "a (b",
        "a ) b",
        '"unterminated phrase',
        '""',
    ],
)
def test_invalid_query_rejected(query):
    with pytest.raises(ValueError):
        parse(query)


def test_tfidf_and_bm25_are_interchangeable(index):
    for scorer in (TfIdf(), BM25()):
        results = ranked_search(index, "python OR java", scorer=scorer, k=3)
        assert 1 <= len(results) <= 3
        assert sorted(results) == results
        assert results[0].score >= results[-1].score
        assert results[0].path.endswith(".txt")


def test_rarity_is_worth_more(index):
    scorer = BM25()
    doc_id = next(iter(parse("java").evaluate(index)))
    assert scorer.score("java", Posting(doc_id, 1), index) > scorer.score(
        "event", Posting(doc_id, 1), index
    )


def test_bm25_saturation(index):
    scorer = BM25()
    doc_id = next(iter(parse("java").evaluate(index)))
    def score(tf):
        return scorer.score("java", Posting(doc_id, tf), index)
    assert score(20) - score(19) < score(2) - score(1)


def test_bm25_doc_length_penalty(index):
    scorer = BM25()
    short = min(index.doc_lengths, key=index.doc_length)
    long = max(index.doc_lengths, key=index.doc_length)
    assert index.doc_length(short) < index.doc_length(long)
    assert scorer.score("python", Posting(short, 1), index) > scorer.score(
        "python", Posting(long, 1), index
    )


def test_snippet_highlights_and_cache_hit(index, caplog):
    with caplog.at_level(logging.INFO):
        first = ranked_search(index, "python OR java", scorer=BM25(), k=3)
        second = ranked_search(index, "python OR java", scorer=BM25(), k=3)
    assert first == second
    assert any("[[" in result.snippet for result in first)
    alpha_id = next(doc_id for doc_id, meta in index.doc_meta.items()
                    if meta.path == "alpha.txt")
    assert "[[python]]" in snippet_for(index, alpha_id, ("python",))
    assert "query-cache MISS" in caplog.text
    assert "query-cache HIT" in caplog.text
    assert index.cache_info().hits >= 1


def test_decorator_preserves_name_and_logs(caplog):
    @timed
    def multiply(value):
        """Just a test."""
        return value * 2

    assert multiply.__name__ == "multiply"
    assert multiply.__doc__ == "Just a test."
    with caplog.at_level(logging.INFO):
        assert multiply(5) == 10
    assert "@timed multiply" in caplog.text


def test_invalid_bm25_parameters():
    with pytest.raises(ValueError):
        BM25(k1=0)
    with pytest.raises(ValueError):
        BM25(b=1.2)


def test_command_line_build_and_search(corpus, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    output = tmp_path / "lab03.bin"
    built = subprocess.run(
        [
            sys.executable, "-m", "findex.ranked", "build",
            str(corpus), "--out", str(output),
        ],
        check=True, text=True, capture_output=True, env=env,
    )
    assert output.exists()
    assert "Positions enabled: True" in built.stdout
    searched = subprocess.run(
        [
            sys.executable, "-m", "findex.ranked", "search",
            str(output), '"event loop" AND python',
            "--corpus", str(corpus), "--repeat", "2",
        ],
        check=True, text=True, capture_output=True, env=env,
    )
    assert "ranked results" in searched.stdout
    assert "[[" in searched.stdout
    assert "query-cache HIT" in searched.stderr
