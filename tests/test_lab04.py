"""Lab 04: installable Typer CLI, Unicode, property-based tests and packaging."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from findex.cli import app
from findex.index import build_index, posting_doc_ids
from findex.query import And, Or, Term, parse
from findex.ranking import BM25, TfIdf
from findex.search import merge_and, merge_not, merge_or
from findex.store import load, save
from findex.tokenize import tokenize

runner = CliRunner()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Hello WORLD", ["hello", "world"]),
        ("комп’ютер", ["комп'ютер"]),
        ("Україна", ["україна"]),
        ("e\u0301", ["é"]),
        ("two-words", ["two", "words"]),
        ("python3 192.168.1.1", ["python3", "192", "168", "1", "1"]),
        ("a_b", ["a", "b"]),
        ("", []),
    ],
)
def test_unicode_parameterized(source: str, expected: list[str]) -> None:
    assert list(tokenize(source)) == expected


def test_cli_help_is_installed_entry_point() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("index", "search", "stats"):
        assert name in result.stdout


def test_cli_index_search_and_stats(tiny_corpus: Path, tmp_path: Path) -> None:
    target = tmp_path / "index.bin"
    built = runner.invoke(app, ["index", str(tiny_corpus), "--out", str(target), "--positions"])
    assert built.exit_code == 0, built.output
    assert target.is_file()
    assert "4 documents" in built.stdout
    searched = runner.invoke(
        app,
        ["search", str(target), '"event loop" AND python', "--corpus", str(tiny_corpus)],
    )
    assert searched.exit_code == 0, searched.output
    assert "Results:" in searched.stdout
    assert "event" in searched.stdout
    stat_result = runner.invoke(app, ["stats", str(target)])
    assert stat_result.exit_code == 0
    assert "Vocabulary" in stat_result.stdout


def test_cli_jsonl_and_stderr(tiny_corpus: Path, tmp_path: Path) -> None:
    target = tmp_path / "index.json"
    assert runner.invoke(app, ["index", str(tiny_corpus), "--out", str(target)]).exit_code == 0
    result = runner.invoke(app, ["-vv", "search", str(target), "python", "--json", "--k", "2"])
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(records) == 2
    assert all("score" in record and "path" in record for record in records)
    assert "query-cache" not in result.stdout
    # -vv diagnostics are allowed on stderr but must not corrupt JSONL stdout.


@pytest.mark.parametrize("query", ["AND", "python OR", "python ("])
def test_cli_bad_query_has_one_line_error(tmp_path: Path, tiny_corpus: Path, query: str) -> None:
    target = tmp_path / "index.bin"
    assert runner.invoke(app, ["index", str(tiny_corpus), "--out", str(target)]).exit_code == 0
    result = runner.invoke(app, ["search", str(target), query])
    assert result.exit_code != 0
    assert "Error:" in result.output
    assert "Traceback" not in result.output


def test_cli_missing_file_exits_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["search", str(tmp_path / "missing.bin"), "python"])
    assert result.exit_code != 0
    assert "Error:" in result.output
    assert "Traceback" not in result.output


def test_parser_tree_not_string_comparison() -> None:
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))


def test_scorer_protocol_structural(tiny_corpus: Path) -> None:
    from findex.object_model import Index
    from findex.ranking import Scorer

    ix = Index(build_index(tiny_corpus))
    try:
        scorers: list[Scorer] = [TfIdf(), BM25()]
        assert len(scorers) == 2
        assert all(scorer.score("python", ix["python"][0], ix) >= 0 for scorer in scorers)
    finally:
        ix.close()


@settings(max_examples=35, deadline=None)
@given(
    st.lists(st.integers(min_value=0, max_value=1000), max_size=35),
    st.lists(st.integers(min_value=0, max_value=1000), max_size=35),
)
def test_property_merge_matches_sets(left: list[int], right: list[int]) -> None:
    a, b = sorted(set(left)), sorted(set(right))
    assert merge_and(a, b) == sorted(set(a) & set(b))
    assert merge_or(a, b) == sorted(set(a) | set(b))
    assert merge_not(a, b) == sorted(set(a) - set(b))


@settings(max_examples=25, deadline=None)
@given(
    st.lists(
        st.text(
            alphabet=st.characters(blacklist_categories=("Cs",)),
            min_size=1,
            max_size=50,
        ),
        min_size=0,
        max_size=5,
    )
)
def test_property_postings_sorted(texts: list[str]) -> None:
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        for number, text in enumerate(texts):
            (folder / f"{number}.txt").write_text(text, encoding="utf-8")
        index = build_index(folder)
        for postings in index.postings.values():
            ids = posting_doc_ids(postings)
            assert ids == sorted(set(ids))


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(st.lists(st.sampled_from(["python", "async", "Привіт", "code", ""]), max_size=8))
def test_property_pickle_and_json_roundtrip(words: list[str]) -> None:
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        (folder / "one.txt").write_text(" ".join(words), encoding="utf-8")
        original = build_index(folder, with_positions=True)
        for suffix in (".bin", ".json"):
            path = folder / f"index{suffix}"
            save(original, path)
            restored = load(path)
            assert restored.doc_lengths == original.doc_lengths
            assert restored.doc_meta == original.doc_meta
            assert restored.with_positions == original.with_positions
            assert restored.postings == original.postings


@pytest.mark.slow
def test_installed_module_as_python_subprocess(tiny_corpus: Path, tmp_path: Path) -> None:
    import subprocess
    import sys

    path = tmp_path / "slow.bin"
    process = subprocess.run(
        [sys.executable, "-m", "findex.cli", "index", str(tiny_corpus), "--out", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert path.exists()
