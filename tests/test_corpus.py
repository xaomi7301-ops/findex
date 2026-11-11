import inspect
import logging

import pytest

from findex.corpus import Document, iter_documents


def test_generator_yields_nested_documents(tmp_path):
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "one.txt").write_text("Привіт", encoding="utf-8")
    (tmp_path / "two.txt").write_text("Hello", encoding="utf-8")
    (tmp_path / "skip.md").write_text("not a .txt", encoding="utf-8")
    assert inspect.isgeneratorfunction(iter_documents)
    iterator = iter_documents(tmp_path)
    assert inspect.isgenerator(iterator)
    actual = list(iterator)
    assert len(actual) == 2
    assert {d.doc_id for d in actual} == {"nested/one.txt", "two.txt"}
    assert all(isinstance(d, Document) for d in actual)
    assert {d.text for d in actual} == {"Привіт", "Hello"}


def test_bad_encoding_is_skipped(tmp_path, caplog):
    (tmp_path / "invalid.txt").write_bytes(b"bad \xff utf8")
    (tmp_path / "valid.txt").write_text("works", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        docs = list(iter_documents(tmp_path))
    assert len(docs) == 1
    assert docs[0].text == "works"
    assert "Skipping unreadable document" in caplog.text


def test_missing_corpus_raises_on_consumption(tmp_path):
    docs = iter_documents(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        next(docs)


def test_single_file_supported(tmp_path):
    filename = tmp_path / "single.txt"
    filename.write_text("one doc", encoding="utf-8")
    docs = list(iter_documents(filename))
    assert [(d.doc_id, d.text) for d in docs] == [("single.txt", "one doc")]
