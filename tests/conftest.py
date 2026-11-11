"""Shared, intentionally tiny corpus fixtures for Lab 04 CLI and index tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    """Four UTF-8 documents, including Unicode and a short phrase."""
    texts = {
        "python.txt": "Python and async event loop. Python code works.",
        "guide.txt": "Async event loop and Python tasks.",
        "other.txt": "Java threads and processes.",
        "ukrainian.txt": "Україна, комп’ютер та програмування Python.",
    }
    for name, text in texts.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path
