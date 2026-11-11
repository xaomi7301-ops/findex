"""Read a corpus lazily, retaining at most the current document's content.

A document is one UTF-8 .txt file. Badly encoded documents are skipped with a
warning; the rest of the corpus is still processed.
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Document(NamedTuple):
    """One textual document with a stable relative ID."""

    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield UTF-8 text documents in a directory tree, one file at a time.

    Does not first build a list of paths and never reads the entire corpus.
    A single .txt file is also accepted as the root path.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")

    if root.is_dir():
        paths = root.rglob("*.txt")
    elif root.is_file() and root.suffix.lower() == ".txt":
        paths = iter((root,))
    else:
        raise ValueError("Corpus must be a directory or a .txt file")

    for path in paths:
        if not path.is_file():
            continue
        try:
            # Only the current file is in memory; no list of all documents.
            with path.open("r", encoding="utf-8", errors="strict") as file:
                text = file.read()
        except (UnicodeError, OSError) as exc:
            logger.warning("Skipping unreadable document %s: %s", path, exc)
            continue

        relative = path.relative_to(root) if root.is_dir() else Path(path.name)
        yield Document(doc_id=relative.as_posix(), path=path, text=text)
