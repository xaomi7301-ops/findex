"""Save/load an inverted index as pickle or interoperable JSON.

SECURITY: pickle.load can execute attacker-supplied code. Load only index files
that you generated yourself and control. Never unpickle downloaded indices.
"""

from __future__ import annotations

import json
import pickle
from array import array
from pathlib import Path

from findex.index import DocMeta, InvertedIndex, Posting

FORMAT_VERSION = 1


def _choose_format(path: Path, format: str) -> str:
    if format == "auto":
        return "json" if path.suffix.lower() == ".json" else "pickle"
    if format not in {"json", "pickle"}:
        raise ValueError("Unsupported index format")
    return format


def save(index: InvertedIndex, path: Path, *, format: str = "auto") -> None:
    """Persist an index; .json selects JSON, all other suffixes select pickle."""
    path = Path(path)
    chosen = _choose_format(path, format)
    path.parent.mkdir(parents=True, exist_ok=True)
    if chosen == "pickle":
        with path.open("wb") as stream:
            pickle.dump(index, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return

    postings = {}
    for term, entries in index.postings.items():
        if isinstance(entries, array):
            postings[term] = [[int(entries[i]), int(entries[i + 1]), []]
                              for i in range(0, len(entries), 2)]
        else:
            postings[term] = [[p.doc_id, p.tf, list(p.positions)] for p in entries]

    payload = {
        "version": FORMAT_VERSION,
        "storage": "slots",  # JSON reloads to standard compact Posting objects.
        "with_positions": index.with_positions,
        "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
        "doc_meta": {
            str(k): {"path": value.path, "title": value.title}
            for k, value in index.doc_meta.items()
        },
        "postings": postings,
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))


def load(path: Path, *, format: str = "auto") -> InvertedIndex:
    """Load a trusted pickle or a JSON index; JSON accepts no executable code."""
    path = Path(path)
    chosen = _choose_format(path, format)
    if chosen == "pickle":
        # WARNING: Loading an untrusted pickle may execute arbitrary Python code.
        with path.open("rb") as stream:
            result = pickle.load(stream)  # noqa: S301
        if not isinstance(result, InvertedIndex):
            raise ValueError("Pickle does not contain an InvertedIndex")
        return result

    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("version") != FORMAT_VERSION:
        raise ValueError("Unsupported JSON index version")
    return InvertedIndex(
        postings={
            term: [Posting(int(doc_id), int(tf), tuple(map(int, positions)))
                   for doc_id, tf, positions in entries]
            for term, entries in data["postings"].items()
        },
        doc_lengths={int(k): int(v) for k, v in data["doc_lengths"].items()},
        doc_meta={
            int(k): DocMeta(path=value["path"], title=value["title"])
            for k, value in data["doc_meta"].items()
        },
        storage="slots",
        with_positions=bool(data.get("with_positions", False)),
    )
