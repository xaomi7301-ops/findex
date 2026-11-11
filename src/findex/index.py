"""Lab 02: single-pass inverted index over Lab 01's lazy corpus pipeline."""

from __future__ import annotations

import argparse
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize


@dataclass(frozen=True)
class PlainPosting:
    """Baseline record: one instance dictionary per object."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Posting:
    """Compact, immutable posting; safe as a set member or dict key."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    """User-facing information kept for each document."""

    path: str
    title: str


type PostingList = list[Posting] | list[PlainPosting] | array[int]
type StorageName = Literal["plain", "slots", "array"]


@dataclass(slots=True)
class InvertedIndex:
    """An index of doc IDs ordered by the order of the streaming build."""

    postings: dict[str, PostingList]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    storage: StorageName = "slots"
    with_positions: bool = False

    @property
    def document_count(self) -> int:
        return len(self.doc_lengths)

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    @property
    def token_count(self) -> int:
        return sum(self.doc_lengths.values())


def posting_doc_ids(postings: PostingList) -> list[int]:
    """Extract sorted IDs from slotted/plain postings or packed (id, tf) pairs."""
    if isinstance(postings, array):
        return list(postings[::2])
    return [posting.doc_id for posting in postings]


def build_index(
    root: Path,
    *,
    storage: StorageName = "slots",
    with_positions: bool = False,
    limit: int | None = None,
    on_document: Callable[[int], None] | None = None,
) -> InvertedIndex:
    """Build from streamed docs, retaining only one document's Counter at a time.

    An ID is assigned incrementally, so all postings are automatically sorted
    by numeric doc_id. Array storage uses unsigned 32-bit (doc_id, tf) pairs.
    """
    if storage not in {"plain", "slots", "array"}:
        raise ValueError("storage must be plain, slots or array")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    if with_positions and storage == "array":
        raise ValueError("array storage does not support positions")

    postings: defaultdict[str, PostingList] = defaultdict(list)
    if storage == "array":
        postings = defaultdict(lambda: array("I"))

    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    constructor: type[PlainPosting] | type[Posting] = (
        PlainPosting if storage == "plain" else Posting
    )

    documents: Iterator[Document] = iter_documents(root)
    if limit is not None:
        documents = islice(documents, limit)
    for doc_id, document in enumerate(documents):
        counts: Counter[str] = Counter()
        positions: defaultdict[str, list[int]] | None = (
            defaultdict(list) if with_positions else None
        )
        length = 0
        for position, term in enumerate(tokenize(document.text)):
            counts[term] += 1
            if positions is not None:
                positions[term].append(position)
            length += 1

        doc_lengths[doc_id] = length
        title = next((line.strip() for line in document.text.splitlines() if line.strip()), "")
        doc_meta[doc_id] = DocMeta(
            path=document.doc_id,
            title=(title[:100] or Path(document.doc_id).stem),
        )
        for term, tf in counts.items():
            target = postings[term]
            if storage == "array":
                if doc_id > 0xFFFFFFFF or tf > 0xFFFFFFFF:
                    raise OverflowError("array('I') supports only unsigned 32-bit values")
                assert isinstance(target, array)
                target.extend((doc_id, tf))
            else:
                assert isinstance(target, list)
                term_positions = tuple(positions[term]) if positions is not None else ()
                target.append(constructor(doc_id, tf, term_positions))
        if on_document is not None:
            on_document(doc_id + 1)

    return InvertedIndex(
        postings=dict(postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        storage=storage,
        with_positions=with_positions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save a Boolean inverted index")
    parser.add_argument("root", type=Path, help="Directory of UTF-8 .txt files")
    parser.add_argument("--out", type=Path, default=Path("index.bin"))
    parser.add_argument("--format", choices=("auto", "pickle", "json"), default="auto")
    parser.add_argument("--storage", choices=("plain", "slots", "array"), default="slots")
    parser.add_argument(
        "--positions", action="store_true", help="Store token positions per posting"
    )
    args = parser.parse_args()

    from findex.index import build_index as canonical_build_index
    from findex.store import save

    tracemalloc.start()
    started = time.perf_counter()
    try:
        # -m executes this file as __main__; use canonical classes for portable pickle.
        index = canonical_build_index(
            args.root, storage=args.storage, with_positions=args.positions
        )
        build_seconds = time.perf_counter() - started
        save_started = time.perf_counter()
        save(index, args.out, format=args.format)
        save_seconds = time.perf_counter() - save_started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    print(f"Documents: {index.document_count:,}")
    print(f"Tokens: {index.token_count:,}")
    print(f"Vocabulary: {index.vocabulary_size:,}")
    print(f"Storage: {index.storage}; positions: {index.with_positions}")
    print(f"Build: {build_seconds:.3f} s; save: {save_seconds:.3f} s")
    print(f"Saved: {args.out} ({args.out.stat().st_size:,} bytes)")
    print(
        f"Peak Python memory (build + save): {peak / (1024 * 1024):.3f} MiB"
    )


if __name__ == "__main__":
    main()
