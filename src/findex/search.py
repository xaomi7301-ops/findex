"""Lab 02 Boolean retrieval with sorted two-pointer merge or hash-based sets."""

from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path

from findex.index import InvertedIndex, posting_doc_ids
from findex.store import load
from findex.tokenize import tokenize


def merge_and(left: list[int], right: list[int]) -> list[int]:
    """Intersection of sorted unique integer IDs in O(len(left) + len(right))."""
    result: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
    return result


def merge_or(left: list[int], right: list[int]) -> list[int]:
    """Union of sorted unique integer IDs without duplicate results."""
    result: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_not(left: list[int], exclude: list[int]) -> list[int]:
    """Sorted difference of left minus excluded IDs, using two pointers."""
    result: list[int] = []
    i = j = 0
    while i < len(left) and j < len(exclude):
        if left[i] == exclude[j]:
            i += 1
            j += 1
        elif left[i] < exclude[j]:
            result.append(left[i])
            i += 1
        else:
            j += 1
    result.extend(left[i:])
    return result


def _lex(query: str) -> list[str]:
    """Split operator words and normalize term words with Lab 01's tokenizer."""
    tokens: list[str] = []
    for part in query.split():
        if part.upper() in {"AND", "OR", "NOT"}:
            tokens.append(part.upper())
        else:
            tokens.extend(tokenize(part))
    return tokens


def search(index: InvertedIndex, query: str, *, engine: str = "merge") -> list[int]:
    """Evaluate left-to-right Boolean query, with adjacent terms as AND.

    Examples: 'python typing', 'python OR json', 'python NOT legacy',
    'python OR NOT deprecated'. NOT is a unary complement over all documents.
    Parentheses and operator precedence are intentionally deferred to Lab 03.
    """
    if engine not in {"merge", "set"}:
        raise ValueError("engine must be 'merge' or 'set'")
    tokens = _lex(query)
    if not tokens:
        return []

    universe = sorted(index.doc_lengths)
    result: list[int] | set[int] | None = None
    operator = "AND"
    awaiting_term = False
    offset = 0
    while offset < len(tokens):
        part = tokens[offset]
        if part in {"AND", "OR"}:
            if result is None or awaiting_term:
                raise ValueError(f"Unexpected binary operator: {part}")
            operator = part
            awaiting_term = True
            offset += 1
            if offset >= len(tokens):
                raise ValueError("Query cannot end with AND or OR")
            continue

        negate = False
        while offset < len(tokens) and tokens[offset] == "NOT":
            negate = not negate
            offset += 1
        if offset >= len(tokens):
            raise ValueError("NOT requires a term")
        part = tokens[offset]
        if part in {"AND", "OR"}:
            raise ValueError("Expected a term, not an operator")
        offset += 1
        ids = posting_doc_ids(index.postings.get(part, []))

        if engine == "merge":
            current: list[int] = merge_not(universe, ids) if negate else ids
            if result is None:
                result = current
            elif operator == "OR":
                result = merge_or(result, current)
            else:
                result = merge_and(result, current)
        else:
            current_set = set(universe) - set(ids) if negate else set(ids)
            if result is None:
                result = current_set
            elif operator == "OR":
                result |= current_set
            else:
                result &= current_set
        operator = "AND"
        awaiting_term = False

    return sorted(result) if engine == "set" else result if result is not None else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a saved findex inverted index")
    parser.add_argument("path", type=Path, help="Trusted .bin (pickle) or .json index")
    parser.add_argument("query", help="Boolean query: words, AND, OR, NOT")
    parser.add_argument("--engine", choices=("merge", "set"), default="merge")
    parser.add_argument("--limit", type=int, default=10, help="Maximum result rows")
    args = parser.parse_args()

    tracemalloc.start()
    started = time.perf_counter()
    try:
        index = load(args.path)
        load_time = time.perf_counter() - started
        started_search = time.perf_counter()
        result = search(index, args.query, engine=args.engine)
        search_time = time.perf_counter() - started_search
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    print(f"Found {len(result):,} document(s) with engine={args.engine}")
    for doc_id in result[: max(args.limit, 0)]:
        meta = index.doc_meta[doc_id]
        print(f"  {doc_id:>5} | {meta.path} | {meta.title}")
    print(f"Load: {load_time:.3f} s; search: {search_time:.6f} s")
    print(f"Peak Python memory (load + search): {peak / (1024 * 1024):.3f} MiB")


if __name__ == "__main__":
    main()
