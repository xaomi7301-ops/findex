"""Lab 08 scorer-only benchmark: repeated, warmed, same corpus and candidates."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from findex.numpy_ranking import score_numpy
from findex.object_model import open_index
from findex.query import parse
from findex.ranking import BM25, TfIdf, score_python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/lab08-scorer.json"))
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    rows = []
    with open_index(args.index) as index:
        start = time.perf_counter()
        numeric = index.numpy_postings
        conversion_ms = (time.perf_counter() - start) * 1000
        queries = {
            "common": "the",
            "rare": "frozenset",
            "three terms": "python OR asyncio OR thread",
        }
        for label, query in queries.items():
            matched = index.matching_ids(query)
            terms = tuple(dict.fromkeys(parse(query).positive_terms()))
            for scorer in (BM25(), TfIdf()):
                expected = score_python(index, matched, terms, scorer, 10)
                actual = score_numpy(index, matched, terms, scorer, 10)
                if [x[0] for x in expected] != [x[0] for x in actual]:
                    raise AssertionError(f"Ranking mismatch: {query}, {scorer}")
                for a, b in zip(expected, actual, strict=True):
                    if abs(a[1] - b[1]) > 1e-10 * max(1.0, abs(a[1])):
                        raise AssertionError(f"Score mismatch: {a} vs {b}")
                py_times = []
                np_times = []
                for repetition in range(args.repeats):
                    # Alternate order to reduce systematic warm-cache bias.
                    if repetition % 2:
                        methods = ((score_numpy, np_times), (score_python, py_times))
                    else:
                        methods = ((score_python, py_times), (score_numpy, np_times))
                    for method, samples in methods:
                        start = time.perf_counter_ns()
                        method(index, matched, terms, scorer, 10)
                        samples.append((time.perf_counter_ns() - start) / 1e6)
                py_ms = statistics.median(py_times)
                np_ms = statistics.median(np_times)
                row = {
                    "query": query,
                    "category": label,
                    "scorer": type(scorer).__name__,
                    "docs_matched": len(matched),
                    "python_ms": py_ms,
                    "numpy_ms": np_ms,
                    "speedup": py_ms / np_ms,
                    "python_min_ms": min(py_times),
                    "python_max_ms": max(py_times),
                    "numpy_min_ms": min(np_times),
                    "numpy_max_ms": max(np_times),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
        report = {
            "documents": index.num_docs,
            "vocabulary": len(index),
            "conversion_ms": conversion_ms,
            "numeric_buffer_bytes": numeric.nbytes,
            "repeats": args.repeats,
            "k": 10,
            "rows": rows,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
