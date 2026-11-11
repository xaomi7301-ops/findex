"""Reproducible Lab 02 benchmark across posting storage, retrieval and formats."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path

from findex.index import InvertedIndex, build_index
from findex.search import search
from findex.store import load, save


def benchmark_format(index: InvertedIndex, path: Path, format: str) -> dict:
    t0 = time.perf_counter()
    save(index, path, format=format)
    save_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    restored = load(path, format=format)
    load_time = time.perf_counter() - t0
    assert restored.doc_lengths == index.doc_lengths
    assert restored.vocabulary_size == index.vocabulary_size
    return {
        "format": format,
        "file_bytes": path.stat().st_size,
        "save_s": round(save_time, 6),
        "load_s": round(load_time, 6),
    }


def benchmark_storage(root: Path, storage: str, output: Path) -> tuple[dict, InvertedIndex]:
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    try:
        index = build_index(root, storage=storage)
        seconds = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    saved = benchmark_format(index, output, "pickle")
    return {
        "storage": storage,
        "documents": index.document_count,
        "tokens": index.token_count,
        "vocabulary": index.vocabulary_size,
        "build_s": round(seconds, 6),
        "build_peak_mib": round(peak / (1024 * 1024), 3),
        **saved,
    }, index


def _query_benchmark(index: InvertedIndex, query: str, repetitions: int) -> dict:
    runs = {}
    for engine in ("merge", "set"):
        durations = []
        for _ in range(repetitions):
            start = time.perf_counter_ns()
            result = search(index, query, engine=engine)
            durations.append((time.perf_counter_ns() - start) / 1000.0)
        runs[engine] = {
            "median_us": round(statistics.median(durations), 3),
            "result_count": len(result),
        }
    if runs["merge"]["result_count"] != runs["set"]["result_count"]:
        raise AssertionError("Search engines disagreed")
    return {"query": query, **runs}


def benchmark_search(index: InvertedIndex, repetitions: int) -> list[dict]:
    if index.vocabulary_size < 2:
        return []
    ranked = sorted(index.postings, key=lambda term: (-len(index.postings[term]), term))
    rare = sorted(index.postings, key=lambda term: (len(index.postings[term]), term))
    # Packed arrays have two unsigned integers per posting.
    if index.storage == "array":
        raise ValueError("The search benchmark requires a regular slots index")
    return [
        _query_benchmark(index, f"{ranked[0]} {ranked[1]}", repetitions),
        _query_benchmark(index, f"{rare[0]} {rare[1]}", repetitions),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure storage and Boolean query trade-offs")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/lab02-results.json"))
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--measure-positions", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    rows = []
    with tempfile.TemporaryDirectory(prefix="findex-lab02-") as temporary:
        folder = Path(temporary)
        compact: InvertedIndex | None = None
        for storage in ("plain", "slots", "array"):
            print(f"Measuring {storage} on {args.root} ...", flush=True)
            row, index = benchmark_storage(args.root, storage, folder / f"{storage}.bin")
            rows.append(row)
            if storage == "slots":
                compact = index
            else:
                del index
                gc.collect()

        assert compact is not None
        formats = [
            benchmark_format(compact, folder / "slots.bin", "pickle"),
            benchmark_format(compact, folder / "slots.json", "json"),
        ]
        queries = benchmark_search(compact, args.repeats)
        positions = None
        if args.measure_positions:
            print("Measuring positions ...", flush=True)
            position_index = build_index(args.root, with_positions=True)
            with_position = benchmark_format(
                position_index, folder / "with-positions.bin", "pickle"
            )
            positions = {
                "without_positions_bytes": formats[0]["file_bytes"],
                "with_positions_bytes": with_position["file_bytes"],
                "ratio": round(with_position["file_bytes"] / formats[0]["file_bytes"], 3),
            }
    report = {
        "corpus": str(args.root),
        "repetitions": args.repeats,
        "storage": rows,
        "formats": formats,
        "queries": queries,
        "positions": positions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSTORAGE  | PEAK MiB | FILE bytes | BUILD s | SAVE s | LOAD s")
    for row in rows:
        print(
            f"{row['storage']:<8} | {row['build_peak_mib']:>8.3f} | "
            f"{row['file_bytes']:>10} | {row['build_s']:>7.3f} | "
            f"{row['save_s']:>6.3f} | {row['load_s']:>6.3f}"
        )
    print("\nFORMAT | FILE bytes | SAVE s | LOAD s")
    for row in formats:
        print(f"{row['format']:<6} | {row['file_bytes']:>10} | "
              f"{row['save_s']:>6.3f} | {row['load_s']:>6.3f}")
    print("\nQUERY | MERGE median us | SET median us | RESULTS")
    for row in queries:
        print(f"{row['query']} | {row['merge']['median_us']} | "
              f"{row['set']['median_us']} | {row['merge']['result_count']}")
    if positions:
        print(f"\nPositions file-size ratio: {positions['ratio']}x")
    print(f"\nSaved measurements to: {args.out}")


if __name__ == "__main__":
    main()
