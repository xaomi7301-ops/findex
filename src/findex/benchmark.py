"""Reproducible Lab 02 benchmark: posting memory, search engines and disk formats."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Literal, TypedDict

from findex.index import InvertedIndex, StorageName, build_index
from findex.search import search
from findex.store import load, save


class FormatRow(TypedDict):
    format: str
    file_bytes: int
    save_s: float
    load_s: float


class StorageRow(FormatRow):
    storage: StorageName
    documents: int
    tokens: int
    vocabulary: int
    build_s: float
    build_peak_mib: float


class SearchEngineRow(TypedDict):
    median_us: float
    result_count: int


class QueryRow(TypedDict):
    query: str
    merge: SearchEngineRow
    set: SearchEngineRow


class PositionsRow(TypedDict):
    without_positions_bytes: int
    with_positions_bytes: int
    ratio: float


def benchmark_format(
    index: InvertedIndex, path: Path, format: Literal["pickle", "json"]
) -> FormatRow:
    start = time.perf_counter()
    save(index, path, format=format)
    save_time = time.perf_counter() - start
    start = time.perf_counter()
    restored = load(path, format=format)
    load_time = time.perf_counter() - start
    assert restored.doc_lengths == index.doc_lengths
    assert restored.vocabulary_size == index.vocabulary_size
    return FormatRow(
        format=format,
        file_bytes=path.stat().st_size,
        save_s=round(save_time, 6),
        load_s=round(load_time, 6),
    )


def benchmark_storage(
    root: Path, storage: StorageName, output: Path
) -> tuple[StorageRow, InvertedIndex]:
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
    row = StorageRow(
        storage=storage,
        documents=index.document_count,
        tokens=index.token_count,
        vocabulary=index.vocabulary_size,
        build_s=round(seconds, 6),
        build_peak_mib=round(peak / (1024 * 1024), 3),
        **saved,
    )
    return row, index


def _time_engine(
    index: InvertedIndex, query: str, engine: Literal["merge", "set"], repetitions: int
) -> SearchEngineRow:
    times: list[float] = []
    result_count = 0
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        found = search(index, query, engine=engine)
        times.append((time.perf_counter_ns() - start) / 1000.0)
        result_count = len(found)
    return {"median_us": round(statistics.median(times), 3), "result_count": result_count}


def _query_benchmark(index: InvertedIndex, query: str, repetitions: int) -> QueryRow:
    merge = _time_engine(index, query, "merge", repetitions)
    set_result = _time_engine(index, query, "set", repetitions)
    if merge["result_count"] != set_result["result_count"]:
        raise AssertionError("Search engines disagreed")
    return {"query": query, "merge": merge, "set": set_result}


def benchmark_search(index: InvertedIndex, repetitions: int) -> list[QueryRow]:
    if index.vocabulary_size < 2:
        return []
    if index.storage == "array":
        raise ValueError("The search benchmark requires a regular slots index")
    ranked = sorted(index.postings, key=lambda term: (-len(index.postings[term]), term))
    rare = sorted(index.postings, key=lambda term: (len(index.postings[term]), term))
    return [
        _query_benchmark(index, f"{ranked[0]} {ranked[1]}", repetitions),
        _query_benchmark(index, f"{rare[0]} {rare[1]}", repetitions),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure posting and search trade-offs")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/lab02-results.json"))
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--measure-positions", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    rows: list[StorageRow] = []
    with tempfile.TemporaryDirectory(prefix="findex-lab02-") as temp:
        folder = Path(temp)
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
        positions: PositionsRow | None = None
        if args.measure_positions:
            print("Measuring positions ...", flush=True)
            position_index = build_index(args.root, with_positions=True)
            with_position = benchmark_format(position_index, folder / "positions.bin", "pickle")
            positions = {
                "without_positions_bytes": formats[0]["file_bytes"],
                "with_positions_bytes": with_position["file_bytes"],
                "ratio": round(with_position["file_bytes"] / formats[0]["file_bytes"], 3),
            }

    report: dict[str, object] = {
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
        print(
            f"{row['format']:<6} | {row['file_bytes']:>10} | "
            f"{row['save_s']:>6.3f} | {row['load_s']:>6.3f}"
        )
    print("\nQUERY | MERGE median us | SET median us | RESULTS")
    for row in queries:
        print(
            f"{row['query']} | {row['merge']['median_us']} | "
            f"{row['set']['median_us']} | {row['merge']['result_count']}"
        )
    if positions:
        print(f"\nPositions file-size ratio: {positions['ratio']}x")
    print(f"\nSaved measurements to: {args.out}")


if __name__ == "__main__":
    main()
