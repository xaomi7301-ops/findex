"""Single-pass corpus statistics and eager-vs-lazy memory benchmarking."""

from __future__ import annotations

import argparse
import gc
import json
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass
class CorpusStats:
    mode: str
    documents: int
    tokens: int
    counts: Counter[str]
    elapsed_seconds: float = 0.0
    peak_bytes: int = 0

    @property
    def vocabulary_size(self) -> int:
        return len(self.counts)

    def top_terms(self, count: int = 50) -> list[tuple[str, int]]:
        return sorted(self.counts.items(), key=lambda item: (-item[1], item[0]))[:count]

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "documents": self.documents,
            "tokens": self.tokens,
            "vocabulary_size": self.vocabulary_size,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "peak_bytes": self.peak_bytes,
            "peak_mib": round(self.peak_bytes / (1024 * 1024), 3),
            "top_50": self.top_terms(),
        }


def collect_stats(root: Path, limit: int | None = None, *, eager: bool = False) -> CorpusStats:
    """Process a corpus in one pass. Eager mode deliberately materializes lists."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")

    counts: Counter[str] = Counter()
    document_count = 0
    token_count = 0
    docs = islice(iter_documents(root), limit)

    if eager:
        document_list = list(docs)
        token_lists = [list(tokenize(doc.text)) for doc in document_list]
        document_count = len(document_list)
        for token_list in token_lists:
            counts.update(token_list)
            token_count += len(token_list)
    else:
        for doc in docs:
            document_count += 1
            for token in tokenize(doc.text):
                counts[token] += 1
                token_count += 1

    return CorpusStats(
        mode="eager" if eager else "lazy",
        documents=document_count,
        tokens=token_count,
        counts=counts,
    )


def measure(root: Path, limit: int | None = None, *, eager: bool = False) -> CorpusStats:
    """Measure a run using tracemalloc and wall-clock time."""
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    try:
        stats = collect_stats(root, limit, eager=eager)
        elapsed = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    stats.elapsed_seconds = elapsed
    stats.peak_bytes = peak
    return stats


def print_stats(stats: CorpusStats) -> None:
    print(f"Mode:              {stats.mode}")
    print(f"Documents:         {stats.documents:,}")
    print(f"Tokens:            {stats.tokens:,}")
    print(f"Vocabulary:        {stats.vocabulary_size:,}")
    print(f"Elapsed:           {stats.elapsed_seconds:.3f} s")
    print(f"Peak tracemalloc:  {stats.peak_bytes / (1024 * 1024):.3f} MiB")
    print("Top 50 terms:")
    for index, (token, count) in enumerate(stats.top_terms(), start=1):
        print(f"  {index:2}. {token:<26} {count:,}")


def print_comparison(lazy: CorpusStats, eager: CorpusStats) -> None:
    if (lazy.documents, lazy.tokens, lazy.counts) != (eager.documents, eager.tokens, eager.counts):
        raise AssertionError("The lazy and eager versions produced different statistics!")
    print("\n| Version | Documents | Peak memory (MiB) | Elapsed (s) |")
    print("|---|---:|---:|---:|")
    for result in (eager, lazy):
        print(
            f"| {result.mode} | {result.documents} | "
            f"{result.peak_bytes / (1024 * 1024):.3f} | {result.elapsed_seconds:.3f} |"
        )
    print("Results verified: counts and full vocabulary are identical.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure document statistics for a UTF-8 corpus")
    parser.add_argument("root", type=Path, help="Folder with .txt files (recursively)")
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument("--eager", action="store_true", help="Use deliberately eager lists")
    parser.add_argument("--compare", action="store_true", help="Benchmark eager and lazy")
    parser.add_argument("--json", type=Path, default=None, metavar="PATH", help="Save results")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.compare and args.eager:
        parser.error("Use either --compare or --eager, not both")

    if args.compare:
        lazy = measure(args.root, args.limit)
        eager = measure(args.root, args.limit, eager=True)
        print_stats(lazy)
        print_comparison(lazy, eager)
        results = {"lazy": lazy.as_dict(), "eager": eager.as_dict()}
    else:
        result = measure(args.root, args.limit, eager=args.eager)
        print_stats(result)
        results = {result.mode: result.as_dict()}

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved results: {args.json}")


if __name__ == "__main__":
    main()
