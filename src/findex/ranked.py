"""Lab 03 CLI: build, ranked search, sanity checks and precision@5 evaluation.

Examples:
  python -m findex.ranked build data/python_docs --out index-lab03.bin
  python -m findex.ranked search index-lab03.bin 'python AND (async OR await)' --corpus data/python_docs
  python -m findex.ranked sanity index-lab03.bin --corpus data/python_docs
  python -m findex.ranked evaluate index-lab03.bin --corpus data/python_docs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from findex.object_model import build_ranked_index, open_index
from findex.ranking import BM25, TfIdf, ranked_search
from findex.store import save


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 03: ranked search with an object-model index")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build a positional index")
    build.add_argument("corpus", type=Path)
    build.add_argument("--out", type=Path, default=Path("index-lab03.bin"))
    build.add_argument("--no-positions", action="store_true")

    for name in ("search", "sanity", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("index", type=Path)
        command.add_argument("--corpus", type=Path, default=Path("data/python_docs"))
        if name == "search":
            command.add_argument("query")
            command.add_argument("--scorer", choices=("bm25", "tfidf"), default="bm25")
            command.add_argument("--k", type=int, default=5)
            command.add_argument("--repeat", type=int, default=2)
        if name == "evaluate":
            command.add_argument(
                "--labels", type=Path, default=Path("benchmarks/lab03-labels.json")
            )
            command.add_argument(
                "--out", type=Path, default=Path("benchmarks/lab03-results.json")
            )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "build":
        ix = build_ranked_index(args.corpus, positions=not args.no_positions)
        try:
            save(ix._raw, args.out)
            print(f"Saved {args.out}: {ix.num_docs} docs, {len(ix)} terms")
            print(f"Positions enabled: {ix.has_positions}")
            print(f"File size: {args.out.stat().st_size:,} bytes")
        finally:
            ix.close()
        return

    # The context manager releases caches and in-memory postings on any exit.
    with open_index(args.index, corpus_root=args.corpus) as ix:
        print(f"Loaded {ix!r}, avg doc length={ix.avg_doc_length:.1f}")
        if args.command == "search":
            if args.repeat < 1:
                parser.error("--repeat must be positive")
            scorer = BM25() if args.scorer == "bm25" else TfIdf()
            results = []
            for repeat in range(args.repeat):
                results = ranked_search(ix, args.query, scorer=scorer, k=args.k)
                print(f"Search run {repeat + 1}: {len(results)} ranked results")
            for rank, item in enumerate(results, start=1):
                print(f"\n{rank:2}. {item}")
                print(f"    {item.snippet}")
            print(f"\nCached query matches: {ix.cache_info()}")
        elif args.command == "sanity":
            from findex.lab03_evaluation import print_sanity

            print_sanity(ix)
        else:
            from findex.lab03_evaluation import evaluate

            evaluate(ix, args.labels, args.out)


if __name__ == "__main__":
    main()
