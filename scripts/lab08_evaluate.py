"""Measure P@5 on ten Lab 03 labels and five verified zero-overlap paraphrases."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from findex.numpy_ranking import score_numpy
from findex.object_model import open_index
from findex.ranking import BM25, ranked_search
from findex.semantic import MiniLM, SemanticData, SemanticSearch
from findex.tokenize import tokenize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/lab08/index-before.bin"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/lab08/embeddings"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/lab08/model"))
    parser.add_argument("--out", type=Path, default=Path("benchmarks/lab08-precision.json"))
    args = parser.parse_args()
    originals = json.loads(Path("benchmarks/lab03-labels.json").read_text("utf-8"))["queries"]
    paraphrases = json.loads(Path("benchmarks/lab08-paraphrases.json").read_text("utf-8"))["queries"]
    rows = []
    with open_index(args.index) as index:
        semantic = SemanticSearch(SemanticData.load(args.embeddings, index), MiniLM(args.model_dir))
        paths = {meta.path: doc_id for doc_id, meta in index.doc_meta.items()}
        numeric = index.numpy_postings
        tasks = [(entry["query"], entry["relevant_paths"], "lab03") for entry in originals]
        for entry in paraphrases:
            relevant = set(entry["relevant_paths"])
            targets = {paths[path] for path in relevant}
            for candidate in entry["candidates"]:
                terms = tuple(dict.fromkeys(tokenize(candidate)))
                overlap = [
                    term for term in terms
                    if targets.intersection(map(int, numeric.postings(term)[0]))
                ]
                if not overlap:
                    tasks.append((candidate, sorted(relevant), "paraphrase"))
                    print(f"Zero-overlap verified: {candidate!r} -> {sorted(relevant)}")
                    break
            else:
                raise ValueError(f"No zero-overlap candidate for {entry['topic']}; revise labels")
        for query, relevant, group in tasks:
            terms = tuple(dict.fromkeys(tokenize(query)))
            matched = frozenset(
                int(doc_id) for term in terms for doc_id in numeric.postings(term)[0]
            )
            and_hits = ranked_search(index, query, scorer=BM25(), k=5)
            or_hits = score_numpy(index, matched, terms, BM25(), 5)
            semantic_hits = semantic.search(index, query, mode="semantic", k=5)
            hybrid_hits = semantic.search(index, query, mode="hybrid", k=5)
            rankings = {
                "bm25_and": [hit.path for hit in and_hits],
                "bm25_or": [index.doc_meta[doc_id].path for doc_id, _ in or_hits],
                "semantic": [hit.path for hit in semantic_hits],
                "hybrid": [hit.path for hit in hybrid_hits],
            }
            row = {
                "query": query, "group": group, "relevant_paths": relevant,
                "zero_overlap_verified": group == "paraphrase",
                "p_at_5": {mode: len(set(top) & set(relevant)) / 5 for mode, top in rankings.items()},
                "top5": rankings,
            }
            rows.append(row)
            print(json.dumps(row["p_at_5"]), query, flush=True)
    means = {
        group: {
            mode: statistics.mean(row["p_at_5"][mode] for row in rows
                                  if group == "all" or row["group"] == group)
            for mode in ("bm25_and", "bm25_or", "semantic", "hybrid")
        }
        for group in ("lab03", "paraphrase", "all")
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows, "mean_p_at_5": means}, indent=2) + "\n", "utf-8")
    print("Mean P@5:", json.dumps(means), flush=True)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
