"""Measured BM25 sanity checks and manually labeled precision@5 evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

from findex.index import Posting, posting_doc_ids
from findex.object_model import Index
from findex.ranking import BM25, Scorer, TfIdf, ranked_search


class LabeledQuery(TypedDict):
    query: str
    relevant_paths: list[str]


class LabelsFile(TypedDict):
    queries: list[LabeledQuery]


class ScoreRow(TypedDict):
    precision_at_5: float
    relevant_found: int
    top5: list[str]


class EvaluationRow(TypedDict):
    query: str
    relevant_paths: list[str]
    missing_label_paths: list[str]
    tfidf: ScoreRow
    bm25: ScoreRow


def print_sanity(index: Index) -> None:
    """Three controlled BM25 checks using frequencies from a real corpus."""
    bm25 = BM25()
    rare = next(
        (
            term
            for term in ("frozenset", "weakref", "asynciogenerator", "metaclass")
            if 0 < index.df(term) < index.num_docs // 3
        ),
        None,
    )
    if rare is None:
        rare = min(
            (term for term in index if term.isalpha() and len(term) > 4),
            key=index.df,
        )
    common = max(index, key=index.df)
    rare_id = posting_doc_ids(index[rare])[0]
    common_id = posting_doc_ids(index[common])[0]
    rare_weight = bm25.score(rare, Posting(rare_id, 1), index)
    common_weight = bm25.score(common, Posting(common_id, 1), index)
    print("SANITY 1 — rare term vs frequent term, controlled tf=1:")
    print(f"  {rare!r}: df={index.df(rare)}, score={rare_weight:.6f}")
    print(f"  {common!r}: df={index.df(common)}, score={common_weight:.6f}")
    print(f"  Rare term has larger contribution: {rare_weight > common_weight}")

    score1 = bm25.score(rare, Posting(rare_id, 1), index)
    score2 = bm25.score(rare, Posting(rare_id, 2), index)
    score19 = bm25.score(rare, Posting(rare_id, 19), index)
    score20 = bm25.score(rare, Posting(rare_id, 20), index)
    print("SANITY 2 — diminishing returns, fixed doc length and term:")
    print(f"  tf 1→2 delta={score2 - score1:.6f}")
    print(f"  tf 19→20 delta={score20 - score19:.6f}")
    print(f"  20th contributes less: {score20 - score19 < score2 - score1}")

    documents = sorted(index.doc_lengths, key=index.doc_length)
    short_id, long_id = documents[0], documents[-1]
    short_score = bm25.score(common, Posting(short_id, 1), index)
    long_score = bm25.score(common, Posting(long_id, 1), index)
    print("SANITY 3 — document length penalty, controlled tf=1:")
    print(f"  short: {index.doc_length(short_id)} tokens, score={short_score:.6f}")
    print(f"  long:  {index.doc_length(long_id)} tokens, score={long_score:.6f}")
    print(f"  Short document ranks higher: {short_score > long_score}")
    print("Checks use genuine corpus df/length statistics with controlled tf values.")


def _precision(index: Index, query: str, relevant: set[str], scorer: Scorer) -> ScoreRow:
    hits = ranked_search(index, query, scorer=scorer, k=5)
    paths = [hit.path for hit in hits]
    count = sum(path in relevant for path in paths)
    return {
        "precision_at_5": round(count / 5.0, 3),
        "relevant_found": count,
        "top5": paths,
    }


def evaluate(index: Index, label_path: Path, out_path: Path) -> dict[str, object]:
    """Evaluate both scorers against 10 manually labeled query sets."""
    data = cast(LabelsFile, json.loads(Path(label_path).read_text(encoding="utf-8")))
    records = data["queries"]
    if len(records) != 10:
        raise ValueError("Lab 03 expects exactly 10 manually labeled queries")

    existing = {meta.path for meta in index.doc_meta.values()}
    rows: list[EvaluationRow] = []
    for item in records:
        query = item["query"]
        relevant: set[str] = set(item["relevant_paths"])
        missing = sorted(relevant - existing)
        if missing:
            print(f"WARNING: labels absent from this corpus for {query!r}: {missing}")
        rows.append(
            {
                "query": query,
                "relevant_paths": sorted(relevant),
                "missing_label_paths": missing,
                "tfidf": _precision(index, query, relevant, TfIdf()),
                "bm25": _precision(index, query, relevant, BM25()),
            }
        )
    average: dict[str, float] = {
        "tfidf": round(sum(row["tfidf"]["precision_at_5"] for row in rows) / len(rows), 3),
        "bm25": round(sum(row["bm25"]["precision_at_5"] for row in rows) / len(rows), 3),
    }
    output: dict[str, object] = {
        "corpus": str(index.corpus_root),
        "num_docs": index.num_docs,
        "labeled_queries": 10,
        "metric": "P@5 = relevant retrieved within top five / 5",
        "rows": rows,
        "mean_p_at_5": average,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nQUERY                         | TF-IDF P@5 | BM25 P@5")
    print("------------------------------|------------|---------")
    for row in rows:
        print(
            f"{row['query'][:29]:<30}| {row['tfidf']['precision_at_5']:^10.3f} "
            f"| {row['bm25']['precision_at_5']:^8.3f}"
        )
    print(f"Mean P@5: TF-IDF={average['tfidf']:.3f}, BM25={average['bm25']:.3f}")
    print(f"Saved actual measurements: {out_path}")
    return output
