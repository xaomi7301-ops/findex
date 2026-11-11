"""Lab 03 measured ranking checks and human-selected precision@5 labels."""

from __future__ import annotations

import json
from pathlib import Path

from findex.index import Posting, posting_doc_ids
from findex.object_model import Index
from findex.ranking import BM25, TfIdf, ranked_search


def print_sanity(index: Index) -> None:
    """Three controlled BM25 checks using statistics of the *real* index.

    Fixed tf=1 (or 19/20) isolates the contribution being tested; corpus
    document frequencies and lengths are always real measured values.
    """
    bm25 = BM25()
    # Prefer a human-readable rare term present in Python documentation.
    rare = next(
        (term for term in ("frozenset", "weakref", "asynciogenerator", "metaclass")
         if 0 < index.df(term) < index.num_docs // 3),
        None,
    )
    if rare is None:
        rare = min((term for term in index if term.isalpha() and len(term) > 4),
                   key=index.df)
    common = max(index, key=index.df)
    rare_id = posting_doc_ids(index[rare])[0]
    common_id = posting_doc_ids(index[common])[0]
    # Held tf constant to compare term rarity; IDs belong to real docs.
    rare_weight = bm25.score(rare, Posting(rare_id, 1), index)
    common_weight = bm25.score(common, Posting(common_id, 1), index)
    print("SANITY 1 — rare term vs frequent term, controlled tf=1:")
    print(f"  {rare!r}: df={index.df(rare)}, score={rare_weight:.6f}")
    print(f"  {common!r}: df={index.df(common)}, score={common_weight:.6f}")
    print(f"  Rare term has larger contribution: {rare_weight > common_weight}")

    sample_id = rare_id
    score1 = bm25.score(rare, Posting(sample_id, 1), index)
    score2 = bm25.score(rare, Posting(sample_id, 2), index)
    score19 = bm25.score(rare, Posting(sample_id, 19), index)
    score20 = bm25.score(rare, Posting(sample_id, 20), index)
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


def evaluate(index: Index, label_path: Path, out_path: Path) -> dict:
    """Evaluate two scorers on ten manually selected query/relevant-path sets.

    Precision@5 = number of relevant documents in first five / 5.
    The labels are deliberately a small, explicit human-selected test set;
    they are not synthetic rankings or inferred from model outputs.
    """
    data = json.loads(Path(label_path).read_text(encoding="utf-8"))
    records = data["queries"]
    if len(records) != 10:
        raise ValueError("Lab 03 expects exactly 10 manually labeled queries")

    existing = {meta.path for meta in index.doc_meta.values()}
    rows: list[dict] = []
    for item in records:
        query = item["query"]
        relevant = set(item["relevant_paths"])
        missing = sorted(relevant - existing)
        if missing:
            print(f"WARNING: labels absent from this corpus for {query!r}: {missing}")
        results = {}
        for name, scorer in (("tfidf", TfIdf()), ("bm25", BM25())):
            hits = ranked_search(index, query, scorer=scorer, k=5)
            paths = [hit.path for hit in hits]
            relevant_retrieved = sum(path in relevant for path in paths)
            results[name] = {
                "precision_at_5": round(relevant_retrieved / 5.0, 3),
                "relevant_found": relevant_retrieved,
                "top5": paths,
            }
        rows.append({
            "query": query,
            "relevant_paths": sorted(relevant),
            "missing_label_paths": missing,
            **results,
        })

    average = {
        key: round(sum(row[key]["precision_at_5"] for row in rows) / len(rows), 3)
        for key in ("tfidf", "bm25")
    }
    output = {
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
