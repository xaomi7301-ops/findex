"""Lab 08: compact int32 postings and vectorized scoring.

The positional Lab 02 index stays readable for Boolean/phrase queries and
serialization. A read-only numeric view is created once per loaded Index.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from findex.object_model import Index
    from findex.ranking import BM25, TfIdf

type IntArray = NDArray[np.int32]
type FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class NumpyPostings:
    """Two contiguous posting buffers, term offsets, and one length array."""

    doc_ids: IntArray
    tfs: IntArray
    doc_lengths: IntArray
    offsets: dict[str, tuple[int, int]]

    @classmethod
    def from_index(cls, index: Index) -> NumpyPostings:
        n = index.num_docs
        lengths = np.empty(n, dtype=np.int32)
        max_int = int(np.iinfo(np.int32).max)
        for doc_id, length in index.doc_lengths.items():
            if not 0 <= doc_id < n or not 0 <= length <= max_int:
                raise ValueError("NumPy index requires contiguous IDs and int32 lengths")
            lengths[doc_id] = length
        total = sum(index.df(term) for term in index)
        ids = np.empty(total, dtype=np.int32)
        tfs = np.empty(total, dtype=np.int32)
        offsets: dict[str, tuple[int, int]] = {}
        start = 0
        for term, postings in index.items():
            count = index.df(term)
            end = start + count
            if isinstance(postings, array):
                if postings.itemsize != 4:
                    raise ValueError("Expected 32-bit packed postings")
                packed = np.frombuffer(postings, dtype=np.uint32)
                if packed.size and int(packed.max()) > max_int:
                    raise OverflowError("Posting does not fit int32")
                ids[start:end] = packed[::2]
                tfs[start:end] = packed[1::2]
            else:
                ids[start:end] = np.fromiter(
                    (p.doc_id for p in postings), dtype=np.int32, count=count
                )
                tfs[start:end] = np.fromiter((p.tf for p in postings), dtype=np.int32, count=count)
            offsets[term] = (start, end)
            start = end
        ids.flags.writeable = False
        tfs.flags.writeable = False
        lengths.flags.writeable = False
        return cls(ids, tfs, lengths, offsets)

    @property
    def nbytes(self) -> int:
        """Bytes in numeric buffers, excluding dictionary and positional index."""
        return self.doc_ids.nbytes + self.tfs.nbytes + self.doc_lengths.nbytes

    def postings(self, term: str) -> tuple[IntArray, IntArray]:
        start, end = self.offsets.get(term, (0, 0))
        return self.doc_ids[start:end], self.tfs[start:end]


def top_k(scores: FloatArray, candidates: IntArray, k: int) -> list[tuple[int, float]]:
    """Partition, include cutoff ties, then sort selected IDs deterministically."""
    if k <= 0 or candidates.size == 0:
        return []
    values = scores[candidates]
    if k < candidates.size:
        split = candidates.size - k
        selected = np.argpartition(values, split)[split:]
        cutoff = float(values[selected].min())
        # argpartition is unstable: include all ties before deterministic sorting.
        candidates = candidates[values >= cutoff]
    ranked = sorted(
        ((int(doc_id), float(scores[doc_id])) for doc_id in candidates),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return ranked[:k]


def score_numpy(
    index: Index,
    matched: frozenset[int],
    terms: tuple[str, ...],
    scorer: BM25 | TfIdf,
    k: int,
) -> list[tuple[int, float]]:
    """One NumPy expression per query term; no Python loop over its postings."""
    from findex.ranking import BM25

    if k <= 0 or not matched:
        return []
    numeric = index.numpy_postings
    scores = np.zeros(index.num_docs, dtype=np.float64)
    for term in terms:
        ids, tfs = numeric.postings(term)
        df = ids.size
        if not df:
            continue
        tf = tfs.astype(np.float64)
        if isinstance(scorer, BM25):
            idf = math.log1p((index.num_docs - df + 0.5) / (df + 0.5))
            lengths = numeric.doc_lengths[ids]
            normalization = 1.0 - scorer.b + scorer.b * lengths / (index.avg_doc_length or 1.0)
            contribution = idf * (tf * (scorer.k1 + 1)) / (tf + scorer.k1 * normalization)
        else:
            idf = math.log((index.num_docs + 1) / (df + 1)) + 1.0
            contribution = (1.0 + np.log(tf)) * idf
        scores[ids] += contribution
    candidates = np.fromiter(matched, dtype=np.int32, count=len(matched))
    return top_k(scores, candidates, k)
