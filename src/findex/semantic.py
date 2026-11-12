"""MiniLM semantic retrieval, document-best chunk scoring and RRF."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from findex.corpus import iter_documents
from findex.numpy_ranking import score_numpy
from findex.object_model import Index
from findex.ranking import BM25, SearchResult
from findex.tokenize import tokenize

logger = logging.getLogger(__name__)
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FILE = "onnx/model_quint8_avx2.onnx"
MODEL_SHA = "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21"
type Vectors = NDArray[np.float32]
type Int64 = NDArray[np.int64]
type SearchMode = Literal["lexical", "semantic", "hybrid"]


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> Vectors: ...


class _Encoding(Protocol):
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


class _Tokenizer(Protocol):
    def enable_truncation(self, max_length: int) -> None: ...
    def enable_padding(self, **kwargs: object) -> None: ...
    def encode_batch(self, inputs: list[str]) -> list[_Encoding]: ...


class _TokenizerFactory(Protocol):
    def from_file(self, path: str) -> _Tokenizer: ...


class _Input(Protocol):
    name: str


class _Session(Protocol):
    def get_inputs(self) -> list[_Input]: ...
    def run(self, outputs: None, feeds: dict[str, Int64]) -> list[object]: ...


class _Options(Protocol):
    intra_op_num_threads: int
    inter_op_num_threads: int


class _ORT(Protocol):
    def SessionOptions(self) -> _Options: ...
    def InferenceSession(
        self, path: str, *, sess_options: _Options, providers: list[str]
    ) -> _Session: ...


def _download(directory: Path, remote: str, name: str, sha: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    if not destination.exists():
        temp = destination.with_suffix(destination.suffix + ".part")
        try:
            url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{remote}"
            with urllib.request.urlopen(url, timeout=120) as source, temp.open("wb") as out:
                while block := source.read(1024 * 1024):
                    out.write(block)
            temp.replace(destination)
        finally:
            temp.unlink(missing_ok=True)
    if sha is not None and hashlib.sha256(destination.read_bytes()).hexdigest() != sha:
        raise ValueError(f"Model checksum mismatch: {destination}; remove and retry")
    return destination


def normalize(vectors: Vectors) -> Vectors:
    """Unit vectors so dot product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.maximum(norms, np.float32(1e-12))).astype(np.float32)


class MiniLM:
    """Pinned quantized ONNX encoder with mean pooling, no PyTorch needed."""

    def __init__(self, model_dir: Path, *, threads: int = 2) -> None:
        model = _download(model_dir, MODEL_FILE, "model.onnx", MODEL_SHA)
        tokenizer = _download(model_dir, "tokenizer.json", "tokenizer.json")
        # The native extension APIs are wrapped by narrow typed protocols.
        ort = cast(_ORT, importlib.import_module("onnxruntime"))
        factory = cast(_TokenizerFactory, importlib.import_module("tokenizers").Tokenizer)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = factory.from_file(str(tokenizer))
        self._tokenizer.enable_truncation(max_length=256)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    def encode(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        enc = self._tokenizer.encode_batch(list(texts))
        feeds: dict[str, Int64] = {
            "input_ids": np.asarray([row.ids for row in enc], dtype=np.int64),
            "attention_mask": np.asarray([row.attention_mask for row in enc], dtype=np.int64),
            "token_type_ids": np.asarray([row.type_ids for row in enc], dtype=np.int64),
        }
        names = {item.name for item in self._session.get_inputs()}
        raw = self._session.run(None, {key: value for key, value in feeds.items() if key in names})
        hidden = np.asarray(cast(NDArray[np.float32], raw[0]), dtype=np.float32)
        if hidden.ndim != 3 or hidden.shape[-1] != 384:
            raise ValueError(f"Unexpected MiniLM output shape: {hidden.shape}")
        mask = feeds["attention_mask"][:, :, None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
        return normalize(pooled.astype(np.float32))


def chunks(text: str, *, size: int = 120, overlap: int = 20) -> Iterator[str]:
    """Overlapping word windows; encoder truncates at 256 wordpieces."""
    if not 0 <= overlap < size:
        raise ValueError("Invalid chunk size/overlap")
    words = text.split()
    if not words:
        yield ""
        return
    for start in range(0, len(words), size - overlap):
        yield " ".join(words[start : start + size])
        if start + size >= len(words):
            break


@dataclass(frozen=True, slots=True)
class SemanticData:
    matrix: Vectors
    doc_ids: NDArray[np.int32]
    paths: tuple[str, ...]
    excerpts: tuple[str, ...]

    def validate(self, index: Index) -> None:
        paths = tuple(index.doc_meta[i].path for i in range(index.num_docs))
        if self.paths != paths:
            raise ValueError("Embeddings belong to a different index/corpus order")
        if self.matrix.shape != (len(self.doc_ids), 384):
            raise ValueError("Invalid embedding matrix shape")
        if len(self.excerpts) != len(self.doc_ids):
            raise ValueError("Invalid embedding excerpts")
        if self.doc_ids.size and (
            int(self.doc_ids.min()) < 0 or int(self.doc_ids.max()) >= index.num_docs
        ):
            raise ValueError("Invalid embedding document ID")

    @classmethod
    def load(cls, directory: Path, index: Index) -> SemanticData:
        data = cast(dict[str, object], json.loads((directory / "meta.json").read_text("utf-8")))
        if data.get("model") != MODEL or data.get("revision") != REVISION:
            raise ValueError("Embedding model/revision mismatch")
        result = cls(
            np.load(directory / "embeddings.npy", mmap_mode="r", allow_pickle=False),
            np.load(directory / "doc_ids.npy", allow_pickle=False),
            tuple(cast(list[str], data["paths"])),
            tuple(cast(list[str], data["excerpts"])),
        )
        if result.matrix.dtype != np.float32 or result.doc_ids.dtype != np.int32:
            raise ValueError("Invalid embedding dtypes")
        result.validate(index)
        return result


def build_embeddings(
    index: Index, corpus: Path, directory: Path, encoder: Encoder, *, batch_size: int = 16
) -> SemanticData:
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    paths = tuple(index.doc_meta[i].path for i in range(index.num_docs))
    by_path = {path: doc_id for doc_id, path in enumerate(paths)}
    all_ids: list[int] = []
    excerpts: list[str] = []
    vectors: list[Vectors] = []
    pending: list[str] = []
    seen: set[str] = set()

    def flush() -> None:
        if pending:
            vectors.append(encoder.encode(pending))
            pending.clear()

    for doc in iter_documents(corpus):
        if doc.doc_id not in by_path:
            continue
        seen.add(doc.doc_id)
        for chunk in chunks(doc.text):
            all_ids.append(by_path[doc.doc_id])
            excerpts.append(chunk[:240])
            pending.append(chunk)
            if len(pending) == batch_size:
                flush()
        if len(seen) % 50 == 0:
            logger.info("Embedded %d/%d documents", len(seen), len(paths))
    flush()
    if seen != set(paths):
        raise ValueError("Corpus does not contain every indexed document")
    matrix = np.concatenate(vectors, axis=0) if vectors else np.empty((0, 384), np.float32)
    result = SemanticData(
        normalize(matrix), np.asarray(all_ids, dtype=np.int32), paths, tuple(excerpts)
    )
    result.validate(index)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "embeddings.npy", result.matrix, allow_pickle=False)
    np.save(directory / "doc_ids.npy", result.doc_ids, allow_pickle=False)
    (directory / "meta.json").write_text(
        json.dumps(
            {"model": MODEL, "revision": REVISION, "paths": paths, "excerpts": excerpts},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return result


def rrf(rankings: Sequence[Sequence[int]], *, constant: int = 60) -> list[tuple[int, float]]:
    """Combine ranks, not incomparable BM25/cosine raw scores."""
    if constant < 1:
        raise ValueError("RRF constant must be positive")
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class SemanticSearch:
    def __init__(self, data: SemanticData, encoder: Encoder) -> None:
        self.data = data
        self.encoder = encoder

    def search(
        self, index: Index, query: str, *, mode: Literal["semantic", "hybrid"], k: int
    ) -> list[SearchResult]:
        if not query.strip() or k < 0:
            raise ValueError("Expected nonempty query and nonnegative k")
        if not k or not self.data.doc_ids.size:
            return []
        vector = self.encoder.encode([query])[0]
        values = self.data.matrix @ vector
        best: dict[int, tuple[float, int]] = {}
        for row, (doc_id, value) in enumerate(zip(self.data.doc_ids, values, strict=True)):
            key, score = int(doc_id), float(value)
            if key not in best or score > best[key][0]:
                best[key] = (score, row)
        semantic = sorted(best, key=lambda doc_id: (-best[doc_id][0], doc_id))
        if mode == "hybrid":
            terms = tuple(dict.fromkeys(tokenize(query)))
            numeric = index.numpy_postings
            matched = frozenset(
                int(doc_id) for term in terms for doc_id in numeric.postings(term)[0]
            )
            lexical = score_numpy(index, matched, terms, BM25(), index.num_docs)
            ranked = rrf([[doc_id for doc_id, _ in lexical], semantic])
        else:
            ranked = [(doc_id, best[doc_id][0]) for doc_id in semantic]
        return [
            SearchResult(
                doc_id=doc_id,
                score=score,
                title=index.doc_meta[doc_id].title,
                path=index.doc_meta[doc_id].path,
                snippet=self.data.excerpts[best[doc_id][1]],
            )
            for doc_id, score in ranked[:k]
        ]
