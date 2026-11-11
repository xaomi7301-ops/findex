"""Lab 05: deterministic map/reduce indexing using threads or spawned processes.

Workers receive only file paths and globally assigned IDs, never loaded documents.
Each worker owns its Counter and postings; the parent alone merges the results.
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from findex.index import DocMeta, InvertedIndex, Posting, PostingList
from findex.tokenize import tokenize

logger = logging.getLogger(__name__)
type ExecutorKind = Literal["serial", "threads", "processes"]


@dataclass(frozen=True, slots=True)
class DocumentPath:
    """One path and its reserved corpus-wide numeric ID."""

    doc_id: int
    path: Path
    relative: str


@dataclass(frozen=True, slots=True)
class PartialTask:
    """Picklable task sent to a worker (no document bodies)."""

    paths: tuple[DocumentPath, ...]
    with_positions: bool = False


@dataclass(slots=True)
class PartialIndex:
    """Worker-local records; never mutated concurrently by another worker."""

    postings: dict[str, list[Posting]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    with_positions: bool
    cpu_seconds: float = 0.0


@dataclass(slots=True)
class ParallelBuild:
    """Merged index plus separately measured reduce and worker CPU time."""

    index: InvertedIndex
    merge_seconds: float
    worker_cpu_seconds: float
    chunks: int


def document_paths(root: Path, *, limit: int | None = None) -> list[DocumentPath]:
    """Enumerate only paths, in the same rglob order as Lab 04's iter_documents."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    root = Path(root).resolve()
    if root.is_dir():
        files = (path for path in root.rglob("*.txt") if path.is_file())
    elif root.is_file() and root.suffix.lower() == ".txt":
        files = iter((root,))
    elif not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")
    else:
        raise ValueError("Corpus must be a directory or a .txt file")

    paths: list[DocumentPath] = []
    for path in files:
        if limit is not None and len(paths) >= limit:
            break
        relative = path.relative_to(root).as_posix() if root.is_dir() else path.name
        paths.append(DocumentPath(len(paths), path, relative))
    return paths


def make_tasks(
    paths: list[DocumentPath], *, workers: int, executor: ExecutorKind, positions: bool
) -> list[PartialTask]:
    """Contiguous ID ranges preserve document order; finer tasks balance uneven files."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if executor not in {"serial", "threads", "processes"}:
        raise ValueError("executor must be serial, threads, or processes")
    if not paths:
        return [PartialTask((), positions)]
    if executor == "serial":
        return [PartialTask(tuple(paths), positions)]
    desired_chunks = min(len(paths), workers * 4)
    chunk_size = max(1, (len(paths) + desired_chunks - 1) // desired_chunks)
    return [
        PartialTask(tuple(paths[start : start + chunk_size]), positions)
        for start in range(0, len(paths), chunk_size)
    ]


def build_partial(task: PartialTask) -> PartialIndex:
    """Process paths locally; module-level function is safe with multiprocessing spawn.

    Identical token order, title extraction and postings as Lab 04 build_index().
    Corrupt/unreadable documents are skipped, consistent with iter_documents().
    """
    start_cpu = time.process_time()
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    for doc in task.paths:
        try:
            with doc.path.open("r", encoding="utf-8", errors="strict") as stream:
                content = stream.read()
        except (UnicodeError, OSError) as exc:
            logger.warning("Skipping unreadable document %s: %s", doc.path, exc)
            continue

        counts: Counter[str] = Counter()
        positions: defaultdict[str, list[int]] | None = (
            defaultdict(list) if task.with_positions else None
        )
        length = 0
        for position, term in enumerate(tokenize(content)):
            counts[term] += 1
            if positions is not None:
                positions[term].append(position)
            length += 1

        doc_lengths[doc.doc_id] = length
        title = next((line.strip() for line in content.splitlines() if line.strip()), "")
        doc_meta[doc.doc_id] = DocMeta(
            path=doc.relative, title=title[:100] or Path(doc.relative).stem
        )
        for term, tf in counts.items():
            term_positions = tuple(positions[term]) if positions is not None else ()
            postings[term].append(Posting(doc.doc_id, tf, term_positions))

    return PartialIndex(
        postings=dict(postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        with_positions=task.with_positions,
        cpu_seconds=time.process_time() - start_cpu,
    )


def merge(partials: Iterable[PartialIndex]) -> InvertedIndex:
    """Deterministically merge disjoint global ID ranges into the canonical index.

    If a document was unreadable, renumber the remaining IDs so that the output
    exactly matches Lab 04's enumerate(iter_documents()) behavior.
    """
    ordered = list(partials)
    if not ordered:
        return InvertedIndex({}, {}, {})
    with_positions = ordered[0].with_positions
    for part in ordered:
        if part.with_positions != with_positions:
            raise ValueError("Cannot merge positional and non-positional indices")

    all_ids = [doc_id for part in ordered for doc_id in part.doc_lengths]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate document IDs across partial indices")
    mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(all_ids))}

    lengths: dict[int, int] = {}
    metadata: dict[int, DocMeta] = {}
    for part in ordered:
        for old_id, length in part.doc_lengths.items():
            lengths[mapping[old_id]] = length
            metadata[mapping[old_id]] = part.doc_meta[old_id]
    lengths = dict(sorted(lengths.items()))
    metadata = dict(sorted(metadata.items()))

    # Sort partial chunks by original ID so term dictionary insertion order is
    # the same as in a single-pass build (important for byte-identical pickle).
    ordered.sort(key=lambda part: min(part.doc_lengths, default=float("inf")))
    merged: dict[str, list[Posting]] = {}
    for part in ordered:
        for term, local in part.postings.items():
            target = merged.setdefault(term, [])
            target.extend(
                Posting(mapping[p.doc_id], p.tf, p.positions) for p in local
            )
    for target in merged.values():
        target.sort(key=lambda posting: posting.doc_id)
    # Runtime values are all list[Posting]; the public API allows other posting types.
    canonical = cast("dict[str, PostingList]", merged)
    return InvertedIndex(canonical, lengths, metadata, "slots", with_positions)


def build_parallel_index(
    root: Path,
    *,
    executor: ExecutorKind = "serial",
    workers: int = 1,
    with_positions: bool = False,
    limit: int | None = None,
    on_document: Callable[[int], None] | None = None,
) -> ParallelBuild:
    """Run the same partial worker in serial, threads or spawn-safe processes.

    Executor.map preserves chunk order. Exceptions from a worker propagate
    to the caller with its original exception chained (including remote stack).
    """
    paths = document_paths(root, limit=limit)
    tasks = make_tasks(paths, workers=workers, executor=executor, positions=with_positions)
    partials: list[PartialIndex] = []
    completed = 0

    if executor == "serial":
        for task in tasks:
            partials.append(build_partial(task))
            completed += len(task.paths)
            if on_document is not None:
                on_document(completed)
    elif executor == "threads":
        with ThreadPoolExecutor(max_workers=workers) as pool:
            try:
                for task, partial in zip(tasks, pool.map(build_partial, tasks), strict=True):
                    partials.append(partial)
                    completed += len(task.paths)
                    if on_document is not None:
                        on_document(completed)
            except Exception as exc:
                raise RuntimeError("threads indexing worker failed") from exc
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            try:
                for task, partial in zip(tasks, pool.map(build_partial, tasks), strict=True):
                    partials.append(partial)
                    completed += len(task.paths)
                    if on_document is not None:
                        on_document(completed)
            except Exception as exc:
                raise RuntimeError("processes indexing worker failed") from exc

    started_merge = time.perf_counter()
    result = merge(partials)
    merge_seconds = time.perf_counter() - started_merge
    return ParallelBuild(
        index=result,
        merge_seconds=merge_seconds,
        worker_cpu_seconds=sum(part.cpu_seconds for part in partials),
        chunks=len(tasks),
    )
