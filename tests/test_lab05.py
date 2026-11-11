"""Lab 05: deterministic serial/threads/spawn-process map-reduce indexing."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.index import build_index, posting_doc_ids
from findex.parallel import (
    DocumentPath,
    PartialIndex,
    PartialTask,
    build_parallel_index,
    build_partial,
    document_paths,
    make_tasks,
    merge,
)
from findex.store import load

runner = CliRunner()


@pytest.mark.parametrize("positions", [False, True])
def test_serial_single_chunk_identical_to_lab04(tiny_corpus: Path, positions: bool) -> None:
    old = build_index(tiny_corpus, with_positions=positions)
    paths = document_paths(tiny_corpus)
    merged = merge([build_partial(PartialTask(tuple(paths), positions))])
    assert old == merged
    assert pickle.dumps(old) == pickle.dumps(merged)


@pytest.mark.parametrize("positions", [False, True])
@pytest.mark.parametrize("executor", ["serial", "threads"])
def test_worker_modes_have_identical_index(
    tiny_corpus: Path, positions: bool, executor: str
) -> None:
    old = build_index(tiny_corpus, with_positions=positions)
    actual = build_parallel_index(
        tiny_corpus, executor=executor, workers=3, with_positions=positions
    )
    assert actual.index == old
    assert pickle.dumps(actual.index) == pickle.dumps(old)
    assert actual.merge_seconds >= 0


@pytest.mark.slow
@pytest.mark.parametrize("positions", [False, True])
def test_spawn_process_index_identical(tiny_corpus: Path, positions: bool) -> None:
    baseline = build_index(tiny_corpus, with_positions=positions)
    processed = build_parallel_index(
        tiny_corpus, executor="processes", workers=2, with_positions=positions
    )
    assert processed.index == baseline
    assert pickle.dumps(processed.index) == pickle.dumps(baseline)
    assert processed.worker_cpu_seconds >= 0


def test_merged_postings_are_sorted_after_reversed_chunks(tiny_corpus: Path) -> None:
    paths = document_paths(tiny_corpus)
    task_list = make_tasks(paths, workers=4, executor="threads", positions=True)
    partials = [build_partial(task) for task in task_list]
    merged = merge(reversed(partials))
    assert merged == build_index(tiny_corpus, with_positions=True)
    for entries in merged.postings.values():
        assert posting_doc_ids(entries) == sorted(posting_doc_ids(entries))


def test_corrupted_document_is_skipped_and_ids_renumbered(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xf9")
    (tmp_path / "z.txt").write_text("zeta", encoding="utf-8")
    original = build_index(tmp_path)
    for mode in ("serial", "threads"):
        parallel = build_parallel_index(tmp_path, executor=mode, workers=2)
        assert parallel.index == original


def test_index_empty_corpus_and_single_file(tmp_path: Path) -> None:
    assert build_parallel_index(tmp_path).index == build_index(tmp_path)
    file = tmp_path / "x.txt"
    file.write_text("Hello world", encoding="utf-8")
    assert build_parallel_index(file).index == build_index(file)


def test_duplicate_ids_fail_explicitly(tiny_corpus: Path) -> None:
    first = build_partial(PartialTask(tuple(document_paths(tiny_corpus)[:1])))
    with pytest.raises(ValueError, match="Duplicate"):
        merge([first, first])


def test_invalid_workers_executor_and_path(tiny_corpus: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build_parallel_index(tiny_corpus, executor="threads", workers=0)
    with pytest.raises(ValueError, match="executor"):
        make_tasks(document_paths(tiny_corpus), workers=1, executor="bad", positions=False)
    with pytest.raises(FileNotFoundError):
        build_parallel_index(tiny_corpus / "missing")
    with pytest.raises(ValueError, match="nonnegative"):
        document_paths(tiny_corpus, limit=-1)


def test_cli_new_options_and_stored_file(tiny_corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "lab05.bin"
    result = runner.invoke(
        app,
        [
            "index", str(tiny_corpus), "--out", str(output), "--executor", "threads",
            "--workers", "3", "--positions",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Executor=threads, workers=3" in result.stdout
    assert load(output) == build_index(tiny_corpus, with_positions=True)


def test_cli_rejects_zero_workers(tiny_corpus: Path) -> None:
    result = runner.invoke(app, ["index", str(tiny_corpus), "--workers", "0"])
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_partial_receives_paths_not_text(tiny_corpus: Path) -> None:
    selected = document_paths(tiny_corpus)
    assert selected
    assert all(isinstance(item.path, Path) for item in selected)
    assert all(isinstance(item, DocumentPath) for item in selected)
    assert not any(hasattr(item, "text") for item in selected)
    assert all(selected[i].doc_id == i for i in range(len(selected)))


def test_partial_has_no_shared_lists(tiny_corpus: Path) -> None:
    paths = document_paths(tiny_corpus)
    left = build_partial(PartialTask(tuple(paths[:1])))
    right = build_partial(PartialTask(tuple(paths[1:])))
    assert isinstance(left, PartialIndex) and isinstance(right, PartialIndex)
    assert left.postings is not right.postings
    assert set(left.doc_lengths).isdisjoint(right.doc_lengths)
