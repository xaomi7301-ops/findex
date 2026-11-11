"""Smoke-test the Lab 05 benchmark and actual PNG generation on a tiny corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from findex.lab05_benchmark import (
    _parse_executors,
    _parse_workers,
    benchmark,
    plot,
    summarise,
)


def test_parse_worker_counts_and_executor_names() -> None:
    assert _parse_workers("1,2,2") == [1, 2]
    assert _parse_executors("serial,threads,threads") == ["serial", "threads"]
    with pytest.raises(ValueError):
        _parse_workers("0")
    with pytest.raises(ValueError):
        _parse_executors("unknown")


def test_summary_needs_matching_serial_baseline() -> None:
    assert summarise([]) == []


@pytest.mark.slow
def test_benchmark_writes_real_samples_and_chart(tiny_corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "lab05-results.json"
    image = tmp_path / "lab05-speedup.png"
    report = benchmark(
        tiny_corpus,
        repeats=3,
        workers=[1, 2],
        executors=["serial", "threads"],
        warmup=0,
        out=output,
        append=False,
    )
    rows = report["summary"]
    assert isinstance(rows, list)
    assert len(rows) == 3
    assert all(row["repetitions"] == 3 for row in rows)
    assert all(row["peak_parent_rss_mib"] > 0 for row in rows)
    assert all(row["documents"] == 4 for row in report["samples"])
    assert output.is_file()
    assert len(json.loads(output.read_text(encoding="utf-8"))["samples"]) == 9

    plot(output, image)
    assert image.is_file()
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    appended = benchmark(
        tiny_corpus,
        repeats=3,
        workers=[1, 2],
        executors=["threads"],
        warmup=0,
        out=output,
        append=True,
    )
    assert len(appended["samples"]) == 9
    assert all(row["speedup"] > 0 for row in appended["summary"])
