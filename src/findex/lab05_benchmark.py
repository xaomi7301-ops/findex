"""Lab 05 reproducible concurrent indexing benchmarks and speedup chart.

Run at least three repetitions per (executor, workers) on the same full corpus.
CPU time for process pools is parent CPU plus worker compute CPU measured in
build_partial; worker startup and result-pickling CPU may be undercounted.
RSS is sampled for the PARENT process only, not summed across child processes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import psutil

from findex.parallel import ExecutorKind, build_parallel_index, document_paths


@dataclass(frozen=True, slots=True)
class Sample:
    """Raw run, including both the parallel step and serial merge."""

    variant: str
    executor: ExecutorKind
    workers: int
    repetition: int
    wall_s: float
    cpu_s: float
    peak_parent_rss_mib: float
    merge_s: float
    documents: int
    tokens: int


@dataclass(frozen=True, slots=True)
class Summary:
    """Median values over repetitions, same variant + executor + workers."""

    variant: str
    executor: ExecutorKind
    workers: int
    repetitions: int
    wall_s: float
    cpu_s: float
    peak_parent_rss_mib: float
    merge_s: float
    speedup: float


def gil_enabled() -> bool | None:
    """Read the actual runtime GIL flag; None on builds without this API."""
    check = cast("Callable[[], bool] | None", getattr(sys, "_is_gil_enabled", None))
    return check() if check is not None else None


def cpu_model() -> str:
    """CPU brand from Windows registry or Linux cpuinfo, with safe fallback."""
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value: object
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if isinstance(value, str):
                    return value.strip()
        except OSError:
            pass
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[-1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


class ParentRssSampler:
    """Sample only the parent process RSS (including during merge)."""

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes = 0

    def _measure(self) -> None:
        try:
            self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)
        except (OSError, psutil.Error):
            pass

    def _sample(self) -> None:
        while not self._stop.wait(0.02):
            self._measure()

    def __enter__(self) -> ParentRssSampler:
        self._stop.clear()
        self._measure()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._measure()


def measure(
    corpus: Path, *, executor: ExecutorKind, workers: int, repetition: int, variant: str
) -> Sample:
    """Benchmark the build (without saving to disk) under equal conditions."""
    with ParentRssSampler() as sampler:
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        built = build_parallel_index(corpus, executor=executor, workers=workers)
        parent_cpu = time.process_time() - started_cpu
        wall = time.perf_counter() - started_wall
    # process_time() includes worker threads but not separate subprocesses.
    cpu = parent_cpu + built.worker_cpu_seconds if executor == "processes" else parent_cpu
    return Sample(
        variant=variant,
        executor=executor,
        workers=workers,
        repetition=repetition,
        wall_s=round(wall, 6),
        cpu_s=round(cpu, 6),
        peak_parent_rss_mib=round(sampler.peak_bytes / 1048576, 3),
        merge_s=round(built.merge_seconds, 6),
        documents=built.index.document_count,
        tokens=built.index.token_count,
    )


def summarise(samples: Sequence[Sample]) -> list[Summary]:
    """Compute independent medians and speedups relative to same-runtime serial."""
    grouped: dict[tuple[str, ExecutorKind, int], list[Sample]] = {}
    for sample in samples:
        grouped.setdefault((sample.variant, sample.executor, sample.workers), []).append(
            sample
        )
    baselines: dict[str, float] = {
        variant: statistics.median(s.wall_s for s in records)
        for (variant, executor, workers), records in grouped.items()
        if executor == "serial" and workers == 1
    }
    output: list[Summary] = []
    for (variant, executor, workers), records in sorted(grouped.items()):
        wall = statistics.median(sample.wall_s for sample in records)
        baseline = baselines.get(variant)
        output.append(
            Summary(
                variant=variant,
                executor=executor,
                workers=workers,
                repetitions=len(records),
                wall_s=round(wall, 4),
                cpu_s=round(statistics.median(s.cpu_s for s in records), 4),
                peak_parent_rss_mib=round(
                    statistics.median(s.peak_parent_rss_mib for s in records), 3
                ),
                merge_s=round(statistics.median(s.merge_s for s in records), 4),
                speedup=round(baseline / wall, 4) if baseline and wall else 0.0,
            )
        )
    return output


def _parse_workers(value: str) -> list[int]:
    counts: list[int] = []
    for component in value.split(","):
        component = component.strip().lower()
        number = (os.cpu_count() or 1) if component == "all" else int(component)
        if number < 1:
            raise ValueError("Worker counts must be positive")
        if number not in counts:
            counts.append(number)
    return counts


def _parse_executors(value: str) -> list[ExecutorKind]:
    names: list[ExecutorKind] = []
    for name in value.split(","):
        clean = name.strip().lower()
        if clean not in {"serial", "threads", "processes"}:
            raise ValueError(f"Unsupported executor: {clean!r}")
        name_typed = cast("ExecutorKind", clean)
        if name_typed not in names:
            names.append(name_typed)
    return names


def benchmark(
    corpus: Path,
    *,
    repeats: int,
    workers: list[int],
    executors: list[ExecutorKind],
    warmup: int,
    out: Path,
    append: bool,
) -> dict[str, object]:
    if repeats < 3:
        raise ValueError("Each configuration requires at least 3 repetitions")
    corpus = corpus.resolve()
    paths = document_paths(corpus)
    state = gil_enabled()
    variant = (
        "GIL-off"
        if state is False
        else "GIL-on" if state is True else "GIL-status-unavailable"
    )
    machine = {
        "cpu_model": cpu_model(),
        "logical_cores": os.cpu_count(),
        "python": sys.version.split()[0],
        "python_abi": sys.abiflags,
        "sys._is_gil_enabled()": state,
        "platform": platform.platform(),
    }
    print(f"CPU: {machine['cpu_model']} ({machine['logical_cores']} logical cores)")
    print(f"Python: {machine['python']}; sys._is_gil_enabled(): {state}")
    print(f"Corpus paths: {len(paths)}; repetitions: {repeats}")

    samples: list[Sample] = []
    if append and out.exists():
        old = cast("dict[str, object]", json.loads(out.read_text(encoding="utf-8")))
        old_corpus = old.get("corpus")
        if old_corpus != str(corpus):
            raise ValueError("Cannot append: different corpus path")
        old_samples = old.get("samples")
        if isinstance(old_samples, list):
            for item in cast("list[dict[str, object]]", old_samples):
                samples.append(
                    Sample(
                        variant=str(item["variant"]),
                        executor=cast("ExecutorKind", item["executor"]),
                        workers=cast("int", item["workers"]),
                        repetition=cast("int", item["repetition"]),
                        wall_s=cast("float", item["wall_s"]),
                        cpu_s=cast("float", item["cpu_s"]),
                        peak_parent_rss_mib=cast("float", item["peak_parent_rss_mib"]),
                        merge_s=cast("float", item["merge_s"]),
                        documents=cast("int", item["documents"]),
                        tokens=cast("int", item["tokens"]),
                    )
                )

    for executor in executors:
        configurations = [1] if executor == "serial" else workers
        print(f"\n{executor.upper()}:", flush=True)
        for _ in range(warmup):
            measure(corpus, executor=executor, workers=1, repetition=-1, variant=variant)
        for worker_count in configurations:
            # A rerun replaces the same variant/configuration's former measurements.
            samples = [
                sample
                for sample in samples
                if not (
                    sample.variant == variant
                    and sample.executor == executor
                    and sample.workers == worker_count
                )
            ]
            for repetition in range(1, repeats + 1):
                sample = measure(
                    corpus,
                    executor=executor,
                    workers=worker_count,
                    repetition=repetition,
                    variant=variant,
                )
                samples.append(sample)
                print(
                    f"  workers={worker_count:>2} trial={repetition}/{repeats} "
                    f"wall={sample.wall_s:.3f}s merge={sample.merge_s:.3f}s "
                    f"RSS={sample.peak_parent_rss_mib:.1f} MiB",
                    flush=True,
                )

    summaries = summarise(samples)
    report: dict[str, object] = {
        "corpus": str(corpus),
        "documents_discovered": len(paths),
        "machines": {variant: machine},
        "samples": [asdict(sample) for sample in samples],
        "summary": [asdict(row) for row in summaries],
        "methodology": (
            ">=3 runs per configuration; independent medians; wall includes merge "
            "and IPC, excludes saving; CPU = parent process time plus child compute "
            "time for process pools (excludes child startup/result-pickling CPU); "
            "parent RSS only (not child RSS); stdout stats never invented"
        ),
    }
    if append and out.exists():
        old = cast("dict[str, object]", json.loads(out.read_text(encoding="utf-8")))
        old_machines = old.get("machines")
        if isinstance(old_machines, dict):
            report["machines"] = {**old_machines, variant: machine}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nVariant   Executor    Workers   Wall s   CPU s   Parent RSS MiB   Merge s   Speedup")
    for row in summaries:
        print(
            f"{row.variant:<9} {row.executor:<11} {row.workers:>3}   "
            f"{row.wall_s:>7.3f} {row.cpu_s:>7.3f} "
            f"{row.peak_parent_rss_mib:>12.2f}   {row.merge_s:>7.3f}  "
            f"{row.speedup:>5.2f}x"
        )
    print(f"\nSaved: {out}")
    return report


def plot(report_path: Path, image_path: Path) -> None:
    """Produce a committed PNG with ideal linear speedup for comparison."""
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    data = cast("dict[str, object]", json.loads(report_path.read_text(encoding="utf-8")))
    rows = cast("list[dict[str, object]]", data["summary"])
    groups: dict[tuple[str, str], list[tuple[int, float]]] = {}
    maximum = 1
    for row in rows:
        speedup = float(cast("float", row["speedup"]))
        if speedup <= 0:
            continue  # No matching serial baseline for this Python runtime.
        count = int(cast("int", row["workers"]))
        maximum = max(maximum, count)
        key = (str(row["variant"]), str(row["executor"]))
        groups.setdefault(key, []).append((count, speedup))

    figure = Figure(figsize=(9, 5.5))
    axes: Axes = figure.add_subplot(1, 1, 1)
    axes.plot([1, maximum], [1, maximum], "--", label="Ideal linear")
    for (variant, executor), values in sorted(groups.items()):
        points = sorted(values)
        axes.plot(
            [x for x, _ in points],
            [y for _, y in points],
            marker="o",
            label=f"{executor} ({variant})",
        )
    axes.set_xlabel("Workers")
    axes.set_ylabel("Speedup vs serial baseline (x)")
    axes.set_title("findex Lab 05 — Indexing speedup")
    axes.set_xticks(sorted({1, maximum, *(x for g in groups.values() for x, _ in g)}))
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(image_path, dpi=160)
    print(f"Saved chart: {image_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 05 concurrent indexing experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("corpus", type=Path)
    run.add_argument("--workers", default="1,2,4,8,all")
    run.add_argument("--executors", default="serial,threads,processes")
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--out", type=Path, default=Path("benchmarks/lab05-results.json"))
    run.add_argument("--append", action="store_true")
    chart = sub.add_parser("plot")
    chart.add_argument("report", type=Path)
    chart.add_argument("--out", type=Path, default=Path("benchmarks/lab05-speedup.png"))
    args = parser.parse_args()
    if args.command == "plot":
        plot(args.report, args.out)
    else:
        if args.warmup < 0:
            parser.error("--warmup must be nonnegative")
        benchmark(
            args.corpus,
            repeats=args.repeats,
            workers=_parse_workers(args.workers),
            executors=_parse_executors(args.executors),
            warmup=args.warmup,
            out=args.out,
            append=args.append,
        )


if __name__ == "__main__":
    # Mandatory under Windows multiprocessing.spawn.
    main()
