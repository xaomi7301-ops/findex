"""Streaming JSONL writer, Rich counters and repeatable crawl measurements."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import aclosing
from dataclasses import asdict, dataclass
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from queue import SimpleQueue
from typing import TextIO

from rich.console import Console
from rich.live import Live
from rich.table import Table

from findex.crawler import CrawlStats, crawl
from findex.lab05_benchmark import ParentRssSampler


@dataclass(frozen=True, slots=True)
class CrawlMeasurement:
    concurrency: int
    pages: int
    wall_s: float
    pages_per_s: float
    errors: int
    peak_rss_mib: float
    retries: int
    peak_in_flight: int


def dashboard(state: CrawlStats, start: float) -> Table:
    table = Table(title="findex crawl")
    table.add_column("Pages")
    table.add_column("Pages/s")
    table.add_column("In flight")
    table.add_column("Errors")
    table.add_column("Queue")
    table.add_row(
        str(state.pages),
        f"{state.pages / max(time.perf_counter() - start, 0.001):.2f}",
        str(state.in_flight),
        str(state.errors),
        str(state.queue_depth),
    )
    return table


def write_line(stream: TextIO, line: str) -> None:
    stream.write(line + "\n")
    stream.flush()


async def collect(
    seed: str,
    *,
    stream: TextIO,
    max_pages: int,
    concurrency: int,
    per_host: int,
    delay: float,
    state: CrawlStats,
    start: float,
    live: Live,
) -> None:
    stop = asyncio.Event()

    async def refresh() -> None:
        while not stop.is_set():
            # Terminal rendering can perform blocking console I/O on Windows.
            await asyncio.to_thread(live.update, dashboard(state, start), refresh=True)
            try:
                async with asyncio.timeout(0.15):
                    await stop.wait()
            except TimeoutError:
                pass

    ticker = asyncio.create_task(refresh())
    try:
        async with aclosing(
            crawl(
                [seed],
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                delay=delay,
                stats=state,
            )
        ) as pages:
            async for page in pages:
                line = json.dumps(asdict(page), ensure_ascii=False)
                # File I/O is not performed on the event loop.
                await asyncio.to_thread(write_line, stream, line)
    finally:
        stop.set()
        await ticker
        await asyncio.to_thread(live.update, dashboard(state, start), refresh=True)


def run_crawl(
    seed: str,
    *,
    out: Path,
    log: Path,
    max_pages: int,
    concurrency: int,
    per_host: int,
    delay: float,
    debug: bool = False,
) -> CrawlMeasurement:
    """Run the event loop once, record parent RSS and per-attempt log."""
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    state = CrawlStats()
    crawl_logger = logging.getLogger("findex.crawler")
    loop_logger = logging.getLogger("asyncio")
    old = [(item, item.level, item.propagate) for item in (crawl_logger, loop_logger)]
    records: SimpleQueue[logging.LogRecord] = SimpleQueue()
    handler = QueueHandler(records)
    file_handler = logging.FileHandler(log, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    listener = QueueListener(records, file_handler)
    crawl_logger.addHandler(handler)
    loop_logger.addHandler(handler)
    crawl_logger.setLevel(logging.INFO)
    loop_logger.setLevel(logging.WARNING)
    crawl_logger.propagate = False
    loop_logger.propagate = False
    listener.start()
    try:
        with out.open("w", encoding="utf-8") as stream, ParentRssSampler() as rss:
            start = time.perf_counter()
            with Live(
                dashboard(state, start),
                console=Console(stderr=True),
                auto_refresh=False,
                transient=False,
            ) as live:
                asyncio.run(
                    collect(
                        seed,
                        stream=stream,
                        max_pages=max_pages,
                        concurrency=concurrency,
                        per_host=per_host,
                        delay=delay,
                        state=state,
                        start=start,
                        live=live,
                    ),
                    debug=debug,
                )
            wall = time.perf_counter() - start
        return CrawlMeasurement(
            concurrency,
            state.pages,
            round(wall, 4),
            round(state.pages / max(wall, 0.001), 4),
            state.errors,
            round(rss.peak_bytes / 1048576, 3),
            state.retries,
            state.peak_in_flight,
        )
    finally:
        listener.stop()
        for item, level, propagate in old:
            item.removeHandler(handler)
            item.setLevel(level)
            item.propagate = propagate
        file_handler.close()
