"""Lab 06 tests: all HTTP requests use MockTransport, no live network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import iter_documents
from findex.crawler import CrawlStats, canonicalize, crawl, fetch
from findex.index import build_index
from findex.store import load


def test_canonical_url() -> None:
    assert canonicalize("HTTP://Example.COM:80/path/?b=2&a=1#frag") == (
        "http://example.com/path?a=1&b=2"
    )


def test_fetch_retries_only_transient_and_honors_retry_after() -> None:
    async def run() -> None:
        codes = iter([429, 503, 200, 404])
        delays: list[float] = []
        calls: list[str] = []

        def respond(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(next(codes), headers={"Retry-After": "2"}, text="body")

        async def sleep(seconds: float) -> None:
            delays.append(seconds)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await fetch(client, "https://example.com/a", sleep=sleep)
            assert result.status == 200
            assert result.size == 4
            assert len(delays) == 2 and all(value >= 2 for value in delays)
            assert (await fetch(client, "https://example.com/b", sleep=sleep)).status == 404
            assert len(calls) == 4

    asyncio.run(run())


def test_fetch_timeout_and_retry_ceiling() -> None:
    async def run() -> None:
        count = 0

        async def respond(request: httpx.Request) -> httpx.Response:
            nonlocal count
            count += 1
            await asyncio.sleep(0.1)
            return httpx.Response(200)

        async def sleep(seconds: float) -> None:
            pass

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await fetch(
                client, "https://example.com/a", timeout=0.001, sleep=sleep
            )
            assert result.error == "timeout"
            assert count == 4

    asyncio.run(run())


def test_crawl_robots_dedup_allowlist_and_termination() -> None:
    async def run() -> None:
        visited: list[str] = []
        state = CrawlStats()

        def respond(request: httpx.Request) -> httpx.Response:
            visited.append(str(request.url))
            path = request.url.path
            if path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nDisallow: /private")
            html = (
                '<title>Python</title><a href="/a/?b=2&a=1">A</a>'
                '<a href="/a?a=1&b=2#fragment">Duplicate</a>'
                '<a href="/private">Blocked</a><a href="https://other.example/">Other</a>'
                if path == "/" else "<title>Asyncio</title><p>Event loop</p>"
            )
            return httpx.Response(200, text=html, headers={"Content-Type": "text/html"})

        async with asyncio.timeout(3):
            pages = [
                page async for page in crawl(
                    ["https://example.com/"], concurrency=5, per_host=2,
                    max_pages=10, delay=0, stats=state, transport=httpx.MockTransport(respond),
                )
            ]
        assert len(pages) == 2
        assert state.pages == 2
        assert visited.count("https://example.com/robots.txt") == 1
        assert visited.count("https://example.com/a?a=1&b=2") == 1
        assert not any("private" in url or "other.example" in url for url in visited)
        assert state.in_flight == 0

    asyncio.run(run())


def test_crawl_page_cap_and_host_limit() -> None:
    async def run() -> None:
        state = CrawlStats()

        async def respond(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.005)
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nDisallow:")
            html = "".join(f'<a href="/{i}">{i}</a>' for i in range(30))
            return httpx.Response(200, text=html, headers={"Content-Type": "text/html"})

        async with asyncio.timeout(3):
            pages = [
                page async for page in crawl(
                    ["https://example.com/"], concurrency=10, per_host=2,
                    max_pages=3, delay=0, stats=state, transport=httpx.MockTransport(respond),
                )
            ]
        assert len(pages) == 3
        assert 1 <= state.peak_in_flight <= 2
        assert state.in_flight == 0

    asyncio.run(run())


def test_redirect_loop_ends_without_duplicate_fetch() -> None:
    async def run() -> None:
        urls: list[str] = []

        def respond(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            urls.append(path)
            if path == "/robots.txt":
                return httpx.Response(404)
            return httpx.Response(302, headers={"Location": "/b" if path == "/" else "/"})

        async with asyncio.timeout(2):
            pages = [
                page async for page in crawl(
                    ["https://example.com"], delay=0, transport=httpx.MockTransport(respond),
                )
            ]
        assert not pages
        assert urls == ["/robots.txt", "/", "/b"]

    asyncio.run(run())


def test_jsonl_stream_index_search(tmp_path: Path) -> None:
    corpus = tmp_path / "crawl.jsonl"
    rows = [
        {"url": "http://example.com/a", "title": "Python", "text": "asyncio event loop"},
        {"url": "http://example.com/b", "title": "Network", "text": "socket requests"},
    ]
    corpus.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    documents = iter_documents(corpus)
    assert next(documents).doc_id == rows[0]["url"]
    assert next(documents).doc_id == rows[1]["url"]
    target = tmp_path / "index.bin"
    runner = CliRunner()
    built = runner.invoke(app, ["index", str(corpus), "--out", str(target)])
    assert built.exit_code == 0, built.output
    assert load(target) == build_index(corpus)
    found = runner.invoke(
        app, ["search", str(target), "asyncio", "--corpus", str(corpus), "--json"]
    )
    assert found.exit_code == 0, found.output
    result = json.loads(found.stdout)
    assert result["path"] == rows[0]["url"]
    assert "asyncio" in result["snippet"]
