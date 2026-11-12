"""Polite asynchronous crawler: one client, bounded requests, streamed pages."""
from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger(__name__)
USER_AGENT = "findex-lab06/0.6 (+https://github.com/xaomi7301-ops/findex/issues)"


def canonicalize(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"Not an HTTP URL: {url}")
    if parts.username or parts.password:
        raise ValueError("Credentials in URLs are not allowed")
    scheme = parts.scheme.lower()
    host = parts.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    if port is not None and (scheme, port) not in {("http", 80), ("https", 443)}:
        host = f"{host}:{port}"
    path = parts.path.rstrip("/") or "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, host, path, query, ""))


def origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status: int | None
    body: str
    size: int
    elapsed: float
    content_type: str = ""
    error: str = ""
    location: str = ""


def retry_delay(value: str | None, attempt: int, base: float) -> float:
    wait = base * 2**attempt + random.uniform(0, base)
    if value:
        try:
            return max(wait, float(value), 0.0)
        except ValueError:
            try:
                stamp = parsedate_to_datetime(value)
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=UTC)
                return max(wait, (stamp - datetime.now(UTC)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    return wait


@dataclass(slots=True)
class CrawlStats:
    pages: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0
    retries: int = 0
    peak_in_flight: int = 0


class RequestGate:
    """Global and per-origin limits, with a delay between request starts."""

    def __init__(self, concurrency: int, per_host: int, delay: float, stats: CrawlStats) -> None:
        self.global_limit = asyncio.Semaphore(concurrency)
        self.per_host = per_host
        self.delay = delay
        self.stats = stats
        self.limits: dict[str, asyncio.Semaphore] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.next_start: dict[str, float] = {}
        self.host_delay: dict[str, float] = {}

    async def request(self, client: httpx.AsyncClient, url: str, timeout: float) -> httpx.Response:
        host = origin(url)
        sem = self.limits.setdefault(host, asyncio.Semaphore(self.per_host))
        lock = self.locks.setdefault(host, asyncio.Lock())
        async with sem, self.global_limit:
            async with lock:
                await asyncio.sleep(max(0.0, self.next_start.get(host, 0) - time.monotonic()))
                self.next_start[host] = time.monotonic() + max(
                    self.delay, self.host_delay.get(host, 0)
                )
            self.stats.in_flight += 1
            self.stats.peak_in_flight = max(self.stats.peak_in_flight, self.stats.in_flight)
            try:
                async with asyncio.timeout(timeout):
                    return await client.get(url, follow_redirects=False)
            finally:
                self.stats.in_flight -= 1


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    gate: RequestGate | None = None,
    retries: int = 3,
    base: float = 0.2,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> FetchResult:
    """At most three retries, only 429/5xx/timeouts. Log every attempt."""
    if not 0 <= retries <= 3:
        raise ValueError("retries must be 0..3")
    for attempt in range(retries + 1):
        started = time.perf_counter()
        response: httpx.Response | None = None
        retryable = False
        error = ""
        try:
            if gate is None:
                async with asyncio.timeout(timeout):
                    response = await client.get(url, follow_redirects=False)
            else:
                response = await gate.request(client, url, timeout)
        except (TimeoutError, httpx.TimeoutException):
            error = "timeout"
            retryable = True
        except httpx.RequestError as exc:
            error = type(exc).__name__
        elapsed = time.perf_counter() - started
        status = response.status_code if response is not None else None
        size = len(response.content) if response is not None else 0
        logger.info(
            "url=%s status=%s bytes=%d elapsed=%.3fs attempt=%d error=%s",
            url, status, size, elapsed, attempt + 1, error or "-",
        )
        if status is not None:
            retryable = status == 429 or 500 <= status < 600
        if retryable and attempt < retries:
            wait = retry_delay(
                response.headers.get("Retry-After") if response is not None else None,
                attempt, base,
            )
            logger.warning("retry url=%s status=%s delay=%.3fs", url, status, wait)
            if gate is not None:
                gate.stats.retries += 1
            await sleep(wait)
            continue
        return FetchResult(
            url, status, response.text if response is not None else "", size, elapsed,
            response.headers.get("Content-Type", "") if response is not None else "",
            error,
            response.headers.get("Location", "") if response is not None else "",
        )
    raise AssertionError("unreachable")


class RobotsCache:
    """One robots request per origin, including concurrent first access."""

    def __init__(self, client: httpx.AsyncClient, gate: RequestGate, timeout: float) -> None:
        self.client = client
        self.gate = gate
        self.timeout = timeout
        self.cache: dict[str, RobotFileParser] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> bool:
        host = origin(url)
        async with self.locks.setdefault(host, asyncio.Lock()):
            if host not in self.cache:
                result = await fetch(
                    self.client, host + "/robots.txt", gate=self.gate, timeout=self.timeout
                )
                parser = RobotFileParser()
                parser.set_url(host + "/robots.txt")
                if result.status == 200:
                    parser.parse(result.body.splitlines())
                elif result.status == 404:
                    parser.parse(["User-agent: *", "Disallow:"])
                else:
                    # Fail closed when permission is unknown.
                    parser.parse(["User-agent: *", "Disallow: /"])
                delay = parser.crawl_delay(USER_AGENT)
                if delay is not None:
                    self.gate.host_delay[host] = float(delay)
                self.cache[host] = parser
            return self.cache[host].can_fetch(USER_AGENT, url)


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.parts: list[str] = []
        self.titles: list[str] = []
        self.in_title = False
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag == "title":
            self.in_title = True
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self.links.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        if tag == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.hidden == 0 and data.strip():
            (self.titles if self.in_title else self.parts).append(data.strip())


def extract(html: str) -> tuple[str, str, list[str]]:
    parser = PageParser()
    parser.feed(html)
    parser.close()
    return " ".join(parser.titles), " ".join(parser.parts), parser.links


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 500,
    concurrency: int = 10,
    per_host: int = 2,
    delay: float = 0.1,
    timeout: float = 10.0,
    allowed_domains: Iterable[str] | None = None,
    stats: CrawlStats | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncGenerator[Page]:
    """Yield successful HTML pages; cancel workers at the budget or on close.

    The allowlist is exact origins (scheme, host, port), not suffix matching.
    Redirect destinations go back through the same allowlist, robots and seen checks.
    """
    if min(max_pages, concurrency, per_host) < 1 or delay < 0 or timeout <= 0:
        raise ValueError("Invalid crawl limits")
    seed_urls = [canonicalize(seed) for seed in seeds]
    allowed = (
        {origin(canonicalize(domain)) for domain in allowed_domains}
        if allowed_domains is not None else {origin(seed) for seed in seed_urls}
    )
    state = stats if stats is not None else CrawlStats()
    frontier: asyncio.Queue[str] = asyncio.Queue()
    results: asyncio.Queue[Page | None] = asyncio.Queue(maxsize=concurrency)
    seen: set[str] = set()
    pending = 0

    def enqueue(raw: str) -> None:
        nonlocal pending
        try:
            url = canonicalize(raw)
        except ValueError:
            return
        if origin(url) not in allowed or url in seen:
            return
        # Bound discovered URLs too, even if a site has no successful pages.
        if len(seen) >= max_pages * 10:
            return
        seen.add(url)
        pending += 1
        frontier.put_nowait(url)
        state.queue_depth = frontier.qsize()

    for seed in seed_urls:
        enqueue(seed)
    if pending == 0:
        return
    gate = RequestGate(concurrency, per_host, delay, state)
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        transport=transport,
        limits=httpx.Limits(max_connections=concurrency),
        trust_env=False,
    ) as client:
        robots = RobotsCache(client, gate, timeout)

        async def worker() -> None:
            nonlocal pending
            while True:
                url = await frontier.get()
                state.queue_depth = frontier.qsize()
                try:
                    if not await robots.allowed(url):
                        logger.info("url=%s status=robots-denied bytes=0 elapsed=0", url)
                    else:
                        result = await fetch(client, url, gate=gate, timeout=timeout)
                        if result.status in {301, 302, 303, 307, 308} and result.location:
                            enqueue(urljoin(url, result.location))
                        elif result.status == 200 and "html" in result.content_type.lower():
                            title, text, links = await asyncio.to_thread(extract, result.body)
                            for link in links:
                                enqueue(urljoin(url, link))
                            page = Page(url, title or url, text, datetime.now(UTC).isoformat())
                            await results.put(page)
                        else:
                            state.errors += 1
                finally:
                    frontier.task_done()
                    pending -= 1
                if pending == 0:
                    await results.put(None)

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(worker()) for _ in range(concurrency)]
            try:
                while state.pages < max_pages:
                    page = await results.get()
                    if page is None:
                        break
                    state.pages += 1
                    yield page
            finally:
                for task in tasks:
                    task.cancel()
                # No queue.join after cancellation: queued items are discarded.
                while not frontier.empty():
                    frontier.get_nowait()
                    frontier.task_done()
                state.queue_depth = 0
