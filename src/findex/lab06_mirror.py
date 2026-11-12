"""Authorized local HTTP mirror and reproducible network benchmark for Lab 06.

Serve excerpts of the Python documentation already downloaded in Lab 01.
The server binds only to loopback, injects latency and one HTTP 429 per run.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlsplit

from findex.crawl_runner import CrawlMeasurement, run_crawl


class MirrorServer(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True

    def __init__(self, port: int, pages: list[str], latency: float) -> None:
        super().__init__(("127.0.0.1", port), MirrorHandler)
        self.pages = pages
        self.latency = latency
        self.retried = False
        self.lock = Lock()


class MirrorHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def send_body(
        self,
        body: str,
        status: int = 200,
        content_type: str = "text/html",
        retry_after: str | None = None,
    ) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        if retry_after is not None:
            self.send_header("Retry-After", retry_after)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        server = self.server
        assert isinstance(server, MirrorServer)
        path = urlsplit(self.path).path
        if path == "/robots.txt":
            self.send_body("User-agent: *\nDisallow: /private\n", content_type="text/plain")
            return
        # This blocking delay is in a separate HTTP server thread, never the crawler loop.
        time.sleep(server.latency)
        if path == "/":
            links = "".join(
                f'<a href="/page/{i}">Page {i}</a> ' for i in range(1, len(server.pages))
            )
            self.send_body(server.pages[0] + links + '<a href="/private">Private</a>')
            return
        if path.startswith("/page/"):
            try:
                number = int(path.rsplit("/", 1)[-1])
            except ValueError:
                number = -1
            if 1 <= number < len(server.pages):
                if number == 1:
                    with server.lock:
                        if not server.retried:
                            server.retried = True
                            self.send_body("Retry later", 429, retry_after="1")
                            return
                self.send_body(server.pages[number] + '<a href="/#again">Home</a>')
                return
        self.send_body("Not found", 404)


def make_pages(root: Path, count: int = 200) -> list[str]:
    """Use real local documentation excerpts, retaining attribution in the HTML."""
    paths = sorted(root.rglob("*.txt"))
    if len(paths) < count:
        raise ValueError(f"Need {count} .txt documents in {root}; found {len(paths)}")
    pages: list[str] = []
    for path in paths[:count]:
        with path.open(encoding="utf-8") as stream:
            body = stream.read(4000)
        title = next((line.strip() for line in body.splitlines() if line.strip()), path.stem)
        relative = path.relative_to(root).as_posix()
        pages.append(
            f"<html><head><title>{escape(title)}</title></head><body>"
            f"<h1>{escape(title)}</h1><p>Python 3.13 documentation — {escape(relative)}</p>"
            f"<pre>{escape(body)}</pre></body></html>"
        )
    return pages


@contextmanager
def mirror(pages: list[str], port: int, latency: float) -> Generator[MirrorServer]:
    server = MirrorServer(port, pages, latency)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def benchmark(root: Path, port: int, latency: float, output: Path) -> None:
    pages = make_pages(root)
    output.mkdir(parents=True, exist_ok=True)
    seed = f"http://127.0.0.1:{port}/"
    rows: list[CrawlMeasurement] = []
    evidence: list[str] = []
    for concurrency in (1, 5, 20):
        log = output / f"crawl-{concurrency}.log"
        with mirror(pages, port, latency):
            result = run_crawl(
                seed,
                out=output / f"crawl-{concurrency}.jsonl",
                log=log,
                max_pages=200,
                concurrency=concurrency,
                per_host=10,
                delay=0.001,
                debug=True,
            )
        if result.pages != 200:
            raise RuntimeError(f"Expected 200 pages, got {result.pages}")
        rows.append(result)
        lines = log.read_text(encoding="utf-8").splitlines()
        evidence.extend(line for line in lines if "status=429" in line or "retry url=" in line)
        evidence.extend(line for line in lines if "page/1 status=200" in line)
        slow = [line for line in lines if "Executing " in line and " took " in line]
        print(f"concurrency={concurrency}: {json.dumps(asdict(result))}")
        print(f"asyncio slow callback warnings: {len(slow)}")
        if slow:
            evidence.extend(slow)
    report = {
        "seed": seed,
        "max_pages": 200,
        "per_host": 10,
        "delay_s": 0.001,
        "server_latency_s": latency,
        "debug": True,
        "results": [asdict(row) for row in rows],
        "failure_and_debug_log": evidence,
    }
    target = output / "lab06-results.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nConcurrency | Pages | Wall s | Pages/s | Errors | Peak RSS MiB")
    for row in rows:
        print(
            f"{row.concurrency} | {row.pages} | {row.wall_s} | "
            f"{row.pages_per_s} | {row.errors} | {row.peak_rss_mib}"
        )
    print(f"Saved: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["serve", "benchmark"])
    parser.add_argument("--corpus", type=Path, default=Path("data/python_docs"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--latency", type=float, default=0.08)
    parser.add_argument("--out", type=Path, default=Path("data/lab06"))
    args = parser.parse_args()
    if args.mode == "benchmark":
        benchmark(args.corpus, args.port, args.latency, args.out)
    else:
        server = MirrorServer(args.port, make_pages(args.corpus), args.latency)
        print(f"Local authorized mirror: http://127.0.0.1:{args.port}/ (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
