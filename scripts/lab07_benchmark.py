"""Run repeatable oha benchmarks against local Uvicorn or the user's public service."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

QUERY = "/search?q=python&k=10&scorer=bm25&page=1"


def wait_ready(base: str, process: subprocess.Popen[bytes] | None = None) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError("Uvicorn exited; inspect server log in benchmarks")
        try:
            if httpx.get(base + "/health", timeout=5).status_code == 200:
                response = httpx.get(base + QUERY, timeout=10)
                response.raise_for_status()
                if response.json()["total"] > 0:
                    return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"Service not ready: {base}/health")


def measure(base: str, output: Path, name: str, concurrency: int, count: int) -> dict:
    command = [
        "oha",
        "--no-tui",
        "--output-format",
        "json",
        "-n",
        str(count),
        "-c",
        str(concurrency),
        base + QUERY,
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    payload = json.loads(result.stdout)
    (output / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    status = payload["statusCodeDistribution"]
    if status.get("200", 0) != count or payload["errorDistribution"]:
        raise RuntimeError(f"{name}: non-200 responses or errors: {status}")
    row = {
        "target": name,
        "concurrency": concurrency,
        "requests": count,
        "rps": payload["summary"]["requestsPerSec"],
        **{key + "_ms": payload["latencyPercentiles"][key] * 1000 for key in ("p50", "p95", "p99")},
        "wall_s": payload["summary"]["total"],
        "status": status,
    }
    print(json.dumps(row), flush=True)
    return row


def stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        # Terminate the supervisor and all Uvicorn workers on Windows.
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def local(output: Path, port: int) -> list[dict]:
    base = f"http://127.0.0.1:{port}"
    rows = []
    for mode, workers in (("async-direct", 1), ("sync", 1), ("sync", 4)):
        name = f"local-{mode}-{workers}"
        env = {
            **os.environ,
            "SEARCH_MODE": mode,
            "INDEX_PATH": str(Path("index-lab07.bin").resolve()),
            "CORPUS_PATH": str(Path("data/python_docs").resolve()),
        }
        command = [
            sys.executable,
            "-m",
            "uvicorn",
            "findex.web:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            str(workers),
            "--no-access-log",
        ]
        with (output / f"{name}.log").open("wb") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
            try:
                wait_ready(base, process)
                # Separate M2 experiment: 20 concurrent requests in each route mode.
                if workers == 1:
                    rows.append(measure(base, output, f"{name}-20", 20, 20))
                rows.append(measure(base, output, name, 50, 500))
            finally:
                stop(process)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-url", help="Your own deployed findex URL, after /health works")
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/lab07"))
    args = parser.parse_args()
    if not shutil.which("oha"):
        raise SystemExit("oha not found; install it and restart VS Code.")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.public_url:
        base = args.public_url.rstrip("/")
        if not base.startswith("https://"):
            raise SystemExit("Use the HTTPS URL of your own deployed service")
        wait_ready(base)
        rows = [measure(base, args.out, "deployed", 20, 200)]
        target = args.out / "public-results.json"
    else:
        rows = local(args.out, args.port)
        target = args.out / "local-results.json"
    target.write_text(
        json.dumps({"query": QUERY, "results": rows}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved: {target}")


if __name__ == "__main__":
    main()
