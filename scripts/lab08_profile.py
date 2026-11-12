"""Measure the unchanged Lab 07 CLI before optimizing Lab 08.

Run from the repository root: uv run python scripts/lab08_profile.py cprofile
Then use the sampling and scalene stages. Raw .prof files stay in data/.
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import pstats
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path("data/python_docs")
DATA = Path("data/lab08")
DOCS = Path("docs")
INDEX = DATA / "index-before.bin"
QUERY = "python OR asyncio OR thread"
BUILD = [
    "index",
    str(CORPUS),
    "--out",
    str(INDEX),
    "--positions",
    "--executor",
    "serial",
]
SEARCH = ["search", str(INDEX), QUERY, "--corpus", str(CORPUS), "--k", "10"]


def run(*args: str) -> None:
    print("\n>", subprocess.list2cmdline(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def report(name: str) -> None:
    output = io.StringIO()
    stats = pstats.Stats(str(DATA / f"{name}.prof"), stream=output)
    stats.sort_stats("tottime").print_stats(20)
    stats.sort_stats("cumulative").print_stats(20)
    print("\nProject functions by self time:", file=output)
    stats.sort_stats("tottime").print_stats(r"[/\\\\]src[/\\\\]findex[/\\\\]", 20)
    report_text = output.getvalue()
    (DOCS / f"lab08-{name}.txt").write_text(report_text, encoding="utf-8")
    print(report_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["cprofile", "sampling", "scalene"])
    args = parser.parse_args()
    # All paths and CLI arguments are identical to the original Lab 07 run.
    if Path.cwd().resolve() != ROOT:
        parser.error("Run from the findex repository root")
    if not CORPUS.is_dir():
        parser.error("Missing data/python_docs; run scripts/download_python_docs.py")
    run("git", "diff", "--exit-code", "lab-07", "--", "src", "pyproject.toml", "uv.lock")
    DATA.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    if args.stage == "cprofile":
        for name, command in (("index", BUILD), ("search", SEARCH)):
            run(
                sys.executable,
                "-m",
                "cProfile",
                "-o",
                str(DATA / f"{name}.prof"),
                "-m",
                "findex.cli",
                *command,
            )
            report(name)
        info = {
            "baseline": subprocess.check_output(
                ["git", "rev-parse", "lab-07"], text=True, cwd=ROOT
            ).strip(),
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "query": QUERY,
            "documents": len(list(CORPUS.rglob("*.txt"))),
            "index_bytes": INDEX.stat().st_size,
            "positions": True,
            "executor": "serial",
        }
        (DOCS / "lab08-baseline.json").write_text(
            json.dumps(info, indent=2) + "\n", encoding="utf-8"
        )
    elif not INDEX.is_file():
        parser.error("Run cprofile first to create the baseline index")
    elif args.stage == "sampling":
        for name, command in (("index", BUILD), ("search", SEARCH)):
            run(
                "uv",
                "run",
                "--locked",
                "--with",
                "py-spy",
                "py-spy",
                "record",
                "--subprocesses",
                "--output",
                str(DOCS / f"lab08-{name}-flame.svg"),
                "--",
                sys.executable,
                "-m",
                "findex.cli",
                *command,
            )
    else:
        for name, command in (("index", BUILD), ("search", SEARCH)):
            run(
                "uv",
                "run",
                "--locked",
                "--with",
                "scalene==2.4.0",
                "scalene",
                "run",
                "--cpu-only",
                "--profile-all",
                "--outfile",
                str(DOCS / f"lab08-{name}-scalene.json"),
                "scripts/lab08_profile_target.py",
                "---",
                *command,
            )
    print(f"Finished {args.stage}. Reports are in docs/.", flush=True)


if __name__ == "__main__":
    main()
