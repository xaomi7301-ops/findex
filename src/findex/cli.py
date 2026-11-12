"""Installed command-line interface: findex index | search | stats.

Console diagnostics and progress use stderr, machine-readable JSONL uses stdout.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from array import array
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from findex.index import PostingList, build_index
from findex.object_model import open_index
from findex.parallel import build_parallel_index
from findex.ranking import BM25, Scorer, TfIdf, ranked_search
from findex.store import load, save

app = typer.Typer(help="findex: build and search a local full-text corpus.", no_args_is_help=True)
error_console = Console(stderr=True)
console = Console()
logger = logging.getLogger(__name__)


class ScorerChoice(StrEnum):
    """Validated ranking modes shown in --help."""

    bm25 = "bm25"
    tfidf = "tfidf"


class SearchModeChoice(StrEnum):
    lexical = "lexical"
    semantic = "semantic"
    hybrid = "hybrid"


class ExecutorChoice(StrEnum):
    """Execution strategies for deterministic map/reduce indexing."""

    serial = "serial"
    threads = "threads"
    processes = "processes"


def _fail(exc: Exception) -> NoReturn:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=1)


@app.callback()
def _configure(
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="-v: INFO; -vv: DEBUG.")
    ] = 0,
) -> None:
    """Configure logging once at the outer CLI entry point."""
    level = logging.DEBUG if verbose >= 2 else logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(levelname)s %(message)s",
        force=True,
    )


@app.command("index")
def index_command(
    corpus: Annotated[Path, typer.Argument(help="Directory of .txt files or a crawl .jsonl file.")],
    out: Annotated[Path, typer.Option("--out", help="Pickle or .json index.")] = Path("index.bin"),
    positions: Annotated[bool, typer.Option("--positions")] = False,
    limit: Annotated[int | None, typer.Option("--limit", min=0)] = None,
    workers: Annotated[int | None, typer.Option("--workers", min=1)] = None,
    executor: Annotated[ExecutorChoice, typer.Option("--executor")] = ExecutorChoice.processes,
) -> None:
    """Build and save with serial, threads or spawned processes and Rich progress."""
    try:
        # Ryzen 5 7430U measurements: processes x8 was fastest (1.30x).
        # Respect explicit --workers and keep serial restricted to one worker.
        if workers is not None:
            selected_workers = workers
        elif executor is ExecutorChoice.serial:
            selected_workers = 1
        else:
            selected_workers = min(8, os.cpu_count() or 1)
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed} documents"),
            console=error_console,
            transient=True,
        ) as progress:
            task = progress.add_task("Indexing corpus", total=limit)

            def report_count(count: int) -> None:
                progress.update(task, completed=count)

            if corpus.suffix.lower() == ".jsonl" and corpus.is_file():
                # Streaming JSONL is indexed synchronously, outside the crawler event loop.
                raw = build_index(
                    corpus, with_positions=positions, limit=limit, on_document=report_count
                )
                save(raw, out)
                typer.echo(f"Indexed {raw.document_count:,} documents; saved {out}.")
                return
            built = build_parallel_index(
                corpus,
                executor=executor.value,
                workers=selected_workers,
                with_positions=positions,
                limit=limit,
                on_document=report_count,
            )
        save(built.index, out)
        typer.echo(
            f"Indexed {built.index.document_count:,} documents, "
            f"{built.index.token_count:,} tokens, {built.index.vocabulary_size:,} terms; "
            f"saved {out} ({out.stat().st_size:,} bytes). "
            f"Executor={executor.value}, workers={selected_workers}, "
            f"merge={built.merge_seconds:.3f}s."
        )
    except (OSError, ValueError, EOFError) as exc:
        _fail(exc)


@app.command("crawl")
def crawl_command(
    seed: Annotated[str, typer.Argument(help="Allowed HTTP(S) seed URL.")],
    max_pages: Annotated[int, typer.Option("--max-pages", min=1)] = 500,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1)] = 10,
    per_host: Annotated[int, typer.Option("--per-host", min=1)] = 2,
    delay: Annotated[float, typer.Option("--delay", min=0)] = 0.1,
    out: Annotated[Path, typer.Option("--out")] = Path("data/crawl.jsonl"),
    log: Annotated[Path, typer.Option("--log")] = Path("crawl.log"),
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """Crawl an authorized site, streaming JSONL and live counters."""
    from findex.crawl_runner import run_crawl

    try:
        result = run_crawl(
            seed,
            out=out,
            log=log,
            max_pages=max_pages,
            concurrency=concurrency,
            per_host=per_host,
            delay=delay,
            debug=debug,
        )
        typer.echo(json.dumps(asdict(result), ensure_ascii=False))
    except (OSError, ValueError) as exc:
        _fail(exc)


@app.command("search")
def search_command(
    index: Annotated[Path, typer.Argument(help="Trusted index (.bin or .json).")],
    query: Annotated[str, typer.Argument(help="Boolean or phrase query.")],
    k: Annotated[int, typer.Option("--k", min=0)] = 10,
    scorer: Annotated[ScorerChoice, typer.Option("--scorer")] = ScorerChoice.bm25,
    is_json: Annotated[bool, typer.Option("--json", help="JSON Lines on stdout.")] = False,
    corpus: Annotated[Path | None, typer.Option("--corpus")] = None,
    mode: Annotated[SearchModeChoice, typer.Option("--mode")] = SearchModeChoice.lexical,
    embeddings: Annotated[Path, typer.Option("--embeddings")] = Path("data/lab08/embeddings"),
    model_dir: Annotated[Path, typer.Option("--model-dir")] = Path("data/lab08/model"),
) -> None:
    """Search a saved index; Rich table, or JSONL suitable for a pipe."""
    try:
        impl: Scorer = BM25() if scorer is ScorerChoice.bm25 else TfIdf()
        with open_index(index, corpus_root=corpus) as ix:
            if mode is SearchModeChoice.lexical:
                results = ranked_search(ix, query, scorer=impl, k=k)
            else:
                from findex.semantic import MiniLM, SemanticData, SemanticSearch

                semantic = SemanticSearch(SemanticData.load(embeddings, ix), MiniLM(model_dir))
                results = semantic.search(ix, query, mode=mode.value, k=k)
            if is_json:
                for result in results:
                    typer.echo(json.dumps(asdict(result), ensure_ascii=False))
                return

            table = Table(title=f"Results: {query}", show_lines=True)
            table.add_column("#", justify="right")
            table.add_column("Score", justify="right")
            table.add_column("Document")
            table.add_column("Snippet", overflow="fold")
            for position, result in enumerate(results, 1):
                # Highlight only markers generated by our own snippet code.
                highlighted = result.snippet.replace("[[", "[bold yellow]").replace(
                    "]]", "[/bold yellow]"
                )
                table.add_row(
                    str(position),
                    f"{result.score:.4f}",
                    f"{result.title}\n[dim]{result.path}[/dim]",
                    highlighted,
                )
            console.print(table)
            console.print(f"Found {len(results)} ranked result(s).")
    except (OSError, ValueError, EOFError, KeyError) as exc:
        _fail(exc)


def _total_frequency(postings: PostingList) -> int:
    if isinstance(postings, array):
        return sum(int(value) for value in postings[1::2])
    return sum(item.tf for item in postings)


@app.command("embed")
def embed_command(
    index: Annotated[Path, typer.Argument(help="Trusted index file.")],
    corpus: Annotated[Path, typer.Option("--corpus")] = Path("data/python_docs"),
    out: Annotated[Path, typer.Option("--out")] = Path("data/lab08/embeddings"),
    model_dir: Annotated[Path, typer.Option("--model-dir")] = Path("data/lab08/model"),
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 16,
    threads: Annotated[int, typer.Option("--threads", min=1)] = 2,
) -> None:
    """Build normalized MiniLM chunk embeddings for the indexed documents."""
    from findex.semantic import MiniLM, build_embeddings

    try:
        with open_index(index, corpus_root=corpus) as ix:
            result = build_embeddings(
                ix, corpus, out, MiniLM(model_dir, threads=threads), batch_size=batch_size
            )
        typer.echo(f"Saved {result.matrix.shape[0]} chunks, {result.matrix.shape[1]} dimensions: {out}")
    except (OSError, ValueError, EOFError, ImportError) as exc:
        _fail(exc)


@app.command("serve")
def serve_command(
    port: Annotated[int | None, typer.Option("--port", min=1, max=65535)] = None,
    workers: Annotated[int | None, typer.Option("--workers", min=1)] = None,
    host: Annotated[str | None, typer.Option("--host")] = None,
) -> None:
    """Run the FastAPI web search server."""
    import uvicorn

    from findex.web import Settings

    settings = Settings()
    uvicorn.run(
        "findex.web:app",
        host=host or settings.host,
        port=port or settings.port,
        workers=workers or settings.workers,
        log_level=settings.log_level,
    )


@app.command("stats")
def stats_command(
    index: Annotated[Path, typer.Argument(help="Trusted index (.bin or .json).")],
) -> None:
    """Show document/token/vocabulary statistics from a saved index."""
    try:
        raw = load(index)
        summary = Table(title=f"findex statistics: {index.name}")
        summary.add_column("Metric")
        summary.add_column("Value", justify="right")
        for label, value in (
            ("Documents", raw.document_count),
            ("Tokens", raw.token_count),
            ("Vocabulary", raw.vocabulary_size),
        ):
            summary.add_row(label, f"{value:,}")
        console.print(summary)

        popular = sorted(
            ((term, _total_frequency(postings)) for term, postings in raw.postings.items()),
            key=lambda row: (-row[1], row[0]),
        )[:50]
        terms = Table(title="Top 50 terms")
        terms.add_column("Term")
        terms.add_column("Frequency", justify="right")
        for word, count in popular:
            terms.add_row(word, f"{count:,}")
        console.print(terms)
    except (OSError, ValueError, EOFError, KeyError) as exc:
        _fail(exc)


if __name__ == "__main__":
    app()
