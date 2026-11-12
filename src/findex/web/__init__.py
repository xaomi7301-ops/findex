"""ASGI search service; Pydantic is used only at HTTP/config boundaries."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, cast
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from findex.corpus import iter_documents
from findex.object_model import Index, load_ranked_index
from findex.ranking import BM25, Scorer, TfIdf, ranked_search

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=ROOT / "templates")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    index_path: Path = Path("index-lab07.bin")
    corpus_path: Path = Path("data/python_docs")
    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    workers: Annotated[int, Field(ge=1)] = 1
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    # Only for controlled benchmark: async-direct deliberately blocks the loop.
    search_mode: Literal["sync", "async-direct"] = "sync"


class SearchRequest(BaseModel):
    q: str = Field(min_length=1, max_length=200)
    k: int = Field(10, ge=1, le=100)
    scorer: Literal["bm25", "tfidf"] = "bm25"
    page: int = Field(1, ge=1, le=10000)


class Hit(BaseModel):
    doc_id: int
    title: str
    path: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    total: int
    took_ms: float
    page: int
    k: int
    scorer: Literal["bm25", "tfidf"]
    results: list[Hit]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    path: str
    text: str


class StatsResponse(BaseModel):
    documents: int
    tokens: int
    vocabulary: int
    index_bytes: int
    uptime_s: float


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_index(request: Request) -> Index:
    index = cast(Index | None, getattr(request.app.state, "index", None))
    if index is None or index.closed:
        raise HTTPException(503, "Index not loaded")
    return index


def perform_search(params: SearchRequest, index: Index) -> SearchResponse:
    started = time.perf_counter()
    scorer: Scorer = BM25() if params.scorer == "bm25" else TfIdf()
    try:
        total = len(index.matching_ids(params.q))
        offset = (params.page - 1) * params.k
        hits = (
            ranked_search(index, params.q, scorer=scorer, k=min(total, offset + params.k))
            if offset < total
            else []
        )
    except (ValueError, RecursionError) as exc:
        raise HTTPException(422, "Invalid search query") from exc
    return SearchResponse(
        query=params.q,
        total=total,
        took_ms=round((time.perf_counter() - started) * 1000, 3),
        page=params.page,
        k=params.k,
        scorer=params.scorer,
        results=[
            Hit(
                doc_id=hit.doc_id,
                title=hit.title,
                path=hit.path,
                score=hit.score,
                snippet=hit.snippet,
            )
            for hit in hits[offset : offset + params.k]
        ],
    )


def document(index: Index, doc_id: int) -> DocumentResponse:
    meta = index.doc_meta.get(doc_id)
    if meta is None:
        raise HTTPException(404, "Document not found")
    root = index.corpus_root
    if root is None:
        raise HTTPException(404, "Document content unavailable")
    try:
        if root.suffix.lower() == ".jsonl":
            text = next((doc.text for doc in iter_documents(root) if doc.doc_id == meta.path), None)
        else:
            base = root.resolve()
            source = (base / meta.path).resolve()
            if not source.is_relative_to(base):
                raise HTTPException(404, "Document not found")
            text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise HTTPException(404, "Document content unavailable") from exc
    if text is None:
        raise HTTPException(404, "Document content unavailable")
    return DocumentResponse(doc_id=doc_id, title=meta.title, path=meta.path, text=text)


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
        application.state.index = None
        application.state.started = time.monotonic()
        try:
            index = await asyncio.to_thread(
                load_ranked_index, config.index_path, corpus_root=config.corpus_path
            )
            application.state.index = index
            logger.info("Loaded %d documents from %s", index.num_docs, config.index_path)
        except (OSError, ValueError, EOFError):
            logger.exception("Index unavailable at startup")
        try:
            yield
        finally:
            index = application.state.index
            if index is not None:
                index.close()
            application.state.index = None

    application = FastAPI(title="findex", version="0.7.0", lifespan=lifespan)
    application.state.settings = config
    application.state.index = None
    application.state.started = time.monotonic()
    application.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @application.middleware("http")
    async def request_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid4().hex
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unexpected error request_id=%s", request_id)
            response = JSONResponse({"detail": "Internal server error"}, status_code=500)
        response.headers["X-Request-ID"] = request_id
        return response

    if config.search_mode == "async-direct":

        @application.get("/search", response_model=SearchResponse)
        async def search_async(
            params: Annotated[SearchRequest, Query()],
            index: Annotated[Index, Depends(get_index)],
        ) -> SearchResponse:
            # Benchmark baseline only: deliberately performs blocking work in the loop.
            return perform_search(params, index)

    else:

        @application.get("/search", response_model=SearchResponse)
        def search_sync(
            params: Annotated[SearchRequest, Query()],
            index: Annotated[Index, Depends(get_index)],
        ) -> SearchResponse:
            return perform_search(params, index)

    @application.get("/docs/{doc_id}", response_model=DocumentResponse)
    def get_document(doc_id: int, index: Annotated[Index, Depends(get_index)]) -> DocumentResponse:
        return document(index, doc_id)

    @application.get("/stats", response_model=StatsResponse)
    def stats(
        request: Request,
        index: Annotated[Index, Depends(get_index)],
        cfg: Annotated[Settings, Depends(get_settings)],
    ) -> StatsResponse:
        try:
            size = cfg.index_path.stat().st_size
        except OSError:
            size = 0
        return StatsResponse(
            documents=index.num_docs,
            tokens=index.raw.token_count,
            vocabulary=len(index),
            index_bytes=size,
            uptime_s=round(time.monotonic() - cast(float, request.app.state.started), 2),
        )

    @application.get("/health")
    def health(index: Annotated[Index, Depends(get_index)]) -> dict[str, str]:
        if index.closed:
            raise HTTPException(503, "Index not loaded")
        return {"status": "ok"}

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(
        request: Request,
        index: Annotated[Index, Depends(get_index)],
        q: Annotated[str, Query(max_length=200)] = "",
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        scorer: Literal["bm25", "tfidf"] = "bm25",
        page: Annotated[int, Query(ge=1, le=10000)] = 1,
    ) -> Response:
        results = None
        error = ""
        if q.strip():
            try:
                results = perform_search(SearchRequest(q=q, k=k, scorer=scorer, page=page), index)
            except HTTPException as exc:
                error = str(exc.detail)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"q": q, "results": results, "scorer": scorer, "k": k, "error": error},
        )

    @application.get("/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def view_document(
        request: Request, doc_id: int, index: Annotated[Index, Depends(get_index)]
    ) -> Response:
        return templates.TemplateResponse(
            request=request, name="document.html", context={"doc": document(index, doc_id)}
        )

    return application


app = create_app()
