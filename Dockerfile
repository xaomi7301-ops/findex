FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts/download_python_docs.py ./scripts/download_python_docs.py
RUN uv sync --locked --no-dev --no-editable
RUN /app/.venv/bin/python scripts/download_python_docs.py \
    && /app/.venv/bin/findex index data/python_docs --out data/index.bin --positions --executor serial \
    && rm -rf data/_downloads

FROM python:3.13-slim AS runtime
WORKDIR /app
RUN groupadd --system findex && useradd --system --gid findex findex
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/data /app/data
ENV PATH="/app/.venv/bin:$PATH" INDEX_PATH=/app/data/index.bin CORPUS_PATH=/app/data/python_docs HOST=0.0.0.0 PORT=8000 WORKERS=1 SEARCH_MODE=sync
USER findex
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health',timeout=3)"
CMD ["findex", "serve"]
