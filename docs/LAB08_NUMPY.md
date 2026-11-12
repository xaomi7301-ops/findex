# Lab 08 — measured scorer optimization (Windows)

AMD Ryzen 5 7430U, Python 3.13, 514 docs / 1,921,193 tokens / 41,255 terms.
100 repeated warmed calls per configuration, median milliseconds; same
Boolean candidates, same query terms, same top-10 document order and scores
within floating-point tolerance. Load, conversion and snippets excluded.

| Query | Scorer | Python ms | NumPy ms | Speedup |
|---|---|---:|---:|---:|
| the (512 docs) | BM25 | 0.6832 | 0.06295 | 10.85x |
| the | TF-IDF | 0.5273 | 0.0484 | 10.89x |
| frozenset (23 docs) | BM25 | 0.0429 | 0.03585 | 1.20x |
| frozenset | TF-IDF | 0.0339 | 0.0292 | 1.16x |
| python OR asyncio OR thread (441 docs) | BM25 | 0.7734 | 0.08565 | 9.03x |
| python OR asyncio OR thread | TF-IDF | 0.5831 | 0.0671 | 8.69x |

The common-term speedup is 10.89x at best, not an end-to-end CLI or HTTP
speedup. Small posting lists offer little opportunity to amortize NumPy
call/array setup overhead. The positional pickle remains for compatibility;
the read-only numeric view adds memory and a one-time conversion cost.
Local verification: 201 passed, one Starlette deprecation warning, 9.51 s;
Pyright strict 0 errors and Ruff passed. Tiny fixture median search 79 us,
build 846.6 us. Raw results in benchmarks/lab08-scorer.json.

## Semantic stage

Use the pinned Apache-2.0 all-MiniLM-L6-v2 quantized ONNX model (23 MB)
instead of PyTorch. Mean-pool token embeddings with the attention mask,
L2-normalize, store float32 .npy and chunk-to-doc IDs. Semantic score is
the maximum cosine across a document's chunks; hybrid uses RRF k=60 over
BM25 OR lexical terms and semantic ranks. Semantic/hybrid queries are
natural-language text, not strict Boolean filters. Chunk windows contain
120 words with 20 overlap; encoder truncates at 256 wordpieces.
Windows and x86-64 AVX2 Linux are the intended targets. The model is
downloaded once to data/lab08/model, with the ONNX checksum verified.

First run the fake-encoder tests, then build actual embeddings and measure
the 15 labeled queries. Public Render deployment and resource usage are
not yet verified. Model, matrix and excerpts are generated files, not
committed to Git. Do not report semantic precision before measuring.
