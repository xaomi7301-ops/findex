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

Actual Windows run: 514 documents, 18,301 chunks, 384 float32 dimensions.
The matrix alone contains 28,110,336 bytes (about 26.81 MiB); metadata,
model and lexical index add memory. All three CLI modes ran successfully.
After adding API tests and correcting Pyright narrowing in lifespan:
207 tests passed in 6.99 s, Ruff passed, Pyright strict reported 0 errors.

## Measured precision@5

| Query group | BM25 AND | BM25 OR | Semantic | Hybrid RRF |
|---|---:|---:|---:|---:|
| Lab 03 (10) | 0.22 | 0.22 | 0.24 | 0.24 |
| Zero-overlap paraphrases (5) | 0.00 | 0.00 | 0.04 | 0.04 |
| All (15) | 0.1467 | 0.1467 | 0.1733 | 0.1733 |

The evaluator verifies zero token overlap with the labeled target before
retrieval. Only `tallying multiplicities` reached its target in the top five
for semantic/hybrid among the five paraphrases; the other four failed.
The improvement on this small label set is modest (13 versus 11 labeled
hits across 75 top-five slots). RRF ties semantic mean P@5, so no claim
that hybrid is universally superior is warranted. The labels are sparse;
relevant but unlabeled documents do not count. The zero-overlap condition
is deliberately hard and produces unnatural phrases.
For `how to run several tasks at the same time`, Coroutines and tasks was
rank 4 by semantic, rank 1 by hybrid and absent from lexical top five.
Raw per-query rankings: benchmarks/lab08-precision.json.

## Deployment preparation

Docker regenerates the official documentation, positional index and
MiniLM vectors in its builder stage. The runtime reads all assets from
/app/data; it remains non-root, with one web worker and two encoder threads.
The model and embeddings are not committed to Git. The mutable Python
documentation archive can change between builds; verify corpus counts.
The Windows server has started successfully; browser/API smoke tests,
Linux image build, memory usage and the public v1.0.0 rollout still need
actual verification. Lab 07 public p95 is historical, not a v1.0.0 metric.
