# Lab 08 — baseline before optimization

The source, dependencies and lockfile are unchanged from tag `lab-07`.
Work is isolated on branch `lab08-work` until tests and deployment are ready.

Run in PowerShell from the repository root:

```powershell
uv sync --locked --python 3.13
uv run python scripts/lab08_profile.py cprofile
uv run python scripts/lab08_profile.py sampling
uv run python scripts/lab08_profile.py scalene
```

cProfile reports self time and cumulative time for the entire CLI, as well
as project functions. py-spy saves two flame graph SVGs. Scalene JSON
separates Python/native/system CPU time; inspect with
`uv run --with scalene==2.4.0 scalene view --cli docs/lab08-index-scalene.json`
(and the search JSON). A single CLI search includes import, pickle load,
ranking, snippets and rendering. For the later scorer comparison we will
load once and measure only scoring, with repeated warm runs.

Use the complete corpus, positional index, serial build and three-term
query `python OR asyncio OR thread`. Raw .prof and temporary indexes live
under ignored data/. Commit the reports, SVGs, and written hypotheses
before changing the production code.

## Baseline observations, 12.11.2026 (before optimization)

| Function / line | Self / cumulative seconds | Python / native / system CPU | Hypothesis |
|---|---:|---|---|
| Index: _io.open | 4.800 / 4.803 | Scalene pathlib._local.py:537: 0.01% Python / 3.63% native / 0.98% system | File opens may dominate cold CLI build; NumPy will not directly fix I/O. |
| Index: tokenize.py:22 tokenize | 1.005 / 1.730 | Scalene tokenize.py:30: 0.57% Python / 1.79% native / 0.12% system | Millions of generator calls and per-token object operations may cost time; measure before changing tokenization. |
| Search: dataclasses.fields | 0.591 / 0.766 | Scalene dataclasses.py:1372: 0.31% Python / 2.58% native / 0.74% system | Avoid rebuilding Python posting objects in the hot search path; reuse loaded index and compact numeric representation. |

Windows, AMD Ryzen 5 7430U (6 physical / 12 logical cores), Python 3.13.
514 documents, 1,921,193 tokens, 41,255 terms, positional pickle 12,034,472 bytes.
cProfile: index 12.666 s / 11,387,038 calls; search 1.854 s / 2,841,452 calls.
Search query `python OR asyncio OR thread`, top 10. `_pickle.load` cumulative
1.347 s; `ranked_search` cumulative 0.007 s, BM25.score 610 calls and 0.002 s
cumulative. These are instrumented CLI timings, not uninstrumented latency.

The three rows above are actionable candidates selected from the index and
search profiles, not a single global sorted top-three list. Index cProfile
also reports threading.wait self 1.270 s and cumulative 28.849 s; these are
overlapping/nested wait timings, not additive elapsed time. Rich progress
and synchronization can affect the full CLI profile. `_pickle.load` includes
the dataclass work, so these numbers must not be added together.

py-spy index flame graph: 459 samples, 0 errors, 100 Hz sampling. Both index and search flame SVGs were committed in c2215ce.
Scalene elapsed: index 6.749 s, search 1.469 s. Percentages in the
table are sampled attribution at individual lines, not percentages within
a function or additive cProfile elapsed times. Search store.py:68
pickle.load: 0.00% Python / 9.42% native / 1.43% system. The dominant
threading.py:995 call is attributed 12.40% Python / 49.59% native /
12.72% system (index), 13.71% / 54.84% / 15.28% (search); this is a
thread/progress attribution, not a ranking kernel. Scalene has little
resolution for the very short scorer in a single CLI invocation.
A warmed scorer-only benchmark is therefore the right experiment. The later scorer benchmark will
isolate scoring with a loaded index and warm repeated calls, and compare
identical rankings, not promise an end-to-end CLI speedup.

Do not infer the result from another computer, and do not compare
cProfile-instrumented seconds against uninstrumented NumPy timings.
