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

## Actual hotspots (fill from measured profiles)

| Function / line | Self / cumulative seconds | Python / native / system CPU | Hypothesis |
|---|---:|---|---|
| Pending measurement | — | — | — |

Do not infer the result from another computer, and do not compare
cProfile-instrumented seconds against uninstrumented NumPy timings.
