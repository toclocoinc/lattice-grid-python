# Large data: how the windowed source works, and what it costs

Card 1618. At `large_threshold` rows (default 100,000) and above,
`LatticeGridWidget` sends no rows up front. The grid is given a **pushdown
source** whose adapter forwards each query to Python over the widget comm:

| the grid asks for | Python answers with | over the whole frame |
|---|---|---|
| a 100-row window (`execute`) with the filter, quick search and sort | those rows + the matching-row total | yes |
| group-by subtotals (`executeGroupedAggregates`) | one entry per group (+ the grand row) | yes |
| a filter menu's distinct values (a group-by with a count) | value + count pairs | yes |
| whole-set statistics (`executeAggregates`) | one number per statistic | yes |
| header histogram buckets (`facets.provider`) | bucket bounds + counts | yes |
| the Statistics panel's profile | one profile | yes |
| a cell edit (`mutate`, pandas only) | ok | writes `w.df` |

Messages are `{type: "lg:req", id, method, payload}` and
`{type: "lg:res", id, ok, result | error}`; the id pairs each answer with its
request, so answers may arrive in any order, and an abandoned request is
dropped. The grid's own loading rows and loading banner (1.79) show while a
window is in flight.

## Why the widget's limit moved

Before this card the whole frame crossed the comm as columnar JSON. Measured in
JupyterLab 4 (cell executed -> first cell on screen, 20 mixed columns): 10k
0.54 s, 25k 1.27 s, 50k 3.66 s, 100k 12.8 s, 250k 76 s, 1M no paint in 240 s.

## Transport

Windows travel as columnar JSON. Arrow IPC was measured on a 100k-row x 20
window: 7 ms vs 825 ms to encode in Python, 14.7 MB vs 27.4 MB, 186 ms vs 247
ms to decode in the browser. It wins only on bulk transfers, which this design
never makes (a request is 100 rows: about 2 ms of Python), and it would add a
193 KB JavaScript library to the widget.

## Measured: 10,000,000 rows x 20 columns

Development machine (16 cores), JupyterLab 4.6.4, headless Chrome, real kernel
comm. `pytest tests/test_large_10m_lab.py -s` reproduces it; set
`LATTICE_LARGE_REPORT=path.json` to keep every sample.

| | measured | budget |
|---|---|---|
| first paint | 346 to 353 ms | < 2 s |
| scroll window (jump from a quiet viewport) | median 69 to 82 ms, max 89 to 91 ms | < 150 ms |
| scroll window in a sorted view | 60 to 65 ms | |
| comm round trip for one 100-row window | 45 ms | |
| sort all 10M rows on a number, to first paint | 2.0 s | |

Python engine alone (10M x 20): window 2 ms; sort on a number 1.6 s, on text
1.6 s (then cached: scrolling the sorted view only slices); filter on a number
32 ms, a category 51 ms, text `contains` 85 ms; quick search 0.6 s; group-by
0.7 s; filter-menu values 0.5 s; one histogram 0.2 to 0.7 s; one profile 0.4 s.

Jumps fired back to back land inside the grid's own 120 ms scroll-settle
window, which coalesces fetches during a drag: 13 to 225 ms.

## Inputs

* **pandas** -- computed with numpy over the frame (pandas 3 shares the data
  copy-on-write, so the widget costs no copy of a 10M-row frame). Editable.
* **Parquet path** -- read once by DuckDB into an Arrow table.
* **Polars** -- its Arrow data, without a copy.

Parquet and Polars are read-only and convert one column to pandas only when a
query touches it. One implementation of the grid's filter, sort, histogram and
profile semantics serves all three; tests hold the three equal to each other
and the pandas answers equal to the grid's own client-side answers.
