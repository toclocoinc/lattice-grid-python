# lattice-grid-jupyter

Edit a **pandas DataFrame** as an interactive [Lattice Grid](https://latticegrid.dev)
inside Jupyter, with a live two-way round-trip back to Python.

```python
import pandas as pd
from lattice_grid_jupyter import LatticeGridWidget

df = pd.DataFrame({"name": ["Ada", "Grace"], "score": [91, 88], "active": [True, False]})

w = LatticeGridWidget(df)   # renders an editable grid
w                           # display it in a notebook cell
# ...edit cells in the grid...
w.df                        # the DataFrame, reflecting your edits (dtypes preserved)
```

Built on [anywidget](https://anywidget.dev), so it works in JupyterLab, classic
Notebook, VS Code, Colab, and anywhere ipywidgets render.

---

## Install

```bash
pip install lattice-grid-jupyter
```

Depends on `anywidget` and `pandas`. The wheel **vendors** the published Lattice
Grid bundle (`@toclocoinc/lattice-grid@1.85.0`) so offline notebooks work with no
network at render time.

### Which grid am I getting?

The one the version number says. A wheel is numbered **`<grid version>.<revision>`**
— `1.69.0.0` carries Lattice Grid `1.69.0` — so the answer is on the tin, and in
Python:

```python
import lattice_grid_pandas
lattice_grid_pandas.GRID_VERSION   # '1.69.0' -- read from the vendored bundle
```

All three packages (`lattice-grid-pandas`, `lattice-grid-jupyter`,
`lattice-grid-dash`) share the number and are released together; the two host
wrappers pin the shared one exactly, so a mixed install is refused rather than
quietly assembled. A revision above `0` is a packaging fix for the same grid.

Every grid release rebuilds these wheels automatically, so `pip install
--upgrade lattice-grid-jupyter` gets you the current grid.

## The grid bundle: CDN vs vendored

Two ways the browser gets the grid JS/CSS, chosen per widget:

| Mode | How | When |
|------|-----|------|
| **CDN** (default) | dynamic `import()` from jsDelivr at render time | online notebooks; smallest saved-notebook size |
| **Vendored** (`offline=True`) | the bundle text ships in the widget model | air-gapped / offline notebooks |

```python
w = LatticeGridWidget(df)                # CDN (default)
w = LatticeGridWidget(df, offline=True)  # vendored, no network needed
```

The default is CDN. Vendored mode carries ~3 MB of grid JS in the widget model,
which inflates a saved `.ipynb`; prefer CDN when you have network. (An alternative
not implemented here would serve the vendored asset over Jupyter's own static
handler instead of the comm; see *Notes* below.)

## What the DataFrame maps to

**dtypes -> grid column types**

| pandas dtype | grid column type |
|--------------|------------------|
| `int*`, `Int*` (nullable), `float*` | `number` |
| `bool`, `boolean` (nullable) | `boolean` |
| `datetime64` (naive or tz-aware) | `timestamp` (ISO 8601) |
| `category`, `object`, `string`, `timedelta` | `text` |

**Index**

* A default `RangeIndex(0..n-1)` is not shown as a column (it adds nothing).
* A named / non-default index becomes one **read-only** leading column.
* A `MultiIndex` becomes one read-only column **per level**.
* Row identity is a **stable, positional id** (`__row_id__`), never the index
  label — so **duplicate index labels** and every other index shape work
  correctly and edits always land on the right row.

**Missing values** — `NaN`, `NaT` and `pd.NA` all serialize to JSON `null`.

## Editing round-trip

Every committed cell edit in the grid flows back to `w.df`, cast to the column's
dtype (an int column stays int, etc.). Index columns are read-only.

## Live updates: Python -> grid

```python
w.set_data(new_df)                 # replace the whole frame, repaint
keys = w.append_rows([{...}, ...]) # append rows (DataFrame or list-of-dicts)
w.append_rows(other_df)
w.delete_rows(keys)                # delete by the keys append_rows returned
```

`append_rows` / `delete_rows` push **incremental** changes to the grid
(`grid.rows.apply({add|remove})`); `set_data` does a full `grid.rows.load()`.
Row keys are stable, so keys returned by `append_rows` stay valid for
`delete_rows` and for later edits.

## Selection, the filtered view and state: grid -> Python

The grid is not a one-way display. What the user does in it comes back as
DataFrames and a plain dict (only row **keys** cross the comm; Python slices the
frame it already holds, so dtypes, the original index and missing values are the
frame's own):

```python
w.selected    # the selected rows: a DataFrame with the original index and dtypes
              # (an EMPTY frame, never None, when nothing is selected)
w.view        # the rows passing the grid's filters (quick search included),
              # in the grid's current sort order
w.state       # filter / sort / column / group / pivot state, a plain dict
w.state = {...}   # restore a state (e.g. one you saved earlier)
```

```python
def on_select(change):
    print(len(change["new"]), "rows selected")        # change["new"] is a DataFrame

w.observe(on_select, names="selected")
w.observe(lambda c: print(len(c["new"]), "rows in view"), names="view")
```

`selected` and `view` notify **once per user change**: the front end debounces
(120 ms), so typing in the quick search does not send a message per keystroke,
and an unchanged result never notifies. Before the grid has reported, `view` is
the whole frame. Row selection is on by default (`selection="multiple"`; override
with `options={"selection": "single"}`).

**Tutorial step: select rows, get a DataFrame.**

1. `w = LatticeGridWidget(df); w`: click a row, Ctrl-click another.
2. In the next cell: `w.selected` shows exactly those rows, with their original
   index labels, so `df.loc[w.selected.index]` addresses the same rows.
3. Type in the grid's search box, then `w.view` is the matching rows in the order
   you see them. `w.view.to_csv("shown.csv")` exports what is on screen.

## Grid options from Python

```python
w = LatticeGridWidget(
    df,
    options={"row_height": 32, "stripe": True},                 # grid options
    columns=[{"field": "score", "title": "Score", "width": 120}],  # per column
    histograms=True,           # header histograms
    profile=True,              # the Statistics tool panel
    pivot="region",            # pivot on a field (or pivot={...} for the option)
)
w.options = {"row_height": 44}   # updates the LIVE grid; the data is not re-sent
```

* Options and column entries are **plain dicts**. `snake_case` keys are accepted
  and mapped to the grid's `camelCase` (`row_height` -> `rowHeight`), but **only
  when the camelCase form is a real grid property name**: ids you chose (a column
  called `sales_total`, the keys of `context`) are never rewritten.
* **Options are data.** Functions (`compute`, `row_class`, hooks, renderers)
  cannot reach the browser: passing one raises `TypeError` rather than being
  dropped. Use the declarative forms (format/type strings, formulas, presets).
* `rows`, `rowKey` and the column set are built from the DataFrame; the keys are
  ignored with a warning. Per-column options go in `columns=[...]`, matched on
  `field`; they survive `set_data`.
* **Unknown or mistyped options are not silent.** The grid validates in the
  browser and its own warning ids come back as Python warnings (class
  `LatticeGridWarning`), e.g. `[lattice] config.unknown:row_hieght: 'row_hieght'
  is not a configuration key this grid recognises` or `config.value:rowHeight`
  for a wrong type. Per-column typos give `column.unknown:<key>`.
* `widget.options = {...}` replaces the options on the live grid by sending the
  options alone (keys you drop are reset). Per-column options and the flags
  apply at construction.

**Recipe: profile a DataFrame.**

```python
w = LatticeGridWidget(df, profile=True, histograms=True)
w
```

Header histograms show each column's distribution and clicking a bar filters on
it (then `w.view` is the filtered frame). Open the **Statistics** tab on the right
edge, click a cell, and the panel shows the present/missing/distinct counts, the
twelve numeric figures and a histogram (or, for text, the top values), all
following the filters.

## Charts

```python
from ipywidgets import HBox
from lattice_grid_jupyter import LatticeGridWidget, LatticeChart

grid = LatticeGridWidget(df)
roc = LatticeChart(grid, type="roc", label="churned", score="score")
HBox([grid, roc])        # filter the grid: the chart redraws in the browser
```

`LatticeChart(grid_or_df, type="bar", **spec)` is a chart **viewer of the grid's
dataset**, bound through the grid's own binding: it follows the grid's filters, quick
search and selection live in the browser, with no round trip to Python. A DataFrame
can be charted directly (`LatticeChart(df, type="scatter", x="a", y="b")`).

| | |
|---|---|
| `type` | any grid chart type, plus the opt-in `roc`, `fan`, `decomposition`, `splom`, `hexbin`, `ridgeline` (proven here: `bar`, `line`, `pie`, `scatter` and those six) |
| `**spec` | the chart spec as plain data: `x`, `y`, `series`, `title`, `scheme`, `measures`, ...; `snake_case` is mapped; an unknown key warns by name (`config.unknown:<key>`) and is dropped; functions raise `TypeError` |
| `height`, `offline`, `licence` | size; vendored (offline) or CDN modules; inherited from the grid when omitted |
| `chart.update(type=..., **spec)` | change the type and replace the spec live (one redraw) |
| `chart.set_data(df)` | new data for a DataFrame-backed chart |
| `chart.draws` | how many times the browser has drawn it |

Chart code loads per type (the base chart module plus the one type module), never as
one bundle with every chart. On a **windowed (large) grid** the chart is drawn from
the grouped aggregates Python already computes (`bar`, `line`, `pie`, ... with `x` and
`y`) or refused with a named warning (`chart.windowed.refused:<type>`); it never reads
the rows. One example per type, the large-grid rules and the layout notes are in
[`docs/charts.md`](docs/charts.md).

## The data router

```python
from lattice_grid_jupyter import LatticeGridWidget, LatticeRouter

router = LatticeRouter(
    sources={"customer": {"data": customers, "key": "customer_id",
                          "join": [{"from": "txn", "many": True, "foreign_key": "customer_id",
                                    "aggregate": {"spend": {"fn": "sum", "field": "amount"}}}]},
             "txn": {"data": txns, "key": "txn_id"}},
    routes={"customer": "customer"},
)
w = LatticeGridWidget(router, route="customer")   # the shaped rows
router.update("txn", new_txns)                    # a keyed diff: only the affected parents re-emit
w.view                                            # the shaped rows as a DataFrame
```

`LatticeRouter(sources=, routes=)` declares DataFrames as sources of the grid's own
**data router** with its own join (lookup, collect, rollup onto the parent), spread,
unnest and field-coercion specs as plain dicts (`snake_case` mapped, an unknown key warned
by name, functions refused). The router **runs in the browser on the grid's module**; Python
does not reimplement join semantics. `router.update(source, df)` sends a keyed diff and
`widget.router_stats` counts the rows the route re-emitted. `widget.view` is the shaped
rows (after the grid's filters) as a DataFrame, so a feature table goes straight into a
model. The spec reference and a feature-table recipe on a seeded customers / transactions /
labels fixture are in [`docs/router.md`](docs/router.md).

## Large data

Below **100,000 rows** the widget sends the whole frame to the browser
(column-major, vectorized) and the grid filters, sorts and counts it there.
**At 100,000 rows and above** it switches to a **windowed source**: no rows are
sent up front, and

* the browser asks Python for each **window of rows as the person scrolls**
  (100 rows per request; the grid's loading rows and loading banner show while
  a window is in flight);
* **sort, filter, quick search, group-by subtotals, the distinct values a
  filter menu lists, header histograms and column profiles** are computed in
  Python over the **whole frame**, and only the results cross the comm: a
  window of rows, a list of groups, a set of bucket counts, a profile;
* `widget.selected` still returns the selected rows as a DataFrame, and
  `widget.view` returns a **lazy handle** instead of materialising millions of
  rows (see below).

```python
w = LatticeGridWidget(big_df)                          # 10M rows: windowed automatically
w = LatticeGridWidget(df, large_threshold=1_000_000)   # move the threshold
w = LatticeGridWidget(df, windowed=True)               # always windowed (False: never)
w.windowed                                             # which mode this widget is in

w = LatticeGridWidget("events.parquet")                # a Parquet file, read by DuckDB
w = LatticeGridWidget(polars_df)                       # a Polars frame (no copy into pandas)
```

**`view` above the threshold** is a `LazyView`: nothing is computed until you
ask, and then only what you ask for.

```python
v = w.view          # <LazyView 1,204,331 of 10,000,000 rows (lazy) ...>
len(v)              # how many rows pass the grid's filters
v.head(20)          # the first 20, in the grid's sort order, as a DataFrame
v.slice(1000, 1100) # any range of the view
v.to_pandas()       # the whole view, explicitly (may be large)
v.positions()       # row positions in the source frame
```

`w.observe(fn, names="view")` fires when the grid's filter, quick search or
sort changes, with a fresh `LazyView`.

**Measured for 10,000,000 rows x 20 columns** (JupyterLab 4, headless Chrome,
16-core development machine): first paint about 350 ms; a scroll jump to a
far, never-fetched window about 70 to 80 ms (median); sorting all 10M rows on a
number about 2 s, after which scrolling the sorted view is as fast as unsorted.
Before this, the widget crossed 5 s to first paint between 50,000 and 100,000
rows. Details and every figure: [`docs/large-data.md`](docs/large-data.md).

**What to know:**

* **Inputs.** A pandas DataFrame (editable: a cell edit reaches `w.df`); a
  Parquet path or a Polars DataFrame (read-only), held as one Arrow table
  (a Parquet file is read by DuckDB's parallel reader) and converted to
  pandas one column at a time, only when a query touches that column. These
  need `pip install "lattice-grid-jupyter[large]"` (duckdb, pyarrow).
* **One set of semantics.** The Python engine follows the grid's own kernels:
  text comparisons ignore case unless the filter says `caseSensitive`, missing
  values sort last in both directions, ties keep frame order, a `ne` /
  `notContains` matches a missing value. Tests hold the windowed answers equal
  to the grid's client-side answers on the same frame.
* **Differences from the client path.** Quick search matches each field's
  value text (`1234.5`, an ISO timestamp), not the formatted cell text, and a
  term cannot span two columns. Text sorts by case-folded order, an
  approximation of the browser's collator for non-ASCII text. Date histograms
  bucket in UTC. Pivoting is not pushed: a pivot over a windowed grid is
  refused by the grid by name (`pushdown:pivot-unsupported`).
* **Cost.** The first query that touches a column over 10M rows converts or
  scans it (a sort on a text column is the slowest: seconds); the order is then
  cached, so scrolling through a sorted view only slices it. Each new filter is
  one pass over the frame.
* `append_rows` / `delete_rows` / `set_data` work above the threshold (the grid
  re-sources itself); `set_data` across the threshold switches the mode.

## Licensing

The grid renders **fully and unwatermarked on `localhost`** (the normal
data-science case) with **no key** — that is the free tier. A widget served from
a non-localhost origin (a deployed notebook server, Voila, Binder on a public
host) should pass a key:

```python
w = LatticeGridWidget(df, licence="LG-...")
```

The key is threaded straight into `createGrid({ licence })`; `grid.licence.state()`
resolves to `localhost`, `licensed`, or `trial`.

## Hosts

JupyterLab 4 and Notebook 7 are proven by automated tests; VS Code and Colab need
a manual check. See [`docs/hosts.md`](docs/hosts.md). `tests/test_grid_version.py`
checks that the widget, the vendored bundle, the chart and router modules, the
name tables and the Dash bundle all carry the same grid.

## Development / tests

```bash
pip install -e ".[dev]"
pytest tests/test_serialize.py tests/test_roundtrip.py   # Python round-trip (no browser)
pytest tests/test_smoke_browser.py                        # real-browser smoke test
pytest                                                    # everything, incl. the Lab/Notebook host proofs and the Dash package
```

The smoke test drives a **real Chromium** via Playwright: it renders the widget's
front end, performs a real edit in the grid, and asserts the edit payload reached
the (mocked) model — the one gap the headless Python tests cannot cover. It uses
the vendored bundle, so it needs no network. If no browser is available it is
skipped (never faked).

## Notes / not in this release

* Vendored mode transfers the bundle through the widget comm; serving it over
  Jupyter's static handler would keep saved notebooks small — a future option.
* `set_data` assumes the same schema (columns/index shape). A different schema
  should use a fresh widget.
* Windows travel as columnar JSON. Arrow IPC was measured (card 1618) and is
  faster only for bulk transfers the windowed design never makes; see
  `docs/large-data.md`.

---
Built with [Lattice Grid](https://www.latticegrid.dev), a JavaScript data grid with a Data Router: one live feed keeps grids, charts, boards, Gantt and KPI tiles in step. [Documentation](https://www.latticegrid.dev/docs/) · [Demos](https://www.latticegrid.dev/demos/) · [Licence](https://www.latticegrid.dev/licence/)
