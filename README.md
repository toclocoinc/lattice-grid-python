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
Grid bundle (`@toclocoinc/lattice-grid@1.40.0`) so offline notebooks work with no
network at render time.

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

## Large DataFrames

The frame is serialized **column-major** (`{field: [values...]}`), not as a
list-of-dicts:

* field names are not repeated per row (roughly halves the payload for wide frames);
* each column is built with a **vectorized** pass, not `DataFrame.iterrows()` —
  which is the real reason a naive list-of-dicts build stalls at 100k rows;
* the grid's memory source **virtualizes rendering**, so only the visible window
  is ever in the DOM.

This handles a **100,000-row** frame without freezing (see `tests/` and the demo
notebook). For frames far larger than fit in the browser (millions of rows), the
grid also exposes `paged` / `remote` / `stream` source modes whose `fetch`
callback would page back into the Python kernel over the comm — that is the
documented extension point, not wired in this release.

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

## Development / tests

```bash
pip install -e ".[dev]"
pytest tests/test_serialize.py tests/test_roundtrip.py   # Python round-trip (no browser)
pytest tests/test_smoke_browser.py                        # real-browser smoke test
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
* Binary (Arrow/typed-array) transfer would beat columnar-JSON for very wide
  numeric frames; columnar-JSON is what ships here.
