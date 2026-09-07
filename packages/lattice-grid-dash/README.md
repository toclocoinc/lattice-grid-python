# lattice-grid-dash

A [Dash](https://dash.plotly.com/) component that renders an editable
[Lattice Grid](https://latticegrid.dev) over a pandas `DataFrame`, with cell edits
and selection surfaced back to Python `@callback`s.

Phase **C2** of the Lattice Grid Python integration (card BACKLOG-0000972,
release 1.41). Built on the same shared serialization layer as the Jupyter widget
(phase C1): [`lattice-grid-pandas`](../lattice-grid-pandas).

## How it works

The component does **not** re-implement a React binding for the grid. It consumes
the grid's **shipped React adapter** (`@toclocoinc/lattice-grid/modules/react`,
grid v1.40.0), which exposes `createLatticeGrid({ React, createGrid })`. We feed
it Dash's own React (aliased at bundle time to the global the dash-renderer
serves) so there is a single React instance on the page. This is the lowest-risk
path identified in the spike: the grid team owns and versions the adapter.

```
DataFrame ──dataframe_to_data()──▶ data prop (columnar, from lattice-grid-pandas)
                                        │
                                   LatticeGrid (React adapter → createGrid)
                                        │  manual edit
              cellChanged prop ◀──setProps──  onCellChanged (adapter callback)
                    │
             Python @callback(Input("grid", "cellChanged"))
```

## Install

```bash
pip install lattice-grid-dash
```

(Editable/dev: `pip install -e packages/lattice-grid-dash`. The built JS bundle is
committed under `lattice_grid_dash/`; rebuild it with `npm install && npm run build`.)

## Usage

```python
import pandas as pd
from dash import Dash, Input, Output, callback, html
import lattice_grid_dash
from lattice_grid_dash import dataframe_to_data, apply_cell_edit

df = pd.DataFrame({"name": ["Ada", "Grace"], "score": [91, 88], "active": [True, False]})

app = Dash(__name__)
app.layout = html.Div([
    lattice_grid_dash.LatticeGrid(id="grid", data=dataframe_to_data(df)),
    html.Pre(id="out"),
])

@callback(Output("out", "children"), Input("grid", "cellChanged"))
def on_edit(edit):
    if not edit:
        return "no edits yet"
    apply_cell_edit(df, edit)          # keep the server-side DataFrame in sync, typed
    return f"{edit['colId']} @ row {edit['key']} = {edit['value']}"

if __name__ == "__main__":
    app.run(debug=True)
```

See [`examples/app.py`](examples/app.py).

## Props

| Prop | Direction | Description |
| --- | --- | --- |
| `data` | in | DataFrame-derived columnar payload from `dataframe_to_data(df)`: `{columns, rowKey, columnar}`. Rows are reconstructed and virtualized in the browser. |
| `columns` | in | Optional explicit column defs (overrides `data.columns`). |
| `options` | in | Passthrough to `createGrid` (e.g. `{"edit": True, "rowHeight": 32}`). |
| `licence` | in | Lattice Grid licence key. Empty on localhost → free/unwatermarked. |
| `cellChanged` | **out** | Last manual edit `{key, colId, value, old, ts}`. Drives Python callbacks. |
| `selectedKeys` | in/out | Selected row keys; set by the grid, settable from Python. |
| `licenceState` | **out** | Grid's resolved licence state (e.g. `"localhost"`). |
| `style`, `className` | in | Host element styling. |

## Licence plumbing

The `licence` prop is a single opaque string handed straight to
`createGrid({ licence })`; the grid resolves it **client-side** (localhost origins
run free, unwatermarked, with no key). The resolved state is surfaced back on the
`licenceState` prop. This is the same contract as the Jupyter widget — the shared
rules live in `lattice_grid_pandas._grid`.

## Grid delivery: vendored (default) vs CDN

The grid JavaScript can reach the browser two ways. **This package implements the
vendored path**; the CDN path is documented here for operators who prefer it.

### Vendored (implemented)

`npm run build` bundles the grid core + React adapter into
`lattice_grid_dash/lattice_grid_dash.min.js` (~2.7 MB) via `esbuild`, and the grid
CSS is vendored as `lattice_grid_dash/lattice-grid.min.css`. Both are registered
as Dash assets (`_js_dist` / `_css_dist`), so the component works **fully offline**
— no network at render time. This mirrors the C1 widget's `offline=True` default
and is what the browser smoke test exercises.

Rebuild:

```bash
cd packages/lattice-grid-dash
npm install
npm run build          # build:js (esbuild) + build:py (dash-generate-components)
```

### CDN (documented alternative)

To ship a thin component bundle that loads the grid from jsDelivr at render time
(smaller wheel, needs network + a relaxed CSP):

1. In `build.mjs`, mark the grid packages external:
   `external: ['react-dom', '@toclocoinc/lattice-grid', '@toclocoinc/lattice-grid/modules/react']`.
2. Before the bundle loads, inject the grid from the CDN, e.g.
   `https://cdn.jsdelivr.net/npm/@toclocoinc/lattice-grid@1.40.0/lattice-grid.esm.min.js`
   and `.../modules/react.esm.min.js` (URLs available from
   `lattice_grid_pandas.cdn_urls()`), and add the stylesheet
   `.../lattice-grid.min.css`.
3. Drop the vendored files from `_js_dist` / `_css_dist`.

The two are mutually exclusive; pick one per build.

## Development / build toolchain

- `npm run build:js` — `esbuild` bundles `src/lib/bundle.js` → `lattice_grid_dash/lattice_grid_dash.min.js`. `react` is aliased to Dash's global React (`src/lib/react-shim.js`); react/react-dom are never bundled.
- `npm run build:py` — `dash-generate-components` (react-docgen v5) reads `src/lib/components/LatticeGrid.react.js` and generates `LatticeGrid.py`, `_imports_.py`, and `metadata.json`.
- `pytest tests/` — unit tests for the data bridge, plus a real-browser Playwright smoke test (system Chrome via `channel='chrome'`).
