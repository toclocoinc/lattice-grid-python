# Charts in the notebook

Card 1619. `LatticeChart` draws a chart **viewer** of a grid's dataset. It is bound
to the grid's own dataset through the grid's own binding, so it follows the grid's
filters, quick search and selection **live in the browser**: filtering the grid
sends nothing to Python and the chart redraws by itself.

```python
from ipywidgets import HBox
from lattice_grid_jupyter import LatticeGridWidget, LatticeChart

grid = LatticeGridWidget(df)
chart = LatticeChart(grid, type="bar", x="segment", y="revenue")
HBox([grid, chart])          # side by side; VBox stacks them
```

* **Display the grid.** The chart borrows the grid the browser has rendered, so the
  grid widget must be displayed (in the same box, or in any cell of the same page).
  Until it is, the chart says so; it then draws without being re-run.
* **A DataFrame directly:** `LatticeChart(df, type="scatter", x="a", y="b")`. The
  frame travels once, as columns, to a hidden grid the chart binds to.
  `chart.set_data(new_df)` redraws. A frame of 100,000 rows or more is refused by
  name: show it in a (windowed) `LatticeGridWidget` and bind the chart to that.
* **The spec is plain data.** Everything after `type` is the chart spec. `snake_case`
  keys map to the grid's camelCase (`empty_text` -> `emptyText`); nested dicts and
  lists pass as they are; functions raise `TypeError`. A key the chart does not
  recognise is **reported by name** as a `LatticeGridWarning`
  (`[lattice] config.unknown:scor: ...`, the grid's own warning id) and left out.
* **Change it live:** `chart.update(type="roc", label="churned", score="p")` (the
  spec you give replaces the old one), or `chart.type = ...` / `chart.spec = {...}`.
* **Loads only what it uses.** The base chart module plus the one module the type
  needs (`roc` -> `chart-roc`): about 440 KB and 5 KB, never one bundle with every
  chart. With `offline=True` on the grid they travel in the widget like the grid
  bundle; otherwise they come from jsDelivr, pinned to the grid version.
* `chart.draws` is how often the browser has drawn the chart, so a notebook can
  see a filter redraw it.

## Large (windowed) grids

A windowed grid keeps its rows in Python and the browser holds one window. A chart
over it never reads rows:

* `bar`, `horizontalBar`, `line`, `step`, `area`, `pie`, `donut`, `funnel` and
  `waterfall` with `x` and `y` are drawn from the **grouped aggregates** Python
  already answers for the grid (the same request the grid's own subtotals use),
  over the whole frame and under the grid's filters. A filter re-aggregates; scrolling
  does not. `y` is summed unless `measures=[{"col": "y", "fn": "avg"}]` says otherwise.
* every other type needs the rows themselves, so it is **refused with a named
  warning** (`chart.windowed.refused:<type>`), the chart shows the reason, and nothing is
  read.

## One example per type

The fixture is seeded and synthetic; every block runs as written.

```python
import numpy as np, pandas as pd
from ipywidgets import HBox, VBox
from lattice_grid_jupyter import LatticeGridWidget, LatticeChart

rng = np.random.default_rng(0)
n = 240
t = np.arange(n)
churned = rng.integers(0, 2, n)
df = pd.DataFrame({
    "day": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
    "segment": rng.choice(["alpha", "beta", "gamma"], n),
    "revenue": rng.normal(100, 20, n),
    "visits": rng.normal(50, 10, n),
    "tenure": rng.normal(24, 6, n),
    "sales": 100 + t * 0.5 + 10 * np.sin(t / 7) + rng.normal(0, 3, n),
    "forecast": 100 + t * 0.5 + 10 * np.sin(t / 7),
    "low": 90 + t * 0.5,
    "high": 110 + t * 0.5 + t * 0.05,
    "trend": 100 + t * 0.5,
    "season": 10 * np.sin(t / 7),
    "resid": rng.normal(0, 3, n),
    "churned": churned,
    "score": np.clip(churned * 0.3 + rng.random(n) * 0.7, 0, 1),
})
grid = LatticeGridWidget(df, offline=True)
```

**Bar** (a sum per category; the grid's filter changes it):

```python
bar = LatticeChart(grid, type="bar", x="segment", y="revenue", title="Revenue by segment")
HBox([grid, bar])
```

**Line:**

```python
line = LatticeChart(grid, type="line", x="day", y="sales")
```

**Pie:**

```python
pie = LatticeChart(grid, type="pie", x="segment", y="revenue")
```

**Scatter** (`selection=True` dims every point except the rows selected in the grid):

```python
scatter = LatticeChart(grid, type="scatter", x="visits", y="revenue", selection=True)
```

**ROC** (also `curve="pr"` for precision-recall and `curve="calibration"`; `label` is the
outcome column, `positive=` names the positive value):

```python
roc = LatticeChart(grid, type="roc", label="churned", score="score")
```

**Fan** (history, forecast and an interval that widens):

```python
fan = LatticeChart(grid, type="fan", x="day", y="sales", forecast="forecast", lower="low", upper="high")
```

**Decomposition** (observed, trend, seasonal and residual panels on one x axis):

```python
decomposition = LatticeChart(grid, type="decomposition", x="day", observed="sales",
                             trend="trend", seasonal="season", residual="resid")
```

**SPLOM** (every pair of 2 to 6 numeric columns):

```python
splom = LatticeChart(grid, type="splom", columns=["revenue", "visits", "tenure"])
```

**Hexbin** (a scatter that reads as density at scale; `radius=` is the hexagon size in pixels):

```python
hexbin = LatticeChart(grid, type="hexbin", x="visits", y="revenue", radius=14)
```

**Ridgeline** (one density ridge per category):

```python
ridgeline = LatticeChart(grid, type="ridgeline", x="segment", y="revenue")
```

**Layout.** Charts and grids are ordinary widgets:

```python
VBox([grid, HBox([roc, fan])])
```
