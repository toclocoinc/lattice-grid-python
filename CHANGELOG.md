# Changelog

Wrapper versions are `<grid version>.<revision>` (see `RELEASING.md`). Automatic
grid bumps (`Lattice Grid X -> Python wrappers X.0`) change only the vendored grid
and are not listed here; a revision above `.0` is.

## Unreleased

### Added

- A DuckDB relation, or `con=` with `table=` or `sql=`, is a grid source: `LatticeGridWidget(con.table("t"))`,
  `LatticeGridWidget(con=con, sql="SELECT ...")`. Every request the large-data mode answers (row
  window, sort, each filter operator and quick search, grouping with aggregates and group child
  windows, header histograms, the Statistics tab, the row count, `w.view` as a lazy handle) runs as
  SQL on that connection with bound parameters (a value is never spliced into the statement),
  holding only the requested window in memory, and answers as the Arrow path does. A Parquet path
  gains `pushdown=True` to take the same route through `read_parquet` instead of loading the file
  as Arrow. Documented in the README ("DuckDB as the source") and the widget docstring.
- A data router's shaped view is computed in the kernel, from the source frames and the
  router declaration the widget already holds (field coercion, lookup, collect and rollup
  joins with every aggregate, spread, unnest), so `w.view` (and `w.df`, `w.selected`) return the
  shaped DataFrame synchronously under `Run All`, in nbconvert/papermill and with no browser. It
  follows the grid's data router module's semantics, and a parity test runs the same declaration
  through the vendored module under node and through the kernel and compares row for row. A
  text date with no `format` that only the browser's own `Date` parser reads is the one
  declaration still shaped by the browser: `w.view` raises `LatticeViewNeedsBrowser` at once
  until the browser reports (read it in a later cell, or use `w.on_view`). A join cycle in a router declaration no longer
  overflows the stack while ordering the sources.

### Fixed

- Grouping a large frame (the windowed source, above `large_threshold` rows) now behaves as in
  memory. `group_default_expanded` (`False`, `True` or a level number) decides which groups start
  open, instead of every group opening; the group order follows the sort, including a sort on an
  aggregate or on the count; and opening a group asks the kernel for its child window, so the rows
  paint rather than staying on `Loading rows...` skeletons. Collapsing and re-opening a group, and
  grouping two levels deep, work. The kernel answers each group level (and the leaf rows of the
  innermost group) in one request.
- `LatticeChart` now follows the notebook host's theme the way the grid does (it kept a
  white background and light axes under JupyterLab dark): it uses the grid widget's own
  host-theme detection, so a chart is dark under a dark host and light otherwise, follows a
  live host switch, and a chart bound to a grid follows that grid's resolved theme. An
  explicit `theme="dark"` / `"light"` on the chart wins over both.
- `LatticeChart` bound to a grid now follows the grid's selection live by
  default, as the docs always said: `fan` and `decomposition` redraw over the
  selected rows (restoring all rows once the selection clears) and other types
  follow it the grid's own way (emphasis, or zoom for `geomap`). Before this,
  the chart spec never turned selection on, so only an explicit
  `selection=True` worked. `selection=False` turns it off.
- The widget now follows the notebook host's theme: the grid renders dark under a
  dark host (JupyterLab dark, VS Code dark, Colab dark) and light otherwise,
  detecting the host's own theme attributes or computed background (never the OS
  scheme) and switching live when the host theme changes. An explicit `theme`
  option still wins over auto-detection. The row-selection checkbox is now drawn
  by the widget so a tick is clearly distinguishable from an empty box in every
  theme (it previously rendered as a solid black square under a dark host).
- `widget.view` on a view only the browser can produce (a router's shaped rows
  before the grid's first report) no longer blocks waiting for the browser --
  on ipykernel 6 that always ended in `LatticeViewTimeout` with a "coroutine
  'Kernel.do_one_iteration' was never awaited" warning, and no wait is safe
  inside a running cell on any ipykernel. It now raises `LatticeViewNeedsBrowser`
  at once, naming the fix: read `widget.view` again in a later cell once the
  grid has rendered, or register the new `widget.on_view(callback)`, called
  with the view each time the browser reports one (and immediately if one is
  already known) from the kernel's normal comm handling between cells -- this
  works on every ipykernel version. Views the kernel can compute are unchanged
  and still return immediately.

## 1.109.0.0

Lattice Grid 1.109.0. `lattice-grid-pandas`, `lattice-grid-jupyter` and
`lattice-grid-dash` move to 1.109.0.0 together, with these widget fixes.

### Fixed

- Large-mode filters (`contains`, `startsWith`, `endsWith`, `notContains` and the
  text comparison operators) no longer print a pandas `FutureWarning` about
  silently downcasting an object-dtype mask; the same rows are returned as before.
- A column of Python lists, tuples or numpy arrays serializes as real JSON arrays
  instead of their text representation, so a `line` (sparkline) cell spec now
  receives actual numbers instead of rendering blank.
- `widget.view` is now computed in the kernel from the frame and the grid's own
  `state` whenever that is possible, instead of depending on a comm message the
  browser sends back; it is now correct in a later cell under `Run All` (or any
  headless run with no front end attached), rather than silently returning an
  empty or unfiltered view. A router's shaped view, which the browser must
  compute, now blocks for that report with a timeout and raises instead of
  returning the empty placeholder frame.
- `set_data` with a different schema now adds, removes and retypes the grid's
  columns instead of only repainting the old ones with new values (new columns
  were previously never shown). It also prunes any sort, filter or column-width
  state that names a column no longer present, instead of leaving a stale
  reference that crashed the next query with a `KeyError` and left the grid
  showing "Failed to load" in every cell.

## 1.85.0.1

Same grid as 1.85.0.0 (Lattice Grid 1.85.0). `lattice-grid-pandas`,
`lattice-grid-jupyter` and `lattice-grid-dash` move to 1.85.0.1 together; the two
host wrappers pin `lattice-grid-pandas==1.85.0.1`.

### Added

- **Selection, filtered view and state come back to Python.** Click rows in the
  grid and read them as a DataFrame with `widget.selected`; filter or sort in the
  grid and read exactly what is on screen with `widget.view`; save and restore the
  whole grid state (sort, filters, columns, grouping) with `widget.state`. Only row
  keys travel back to Python, and rapid changes are batched into one update.
- **Grid options from Python.** Pass `options=` and `columns=` using snake_case
  names; turn on histograms, profiling (the Statistics panel) and pivot with
  `histograms=`, `profile=` and `pivot=`. Unknown or mistyped options raise a
  `LatticeGridWarning` in Python instead of failing silently in the browser.
- **Large dataframes.** Frames of 100,000 rows and more are served to the grid in
  windows instead of being copied into the browser, so a ten-million-row DataFrame,
  Parquet file or Polars frame scrolls, sorts, filters, searches and groups in
  JupyterLab. Install the extra with `pip install "lattice-grid-jupyter[large]"`.
  See `docs/large-data.md` for the measured limits.
- **Charts in the notebook.** `LatticeChart` draws the grid's chart types from a
  DataFrame, bound to the same data as a grid so selecting and filtering stay in
  step, with `HBox`/`VBox` layout. See `docs/charts.md`.
- **The data router in the notebook.** `LatticeRouter` describes sources and
  specs as plain dicts, updates them without re-sending unchanged data, and hands
  shaped rows back as a DataFrame. See `docs/router.md`.
- Hosts: JupyterLab 4 and Notebook 7 are tested for real; steps for VS Code and
  Colab are in `docs/hosts.md`.

### Internal

- `tools/bump_grid.py` also vendors the chart modules and the data-router module
  from the grid tarball and regenerates the option and chart-spec name tables, so
  every automatic release carries them at the new grid;
  `tests/test_grid_version.py` checks they agree with `GRID_VERSION`.
