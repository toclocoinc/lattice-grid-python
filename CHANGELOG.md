# Changelog

Wrapper versions are `<grid version>.<revision>` (see `RELEASING.md`). Automatic
grid bumps (`Lattice Grid X -> Python wrappers X.0`) change only the vendored grid
and are not listed here; a revision above `.0` is.

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
