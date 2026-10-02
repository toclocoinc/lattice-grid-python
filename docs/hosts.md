# Supported notebook hosts

`LatticeGridWidget` is an anywidget, so it renders wherever ipywidgets render.
What is **proven by an automated test** and what needs a **manual check** is
different, and this page keeps them apart.

| Host | Status | How |
|------|--------|-----|
| JupyterLab 4 | proven by test | `tests/test_hosts_jupyter.py` starts a real server, opens a notebook in headless Chrome, renders the widget, edits a cell for real, and reads the changed DataFrame back from the kernel |
| Jupyter Notebook 7 | proven by test | same test, `notebook` flavour |
| VS Code (Jupyter extension) | **manual check** | cannot be driven headless from CI |
| Google Colab | **manual check** | cannot be driven headless from CI |

Grid version under test: `lattice_grid_pandas.GRID_VERSION`, which reads the
`grid_bundle.json` that `tools/bump_grid.py` writes from the npm tarball (the
widget re-exports it). `tests/test_grid_version.py` fails if that record, the
vendored bundles' banners, the generated name tables and the wheel's version
disagree.

## Manual check: VS Code

1. Install VS Code with the *Jupyter* and *Python* extensions, and a Python
   environment with `pip install lattice-grid-jupyter`.
2. New notebook, select that environment as the kernel.
3. Run this cell:
   ```python
   import pandas as pd
   from lattice_grid_jupyter import LatticeGridWidget
   df = pd.DataFrame({"name": ["Ada", "Grace", "Linus"], "score": [91, 88, 77]})
   w = LatticeGridWidget(df)          # CDN; use offline=True with no network
   w
   ```
4. Expect: a grid with 3 rows and a `localhost`-free (unwatermarked) render.
5. Double-click the `91`, type `99`, press Enter.
6. Run a new cell `w.df` : expect `score` of row 0 to be `99`, dtype `int64`.
7. (1616/1617) Click a row, run `w.selected`: expect a one-row DataFrame with the
   original index. Type in the grid's quick search, run `w.view`: expect only
   matching rows.
8. Record: VS Code version, extension versions, pass/fail per step.

## Manual check: Google Colab

1. Open a new Colab notebook.
2. First cell: `!pip install -q lattice-grid-jupyter` (restart the runtime if
   Colab asks).
3. Enable custom widgets: `from google.colab import output; output.enable_custom_widget_manager()`.
4. Run the same cell as above. Colab is **not** localhost, so the grid shows the
   licence watermark unless you pass `licence="LG-..."` (or a free-tier key).
5. Edit a cell, run `w.df`, and check steps 5-7 above.
6. Record: pass/fail per step, and whether the CDN bundle loaded (the
   default) or `offline=True` was needed.
