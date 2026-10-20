"""BACKLOG-0004772: ``w.view`` is correct across cells with no browser at all.

In the small (non-windowed) mode, ``w.view`` used to depend on a comm message
the browser sends back; under ``Run All`` (and nbconvert/papermill, which never
attach a front end) that message never arrives, so a later cell reading
``w.view`` saw the kernel's own answer instead -- which must now be computed
correctly from the frame and ``state`` it already holds. These tests run a
real notebook through a real kernel with ``nbclient``, exactly as Run All
would, and no browser anywhere in the process.
"""

import textwrap

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")


def _run_cells(*cells: str):
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(textwrap.dedent(c)) for c in cells]
    nbclient.NotebookClient(nb, kernel_name="python3", timeout=60).execute()
    return nb


def test_a_filter_set_in_one_cell_is_seen_by_view_in_the_next_cell_headless():
    # one cell builds the grid and sets a filter in state; the NEXT cell -- as
    # a separate execution, the way Run All drives a notebook -- reads w.view.
    _run_cells(
        """
        import pandas as pd
        from lattice_grid_jupyter import LatticeGridWidget
        df = pd.DataFrame({"name": ["Ada", "Grace", "Linus", "Ada"], "score": [91, 88, 77, 50]})
        w = LatticeGridWidget(df)
        w.state = {"version": 2, "filters": {"col": "score", "op": "gte", "value": 80}}
        """,
        """
        names = sorted(w.view["name"].tolist())
        assert names == ["Ada", "Grace"], names
        assert w.view.shape == (2, 2), w.view.shape
        assert w._view_keys is None   # no front end ever attached to report one
        """,
    )


def test_set_data_with_a_new_schema_is_visible_in_the_next_cell_headless():
    _run_cells(
        """
        import pandas as pd
        from lattice_grid_jupyter import LatticeGridWidget
        w = LatticeGridWidget(pd.DataFrame({"name": ["Ada"], "score": [91]}),
                               state={"version": 2, "sort": [{"col": "score", "dir": "desc"}]})
        w.set_data(pd.DataFrame({"name": ["Zoe"], "region": ["west"]}))
        """,
        """
        assert sorted(c["field"] for c in w._columns) == ["name", "region"]
        assert list(w.view["name"]) == ["Zoe"]   # the stale sort on 'score' did not raise
        """,
    )
