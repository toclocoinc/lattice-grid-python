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


def test_a_routers_shaped_view_is_read_in_a_later_cell_with_no_browser():
    # BACKLOG-0005044: the kernel shapes the route from the source frames and the declaration
    # (collect/rollup join, lookup, coercion), so w.view needs no browser report, under Run All.
    _run_cells(
        """
        import pandas as pd
        from lattice_grid_jupyter import LatticeGridWidget, LatticeRouter
        cust = pd.DataFrame({"cid": ["a", "b", "c"], "plan": ["pro", "free", "pro"],
                             "signup": ["01/02/2024", "15/03/2024", "30/04/2024"]})
        txn = pd.DataFrame({"tid": ["t1", "t2", "t3", "t4"], "cid": ["a", "a", "b", "a"], "amt": [5.0, 3.0, 9.0, 5.0]})
        lab = pd.DataFrame({"cid": ["a", "b"], "churned": [True, False]})
        router = LatticeRouter(sources={
            "customer": {"data": cust, "key": "cid",
                         "fields": {"signup": {"type": "date", "format": "dd/MM/yyyy"}},
                         "join": [{"from": "txn", "many": True, "foreign_key": "cid",
                                   "aggregate": {"n": {"fn": "count"}, "spend": {"fn": "sum", "field": "amt"}}},
                                  {"from": "lab", "local_key": "cid", "fields": ["churned"], "missing": "null"}]},
            "txn": {"data": txn, "key": "tid"}, "lab": {"data": lab, "key": "cid"}}, routes={"customer": "customer"})
        w = LatticeGridWidget(router, route="customer", offline=True)
        w.state = {"version": 2, "filters": {"col": "plan", "op": "eq", "value": "pro"}}
        """,
        """
        v = w.view
        assert list(v.cid) == ["a", "c"], v
        assert list(v.n) == [3, 0] and list(v.spend) == [13.0, 0.0], v
        assert v.churned.iloc[0] == 1 and pd.isna(v.churned.iloc[1]), v.churned.tolist()   # the join's missing: null
        assert str(v.signup.dtype).startswith("datetime64") and v.signup.iloc[0].strftime("%Y-%m-%d") == "2024-02-01"
        assert w._shaped == {} and w._view_keys is None   # no front end ever attached to report anything
        """,
        """
        txn2 = txn.copy(); txn2.loc[0, "amt"] = 100.0
        router.update("txn", txn2)
        assert list(w.view.spend) == [108.0, 0.0], w.view
        """,
    )
