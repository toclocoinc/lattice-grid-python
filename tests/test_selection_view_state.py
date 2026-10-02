"""Card 1616: selection, filtered view and state back to Python."""

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # noqa: E402

from _hosts import launch_browser  # noqa: E402
from _page import page_html, sets  # noqa: E402
from lattice_grid_jupyter import LatticeGridWidget  # noqa: E402


def _rich_frame():
    return pd.DataFrame(
        {
            "name": ["Ada", "Grace", "Linus", "Ken"],
            "score": pd.array([91, None, 77, 50], dtype="Int64"),
            "ratio": [0.9, np.nan, 0.7, 0.5],
            "when": pd.to_datetime(["2021-01-01", None, "2021-03-01", "2021-04-01"]),
            "kind": pd.Categorical(["a", "b", "a", "c"]),
        },
        index=pd.Index([10, 20, 30, 40], name="id"),
    )


# --- selected -----------------------------------------------------------------
def test_selected_is_empty_frame_not_none_with_dtypes():
    w = LatticeGridWidget(_rich_frame())
    sel = w.selected
    assert isinstance(sel, pd.DataFrame) and len(sel) == 0
    assert sel is not None
    assert list(sel.columns) == list(w.df.columns)
    assert (sel.dtypes == w.df.dtypes).all()  # empty, but still typed


def test_selected_keeps_original_index_and_dtypes():
    w = LatticeGridWidget(_rich_frame())
    w._selected_keys = ["2", "0"]  # what the comm delivers: keys only
    sel = w.selected
    assert list(sel.index) == [10, 30]  # original index labels, frame order
    assert sel.index.name == "id"
    assert (sel.dtypes == w.df.dtypes).all()
    assert str(sel["score"].dtype) == "Int64"
    assert isinstance(sel["kind"].dtype, pd.CategoricalDtype)
    assert sel.loc[10, "name"] == "Ada" and sel.loc[30, "score"] == 77


def test_selected_missing_values_survive_nan_none_nat():
    w = LatticeGridWidget(_rich_frame())
    w._selected_keys = ["1"]
    sel = w.selected
    assert pd.isna(sel["score"].iloc[0]) and pd.isna(sel["ratio"].iloc[0])
    assert pd.isna(sel["when"].iloc[0])


def test_selected_ignores_keys_of_deleted_rows():
    w = LatticeGridWidget(_rich_frame())
    w._selected_keys = ["0", "1"]
    w.delete_rows("0")
    assert list(w.selected.index) == [20]


# --- view ---------------------------------------------------------------------
def test_view_is_everything_before_the_grid_reports():
    w = LatticeGridWidget(_rich_frame())
    assert len(w.view) == 4


def test_view_follows_the_grids_filter_and_sort_order():
    w = LatticeGridWidget(_rich_frame())
    w._view_keys = ["3", "0", "2"]  # filtered (row 1 out) and sorted by the grid
    v = w.view
    assert list(v.index) == [40, 10, 30]
    assert (v.dtypes == w.df.dtypes).all()
    w._view_keys = []
    assert len(w.view) == 0 and list(w.view.columns) == list(w.df.columns)


# --- observe: once per change -------------------------------------------------
def test_observe_selected_and_view_fire_once_per_change():
    w = LatticeGridWidget(_rich_frame())
    seen = {"selected": [], "view": []}
    w.observe(lambda c: seen["selected"].append(list(c["new"].index)), names="selected")
    w.observe(lambda c: seen["view"].append(list(c["new"].index)), names="view")

    w._selected_keys = ["1"]
    w._selected_keys = ["1"]            # no change -> no event
    w._selected_keys = ["1", "2"]
    w._view_keys = ["0", "1"]
    w._view_keys = ["0", "1"]           # no change -> no event
    w._view_keys = ["0"]

    assert seen["selected"] == [[20], [20, 30]]
    assert seen["view"] == [[10, 20], [10]]


def test_observe_payload_is_a_dataframe_with_dtypes():
    w = LatticeGridWidget(_rich_frame())
    got = []
    w.observe(lambda c: got.append(c["new"]), names="selected")
    w._selected_keys = ["0"]
    assert (got[0].dtypes == w.df.dtypes).all()


# --- state --------------------------------------------------------------------
def test_state_round_trip_is_a_plain_dict_and_restores():
    w = LatticeGridWidget(_rich_frame())
    sent = []
    real = w.send_state
    w.send_state = lambda key=None: (sent.append(key), real(key))[1]
    st = {"version": 2, "sort": [{"col": "score", "dir": "desc"}], "quick": "a"}
    w.state = st
    assert w.state == st and type(w.state) is dict
    assert "state" in (sent[0] if isinstance(sent[0], (list, set, tuple)) else [sent[0]])
    import json
    json.dumps(w.state)  # plain, serialisable


def test_state_can_be_given_to_the_constructor():
    st = {"version": 2, "sort": [{"col": "name", "dir": "asc"}]}
    assert LatticeGridWidget(_rich_frame(), state=st).state == st


# --- real browser -------------------------------------------------------------
def _df():
    return pd.DataFrame({
        "name": ["Ada", "Grace", "Linus", "Ada"],
        "score": [91, 88, 77, 50],
    }, index=[5, 6, 7, 8])


def _open(p, w):
    browser = launch_browser(p)
    if browser is None:
        pytest.skip("no real browser available to launch")
    page = browser.new_page(viewport={"width": 1000, "height": 600})
    page.set_content(page_html(w))
    page.wait_for_function("window.__rendered === true || window.__error", timeout=30000)
    assert page.evaluate("window.__error") is None
    page.wait_for_selector(".lat-cell[role=gridcell]", timeout=10000)
    return browser, page


def test_real_browser_selection_filter_sort_and_state_reach_python():
    w = LatticeGridWidget(_df(), offline=True)
    with sync_playwright() as p:
        browser, page = _open(p, w)
        try:
            # a REAL click on a row selects it; ctrl-click adds a second
            page.locator(".lat-cell[role=gridcell]", has_text="Grace").first.click()
            page.locator(".lat-cell[role=gridcell]", has_text="Linus").first.click(modifiers=["Control"])
            page.wait_for_function("window.__sets.some(s => s[0]==='_selected_keys' && s[1].length===2)", timeout=4000)
            sel_keys = sets(page, "_selected_keys")[-1]

            # a real filter and a real sort: quick search "Ada", then sort score asc
            page.evaluate("""(() => { const g = document.querySelector('div[style*=height]').__latticeGrid;
                g.filters.quick('Ada'); g.sort.set([{col:'score', dir:'asc'}]); })()""")
            page.wait_for_function("window.__sets.some(s => s[0]==='_view_keys' && s[1].length===2)", timeout=4000)
            view_keys = sets(page, "_view_keys")[-1]
            page.wait_for_function("window.__sets.some(s => s[0]==='state' && s[1].quick==='Ada')", timeout=4000)
            state = sets(page, "state")[-1]
        finally:
            browser.close()

    w._selected_keys = sel_keys
    w._view_keys = view_keys
    w.state = state
    assert sorted(w.selected["name"]) == ["Grace", "Linus"]
    assert list(w.selected.index) == [6, 7]  # original index
    assert w.selected["score"].dtype == w.df["score"].dtype
    assert list(w.view["name"]) == ["Ada", "Ada"]
    assert list(w.view["score"]) == [50, 91]  # the grid's sort order
    assert list(w.view.index) == [8, 5]
    assert state["quick"] == "Ada" and state["sort"][0]["col"] == "score"


def test_real_browser_filtering_is_debounced_to_one_report():
    w = LatticeGridWidget(_df(), offline=True)
    with sync_playwright() as p:
        browser, page = _open(p, w)
        try:
            before = len(sets(page, "_view_keys"))
            # a typing storm: six keystrokes inside one debounce window
            page.evaluate("""(() => { const g = document.querySelector('div[style*=height]').__latticeGrid;
                for (const t of ['A','Ad','Ada','Ada ','Ada','Ada']) g.filters.quick(t); })()""")
            page.wait_for_timeout(700)
            after = sets(page, "_view_keys")
        finally:
            browser.close()
    assert len(after) - before == 1, after  # exactly one report for the storm
    assert after[-1] == ["0", "3"]


def test_real_browser_python_state_restores_the_grid():
    w = LatticeGridWidget(_df(), offline=True)
    with sync_playwright() as p:
        browser, page = _open(p, w)
        try:
            page.evaluate("""window.__pyset('state', {version: 2, quick: 'Linus',
                sort: [{col:'score', dir:'desc'}]})""")
            page.wait_for_function("window.__sets.some(s => s[0]==='_view_keys' && s[1].length===1)", timeout=4000)
            assert sets(page, "_view_keys")[-1] == ["2"]
            n = len(sets(page, "state"))
            page.wait_for_timeout(500)
            # the restore's echo must not loop forever
            assert len(sets(page, "state")) - n <= 1
        finally:
            browser.close()
