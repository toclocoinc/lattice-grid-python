"""Card 1618: the windowed source in a real browser, held equal to the grid's own
client-side path on the same frame.

Each test renders the widget's real front end (widget.js + the vendored grid)
twice in Chromium: once in the client mode (every row in the browser, the
grid computes) and once windowed (no rows in the browser; every query goes over
the comm to the real Python widget). The same operation is applied through the
grid's own API to both and the answers are compared. Skipped by name when no
browser can be launched, never faked.
"""

import json
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # noqa: E402

from _hosts import launch_browser  # noqa: E402
from _page import bind_python, page_html, sets  # noqa: E402
from lattice_grid_jupyter import LatticeGridWidget  # noqa: E402

GRID = "document.querySelector('div[style*=height]').__latticeGrid"


def frame(n=60, seed=5):
    r = np.random.default_rng(seed)
    names = np.array(["Ada", "Grace", "Linus", "Ken", "Barbara", "Alan"])
    df = pd.DataFrame({
        "name": names[r.integers(0, len(names), n)].astype(object),
        "score": r.integers(0, 50, n).astype("int64"),
        "ratio": np.round(r.random(n) * 10, 3),
        "kind": pd.Categorical(np.array(["a", "b", "c"])[r.integers(0, 3, n)]),
        "ok": r.random(n) > 0.5,
    })
    df.loc[[3, 17], "ratio"] = np.nan
    df.loc[[5], "name"] = None
    return df


class Session:
    """One real browser; pages for a widget in either mode."""

    def __init__(self, p):
        self.browser = launch_browser(p)
        if self.browser is None:
            pytest.skip("no real browser available to launch")

    def open(self, w, delay_s=0.0, tz="UTC"):
        log = []
        ctx = self.browser.new_context(viewport={"width": 1200, "height": 700}, timezone_id=tz)
        page = ctx.new_page()
        bind_python(page, w, delay_s=delay_s, log=log)
        page.set_content(page_html(w))
        page.wait_for_function("window.__rendered === true || window.__error", timeout=30000)
        assert page.evaluate("window.__error") is None, page.evaluate("window.__error")
        page.wait_for_selector(".lat-cell[role=gridcell]", timeout=15000)
        page.lg_log = log
        return page

    def close(self):
        self.browser.close()


def settle(page, ms=150):
    """Wait until every display row is loaded (no window in flight)."""
    page.wait_for_function(
        f"""(() => {{ const g = {GRID}; const n = g.rows.count(); if (!n) return true;
          for (let i = 0; i < n; i++) {{ const r = g.rows.get(i); if (!r || (!r.group && !r.data)) return false; }}
          return true; }})()""", timeout=15000)
    page.wait_for_timeout(ms)


def leaf_keys(page):
    settle(page)
    return page.evaluate(f"""(() => {{ const g = {GRID}; const out = [];
      for (let i = 0; i < g.rows.count(); i++) {{ const r = g.rows.get(i); if (r && !r.group) out.push(String(r.key)); }}
      return out; }})()""")


def both(sess, df, setup_js, settle_rows=True, **kw):
    """Apply the same grid-API calls to a client and a windowed widget."""
    out = []
    for windowed in (False, True):
        w = LatticeGridWidget(df, offline=True, windowed=windowed, **kw)
        page = sess.open(w)
        page.evaluate(f"(() => {{ const g = {GRID}; {setup_js} }})()")
        if settle_rows:
            settle(page, 400)
        out.append((w, page))
    return out


# --- sort pushed ----------------------------------------------------------------
def test_real_browser_sort_pushed_equals_client():
    df = frame()
    with sync_playwright() as p:
        s = Session(p)
        try:
            for spec in ("[{col:'ratio', dir:'desc'}]",
                         "[{col:'kind', dir:'asc'}, {col:'score', dir:'desc'}]",
                         "[{col:'name', dir:'asc'}]"):
                (wc, client), (ww, win) = both(s, df, f"g.sort.set({spec});")
                assert leaf_keys(win) == leaf_keys(client), spec
                assert "execute" in win.lg_log and not client.lg_log
        finally:
            s.close()


# --- filter / quick search pushed ---------------------------------------------------
FILTERS = [
    "{col:'score', op:'gt', value:30}",
    "{col:'name', op:'eq', value:'ada'}",
    "{col:'name', op:'ne', value:'ada'}",
    "{col:'name', op:'contains', value:'AR'}",
    "{col:'name', op:'in', value:['ken', 'Alan']}",
    "{col:'ratio', op:'blank'}",
    "{col:'ratio', op:'between', value:[2, 6]}",
    "{col:'kind', op:'eq', value:'b'}",
    "{col:'ok', op:'eq', value:true}",
    "{op:'or', conditions:[{col:'kind', op:'eq', value:'a'}, {col:'score', op:'lt', value:5}]}",
]


def test_real_browser_filter_pushed_equals_client():
    df = frame()
    with sync_playwright() as p:
        s = Session(p)
        try:
            for f in FILTERS:
                (wc, client), (ww, win) = both(s, df, f"g.filters.set({f});")
                got, want = leaf_keys(win), leaf_keys(client)
                assert got == want, f
                assert len(want) < len(df), f  # the filter really narrowed
            (wc, client), (ww, win) = both(s, df, "g.filters.quick('lin');")
            assert leaf_keys(win) == leaf_keys(client) and leaf_keys(client)
        finally:
            s.close()


# --- window request ---------------------------------------------------------------
def test_real_browser_window_request_fetches_far_rows_on_scroll():
    df = pd.DataFrame({"n": np.arange(5000), "label": [f"row-{i}" for i in range(5000)]})
    w = LatticeGridWidget(df, offline=True, windowed=True)
    with sync_playwright() as p:
        s = Session(p)
        try:
            page = s.open(w)
            first = page.lg_log.count("execute")
            assert page.evaluate(f"{GRID}.rows.count()") == 5000
            assert page.evaluate(f"{GRID}.rows.get(3000)") is None or \
                page.evaluate(f"{GRID}.rows.get(3000).data") is None  # never sent up front
            page.evaluate(f"""(() => {{ const host = document.querySelector('div[style*=height]');
              const vp = [...host.querySelectorAll('div')].find(d => d.scrollHeight > d.clientHeight + 1000);
              vp.scrollTop = vp.scrollHeight * 3000 / 5000; }})()""")
            page.wait_for_function(f"(() => {{ const r = {GRID}.rows.get(3000); return r && r.data; }})()",
                                   timeout=10000)
            row = page.evaluate(f"{GRID}.rows.get(3000).data")
            assert row["n"] == 3000 and row["label"] == "row-3000"
            page.wait_for_selector(".lat-cell[role=gridcell]:has-text('row-3000')", timeout=5000)
            assert page.lg_log.count("execute") > first
        finally:
            s.close()


# --- loading rows / banner while a window is in flight ------------------------------
def test_real_browser_loading_shows_while_a_window_is_in_flight():
    df = pd.DataFrame({"n": np.arange(5000), "label": [f"row-{i}" for i in range(5000)]})
    w = LatticeGridWidget(df, offline=True, windowed=True)
    with sync_playwright() as p:
        s = Session(p)
        try:
            page = s.open(w, delay_s=1.2)
            page.evaluate(f"""(() => {{ const host = document.querySelector('div[style*=height]');
              const vp = [...host.querySelectorAll('div')].find(d => d.scrollHeight > d.clientHeight + 1000);
              vp.scrollTop = vp.scrollHeight * 0.7; }})()""")
            # the grid's loading treatment (1.79): shimmer bars / a "Loading rows" block,
            # and aria-busy on the body, while Python has not answered yet
            page.wait_for_selector("[aria-busy=true]", timeout=3000)
            page.wait_for_selector(".lat-skeleton-bar, .lat-cell--loading-block", timeout=3000)
            # ...and gone once the window has landed
            page.wait_for_function("!document.querySelector('[aria-busy=true]') && "
                                   "!document.querySelector('.lat-skeleton-bar, .lat-cell--loading-block')",
                                   timeout=10000)
            assert page.evaluate(f"{GRID}.rows.get(3500).data.n") == 3500
        finally:
            s.close()


# --- group-by aggregates -----------------------------------------------------------
def group_rows(page):
    """The group rows, and the leaf keys under the first group (loaded in its window)."""
    page.wait_for_function(f"""(() => {{ const g = {GRID}; let groups = 0;
      for (let i = 0; i < Math.min(g.rows.count(), 12); i++) {{ const r = g.rows.get(i);
        if (!r || (!r.group && !r.data)) return false; if (r.group) groups++; }}
      return groups > 0; }})()""", timeout=15000)
    page.wait_for_timeout(300)
    return page.evaluate(f"""(() => {{ const g = {GRID}; const out = [], first = [];
      let seen = 0;
      for (let i = 0; i < g.rows.count(); i++) {{ const r = g.rows.get(i);
        if (r && r.group) {{ seen++; out.push({{key: String(r.key), leafCount: r.leafCount,
          score: r.totals ? r.totals.score : undefined}}); }}
        else if (seen === 1 && r && r.data) first.push(String(r.key)); }}
      return {{groups: out, firstLeaves: first}}; }})()""")


def test_real_browser_group_by_aggregates_pushed_equals_client():
    df = frame()
    with sync_playwright() as p:
        s = Session(p)
        try:
            (wc, client), (ww, win) = both(s, df, "g.columns.group(['kind']);",
                                           columns=[{"field": "score", "total": "sum"}], settle_rows=False)
            want, got = group_rows(client), group_rows(win)
            assert len(want["groups"]) == 3 and got["groups"] == want["groups"]
            assert want["firstLeaves"] and got["firstLeaves"] == want["firstLeaves"]
            assert "grouped" in win.lg_log
        finally:
            s.close()


# --- distinct values for filter menus ----------------------------------------------
POPUP = "(() => { const d = document.querySelector('[role=dialog]'); return d ? d.innerText : ''; })()"


def test_real_browser_filter_menu_values_come_from_python():
    df = frame()
    df.loc[40:, "name"] = "Zed"  # a value outside the first window's rows
    with sync_playwright() as p:
        s = Session(p)
        try:
            lists = []
            for windowed in (False, True):
                w = LatticeGridWidget(df, offline=True, windowed=windowed, page_size=20,
                                      columns=[{"field": "name", "filter": {"type": "set"}}])
                page = s.open(w)
                # the real header menu, a set filter: the value list
                page.locator("button.lat-header-filter[aria-label='Filter name']").click()
                page.wait_for_function(POPUP + ".includes('Zed')", timeout=8000)
                text = page.evaluate(POPUP)
                lists.append(sorted(v for v in df["name"].dropna().unique() if v in text))
                if windowed:
                    assert "grouped" in page.lg_log      # asked of Python: a group-by with a count
                    assert page.evaluate(f"{GRID}.rows.get(45) && {GRID}.rows.get(45).data") is None
            assert lists[0] == lists[1] == sorted(df["name"].dropna().unique())
        finally:
            s.close()


# --- header histograms --------------------------------------------------------------
def facet_of(page, col):
    page.evaluate(f"{GRID}.facets.get('{col}')")
    page.wait_for_function(
        f"(() => {{ const f = {GRID}.facets.get('{col}'); return f && f.counts && !f.stale; }})()",
        timeout=10000)
    return page.evaluate(f"""(() => {{ const f = {GRID}.facets.get('{col}');
      return {{bounds: f.bounds, counts: Array.from(f.counts),
               unfiltered: f.unfiltered ? Array.from(f.unfiltered) : null}}; }})()""")


def test_real_browser_header_histograms_pushed_equal_client():
    df = frame()
    with sync_playwright() as p:
        s = Session(p)
        try:
            pages = both(s, df, "g.filters.set({col:'kind', op:'eq', value:'a'});", histograms=True)
            (wc, client), (ww, win) = pages
            for col in ("score", "ratio", "name", "ok"):
                a, b = facet_of(client, col), facet_of(win, col)
                assert b["counts"] == a["counts"], col
                assert [x.get("from", x.get("value")) for x in b["bounds"]["buckets"]] == \
                    [x.get("from", x.get("value")) for x in a["bounds"]["buckets"]], col
            assert "facet" in win.lg_log
        finally:
            s.close()


# --- profile statistics ---------------------------------------------------------------
def test_real_browser_profile_pushed_equals_client():
    df = frame()
    with sync_playwright() as p:
        s = Session(p)
        try:
            (wc, client), (ww, win) = both(s, df, "g.filters.set({col:'score', op:'gte', value:10});",
                                           profile=True)
            for col in ("ratio", "score", "name"):
                a = client.evaluate(f"{GRID}.statistics.profile('{col}')")
                win.evaluate(f"{GRID}.statistics.profile('{col}')")
                win.wait_for_function(f"!!{GRID}.statistics.profile('{col}')", timeout=10000)
                b = win.evaluate(f"{GRID}.statistics.profile('{col}')")
                for k in ("rows", "present", "missing", "numeric", "distinct", "outliers"):
                    assert b[k] == a[k], (col, k, a[k], b[k])
                for k in ("min", "max", "mean", "median", "q1", "q3", "stddev"):
                    assert (a[k] is None and b[k] is None) or b[k] == pytest.approx(a[k], rel=1e-9), (col, k)
                assert [h["count"] for h in b["histogram"]] == [h["count"] for h in a["histogram"]], col
                assert b.get("topValues") == a.get("topValues"), col
            assert "profile" in win.lg_log
        finally:
            s.close()


# --- selection, edits and threshold crossing in the browser ---------------------------------
def test_real_browser_selection_and_edit_reach_python_when_windowed():
    df = frame()
    df.index = pd.Index(range(500, 560), name="id")
    w = LatticeGridWidget(df, offline=True, windowed=True)
    with sync_playwright() as p:
        s = Session(p)
        try:
            page = s.open(w)
            settle(page)
            page.locator(".lat-cell[role=gridcell]").nth(2).click()
            page.wait_for_function("window.__sets.some(s => s[0]==='_selected_keys' && s[1].length===1)",
                                   timeout=4000)
            w._selected_keys = sets(page, "_selected_keys")[-1]
            sel = w.selected
            assert len(sel) == 1 and sel.index[0] in df.index and (sel.dtypes == df.dtypes).all()

            cell = page.locator(".lat-row").nth(0).locator(".lat-cell[role=gridcell]").nth(2)
            cell.dblclick()
            page.wait_for_selector("input.lat-editor__input", timeout=5000)
            page.keyboard.press("Control+A")
            page.keyboard.type("49")
            page.keyboard.press("Enter")
            page.wait_for_function("true", timeout=1000)
            page.wait_for_timeout(500)
            assert "mutate" in page.lg_log
        finally:
            s.close()
    assert (w.df["score"] == 49).any()


def test_real_browser_set_data_across_the_threshold_switches_the_source():
    small = frame(40)
    big = pd.DataFrame({"name": [f"n{i}" for i in range(400)], "score": np.arange(400),
                        "ratio": np.zeros(400), "kind": pd.Categorical(["a"] * 400), "ok": [True] * 400})
    w = LatticeGridWidget(small, offline=True, large_threshold=100)
    with sync_playwright() as p:
        s = Session(p)
        try:
            page = s.open(w)
            assert w._mode == "client" and page.evaluate(f"{GRID}.rows.count()") == 40
            assert "execute" not in page.lg_log
            w.set_data(big)
            page.evaluate("""(st) => { for (const [k, v] of Object.entries(st)) window.__pyset(k, v); }""",
                          {"_mode": w._mode, "_columnar": w._columnar, "_engine_caps": w._engine_caps,
                           "_columns": w._columns, "_data_version": w._data_version})
            page.wait_for_function(f"{GRID}.rows.count() === 400", timeout=10000)
            page.wait_for_function(f"!!({GRID}.rows.get(0) && {GRID}.rows.get(0).data)", timeout=10000)
            assert "execute" in page.lg_log and w._mode == "windowed"
            assert page.evaluate(f"{GRID}.rows.get(0).data.name") == "n0"
            w.set_data(small)
            page.evaluate("""(st) => { for (const [k, v] of Object.entries(st)) window.__pyset(k, v); }""",
                          {"_mode": w._mode, "_columnar": w._columnar, "_columns": w._columns,
                           "_data_version": w._data_version})
            page.wait_for_function(f"{GRID}.rows.count() === 40", timeout=10000)
        finally:
            s.close()
