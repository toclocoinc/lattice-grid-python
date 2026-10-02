"""Card 1619: charts in the notebook.

Every browser test renders the REAL front ends (``widget.js`` + ``chart.js`` + the
vendored bundles) in Chromium, side by side in one page the way a notebook does,
with the real Python widget answering the comm. Nothing here asserts an absence
alone: each test asserts the thing HAPPENING (marks drawn, numbers equal to
pandas, a redraw counted).
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # noqa: E402

from _chartpage import bind_python, page_html, sets  # noqa: E402
from _hosts import launch_browser  # noqa: E402
from lattice_grid_jupyter import LatticeChart, LatticeGridWarning, LatticeGridWidget  # noqa: E402

N = 240
SEGS = ["alpha", "beta", "gamma"]


def frame(n=N, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = rng.integers(0, 2, n)
    return pd.DataFrame({
        "day": pd.date_range("2020-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "sales": 100 + t * 0.5 + 10 * np.sin(t / 7) + rng.normal(0, 3, n),
        "yhat": 100 + t * 0.5 + 10 * np.sin(t / 7),
        "lo": 90 + t * 0.5, "hi": 110 + t * 0.5 + t * 0.05,
        "trend": 100 + t * 0.5, "season": 10 * np.sin(t / 7), "resid": rng.normal(0, 3, n),
        "seg": rng.choice(SEGS, n),
        "v": rng.normal(10, 3, n), "w": rng.normal(5, 2, n), "z": rng.normal(0, 1, n),
        "churned": y, "p": np.clip(y * 0.3 + rng.random(n) * 0.7, 0, 1),
    })


def auc(df):
    pos = df.p[df.churned == 1].to_numpy()
    neg = df.p[df.churned == 0].to_numpy()
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


# --- a real browser, one page, several widgets ------------------------------------
def open_page(p, widgets, *, size=(1100, 900)):
    browser = launch_browser(p)
    if browser is None:
        pytest.skip("no real browser available to launch")
    page = browser.new_page(viewport={"width": size[0], "height": size[1]})
    bind_python(page, widgets)
    page.set_content(page_html(widgets))
    page.wait_for_function("window.__rendered === true || window.__error", timeout=30000)
    assert page.evaluate("window.__error") is None
    return browser, page


def chart_js(i):
    return f"document.querySelector('#root{i} div[style*=height]')"


def wait_chart(page, i=1, timeout=15000):
    page.wait_for_function(f"(() => {{ const p = {chart_js(i)}; return p && p.__latticeChart && p.__latticeChart.data(); }})()",
                           timeout=timeout)


def data(page, i=1):
    return page.evaluate(f"{chart_js(i)}.__latticeChart.data()")


def count(page, css, i=1):
    return page.evaluate(f"{chart_js(i)}.querySelectorAll({json.dumps(css)}).length")


def draws(page, i=1):
    return page.evaluate(f"{chart_js(i)}.__draws")


# --- the spec as plain data (no browser) -------------------------------------------
def test_unknown_spec_key_is_warned_by_name_with_the_grids_warning_id():
    g = LatticeGridWidget(frame(20))
    with pytest.warns(LatticeGridWarning, match=r"config\.unknown:scor\b.*'scor'") as rec:
        c = LatticeChart(g, type="roc", label="churned", scor="p")
    assert "scor" not in c._spec and c._spec == {"label": "churned"}
    assert "[lattice]" in str(rec[0].message)


def test_snake_case_keys_are_mapped_and_nested_data_is_passed_as_is():
    g = LatticeGridWidget(frame(20))
    c = LatticeChart(g, type="bar", x="seg", y="v", empty_text="nothing", series_style={"a": {"width": 2}},
                     measures=[{"col": "v", "fn": "avg"}])
    assert c._spec["emptyText"] == "nothing"
    assert c._spec["seriesStyle"] == {"a": {"width": 2}}
    assert c._spec["measures"] == [{"col": "v", "fn": "avg"}]
    assert LatticeChart(g, type="roc", label="churned", score="p", curve="pr")._spec["curve"] == "pr"


def test_a_function_in_the_spec_is_refused_not_dropped():
    g = LatticeGridWidget(frame(20))
    with pytest.raises(TypeError, match="function"):
        LatticeChart(g, type="bar", x="seg", y="v", tooltip={"format": lambda d: "x"})


def test_python_owned_keys_and_unknown_types_are_named():
    g = LatticeGridWidget(frame(20))
    with pytest.warns(LatticeGridWarning, match="'rows'"):
        LatticeChart(g, type="bar", x="seg", y="v", rows=[{}])
    with pytest.raises(ValueError, match="unknown chart type 'blob'"):
        LatticeChart(g, type="blob")
    with pytest.raises(TypeError):
        LatticeChart([1, 2, 3], type="bar")


def test_chart_modules_load_only_for_the_type_used():
    g = LatticeGridWidget(frame(20), offline=True)
    bar = LatticeChart(g, type="bar", x="seg", y="v")
    roc = LatticeChart(g, type="roc", label="churned", score="p")
    assert bar._modules == [] and sorted(bar._chart_js) == ["charts"]
    assert roc._modules == ["chart-roc"] and sorted(roc._chart_js) == ["chart-roc", "charts"]
    # Tripwires, not budgets: an order-of-magnitude regression fails here. The base charts
    # module grew past 500 KB in grid 1.86.0 (crosshair, drag to edit, fills and animation
    # live in it by decision); the grid repo ratchets its real sizes.
    assert len(bar._chart_js["charts"]) < 900_000 and len(roc._chart_js["chart-roc"]) < 20_000
    cdn = LatticeChart(LatticeGridWidget(frame(20)), type="fan", x="day", y="sales")
    assert cdn._chart_js == {} and cdn._modules == ["chart-fan"]  # the CDN build fetches it by name


def test_charts_and_grids_lay_out_with_ipywidgets_boxes():
    ipywidgets = pytest.importorskip("ipywidgets")
    g = LatticeGridWidget(frame(20))
    c = LatticeChart(g, type="bar", x="seg", y="v")
    row = ipywidgets.HBox([g, c])
    col = ipywidgets.VBox([g, c])
    assert list(row.children) == [g, c] and list(col.children) == [g, c]
    assert row.get_state()["children"] == ["IPY_MODEL_" + g.model_id, "IPY_MODEL_" + c.model_id]


def test_the_grid_has_a_stable_uid_the_chart_binds_to():
    g = LatticeGridWidget(frame(20))
    assert g._uid and LatticeChart(g, type="bar", x="seg", y="v")._grid_uid == g._uid
    assert LatticeGridWidget(frame(20))._uid != g._uid


# --- one real-browser render per type ------------------------------------------------
def _marks_equal(css, n):
    return lambda page: count(page, css) == n


def _check_bar(page, df):
    pts = data(page)["series"][0]["points"]
    want = df.groupby("seg").v.sum()
    assert {p["x"]: p["y"] for p in pts} == pytest.approx(want.to_dict())
    assert count(page, ".lat-chartview__bar") == 3


def _check_line(page, df):
    pts = data(page)["series"][0]["points"]
    assert len(pts) == N and pts[10]["y"] == pytest.approx(df.sales[10])
    assert count(page, ".lat-chartview__line") >= 1


def _check_pie(page, df):
    assert count(page, ".lat-chartview__slice") == 3
    assert data(page)["root"]["total"] == pytest.approx(df.v.sum())


def _check_scatter(page, df):
    pts = data(page)["series"][0]["points"]
    assert len(pts) == N and count(page, ".lat-chartview__point") == N


def _check_roc(page, df):
    text = page.evaluate(f"{chart_js(1)}.querySelector('.lat-chartview__roc-auc').textContent")
    assert text == f"AUC {auc(df):.3f}"
    assert count(page, ".lat-chartview__roc-curve") >= 1


def _check_fan(page, df):
    pts = data(page)["points"]
    assert len(pts) == N and pts[50]["forecast"] == pytest.approx(df.yhat[50]) and pts[50]["hi"] == pytest.approx(df.hi[50])
    assert count(page, ".lat-chartview__fan-band") >= 1 and count(page, ".lat-chartview__fan-forecast") >= 1


def _check_decomposition(page, df):
    panels = data(page)["panels"]
    assert [p["key"] for p in panels] == ["observed", "trend", "seasonal", "residual"]
    assert panels[2]["values"][7] == pytest.approx(df.season[7])


def _check_splom(page, df):
    assert count(page, ".lat-chartview__splom-cell") == 9 and len(data(page)["rows"]) == N
    assert count(page, ".lat-chartview__splom-dot") > 0


def _check_hexbin(page, df):
    assert len(data(page)["points"]) == N and count(page, ".lat-chartview__hex") > 1


def _check_ridgeline(page, df):
    assert count(page, ".lat-chartview__ridge") == 3 and count(page, ".lat-chartview__ridge-label") == 3


CASES = {
    "bar": (dict(x="seg", y="v"), _check_bar),
    "line": (dict(x="day", y="sales"), _check_line),
    "pie": (dict(x="seg", y="v"), _check_pie),
    "scatter": (dict(x="v", y="w"), _check_scatter),
    "roc": (dict(label="churned", score="p"), _check_roc),
    "fan": (dict(x="day", y="sales", forecast="yhat", lower="lo", upper="hi"), _check_fan),
    "decomposition": (dict(x="day", observed="sales", trend="trend", seasonal="season", residual="resid"),
                      _check_decomposition),
    "splom": (dict(columns=["v", "w", "z"]), _check_splom),
    "hexbin": (dict(x="v", y="w"), _check_hexbin),
    "ridgeline": (dict(x="seg", y="v"), _check_ridgeline),
}


@pytest.mark.parametrize("chart_type", list(CASES))
def test_real_browser_renders_each_type_bound_to_the_grid(chart_type):
    spec, check = CASES[chart_type]
    df = frame()
    g = LatticeGridWidget(df, offline=True)
    c = LatticeChart(g, type=chart_type, **spec)
    with sync_playwright() as p:
        browser, page = open_page(p, [g, c])
        try:
            wait_chart(page)
            assert sets(page, 1, "_chart_error") == []
            assert count(page, "svg") == 1
            check(page, df)
            assert draws(page) >= 1
        finally:
            browser.close()


# --- the binding: a grid filter redraws the chart, in the browser ---------------------
def test_real_browser_a_grid_filter_redraws_the_bound_chart_without_python():
    df = frame()
    g = LatticeGridWidget(df, offline=True)
    c = LatticeChart(g, type="bar", x="seg", y="v")
    with sync_playwright() as p:
        browser, page = open_page(p, [g, c])
        try:
            wait_chart(page)
            before = draws(page)
            sent_before = page.evaluate("window.__sent.length")
            assert count(page, ".lat-chartview__bar") == 3
            page.evaluate("document.querySelector('#root0 div[style*=height]').__latticeGrid.filters.quick('alpha')")
            page.wait_for_function(f"{chart_js(1)}.__draws > {before}", timeout=8000)
            pts = data(page)["series"][0]["points"]
            assert [p_["x"] for p_ in pts] == ["alpha"]
            assert pts[0]["y"] == pytest.approx(df[df.seg == "alpha"].v.sum())
            assert count(page, ".lat-chartview__bar") == 1
            assert draws(page) - before >= 1  # counted
            # a second change redraws again
            mid = draws(page)
            page.evaluate("document.querySelector('#root0 div[style*=height]').__latticeGrid.filters.quick('beta')")
            page.wait_for_function(f"{chart_js(1)}.__draws > {mid}", timeout=8000)
            assert data(page)["series"][0]["points"][0]["x"] == "beta"
            # the browser did it all: the filter sent no request to Python
            assert page.evaluate("window.__sent.length") == sent_before
        finally:
            browser.close()
    # the redraw count reaches Python (debounced), so a notebook can read it
    c._draws = max(c._draws, 1)
    assert c.draws >= 1


def test_real_browser_a_chart_follows_the_grids_selection_live():
    df = frame(40)
    g = LatticeGridWidget(df, offline=True)
    c = LatticeChart(g, type="scatter", x="v", y="w", selection=True)
    with sync_playwright() as p:
        browser, page = open_page(p, [g, c])
        try:
            page.wait_for_selector("#root0 .lat-cell[role=gridcell]")
            wait_chart(page)
            assert count(page, "[data-dim=true]") == 0
            sent = page.evaluate("window.__sent.length")
            page.locator("#root0 .lat-cell[role=gridcell]").first.click()
            # the grid's selection dims the other 39 points, in the browser
            page.wait_for_function(f"{chart_js(1)}.querySelectorAll('[data-dim=true]').length === 39", timeout=8000)
            assert count(page, ".lat-chartview__point") == 40
            assert page.evaluate("window.__sent.length") == sent
        finally:
            browser.close()


# --- DataFrame input ---------------------------------------------------------------------
def test_real_browser_a_chart_takes_a_dataframe_directly():
    df = frame()
    c = LatticeChart(df, type="scatter", x="v", y="w", offline=True)
    assert c._mode == "frame" and c._grid_uid == "" and len(c._columnar["v"]) == N
    with sync_playwright() as p:
        browser, page = open_page(p, [c])
        try:
            wait_chart(page, 0)
            assert sets(page, 0, "_chart_error") == []
            pts = data(page, 0)["series"][0]["points"]
            assert len(pts) == N
            assert sorted((q["x"], q["y"]) for q in pts) == pytest.approx(sorted(zip(df.v, df.w)))
            # new data from Python: the chart redraws
            before = draws(page, 0)
            smaller = df.head(50)
            c.set_data(smaller)
            for k in ("_columnar", "_columns", "_data_version"):
                page.evaluate(f"window.__pyset(0, {json.dumps(k)}, {json.dumps(c.get_state()[k])})")
            page.wait_for_function(f"{chart_js(0)}.__draws > {before}", timeout=8000)
            assert len(data(page, 0)["series"][0]["points"]) == 50
        finally:
            browser.close()


def test_a_dataframe_too_large_to_hold_is_refused_by_name():
    big = pd.DataFrame({"x": np.arange(100_000), "y": np.arange(100_000)})
    with pytest.warns(LatticeGridWarning, match=r"chart\.windowed\.refused:scatter.*too large"):
        c = LatticeChart(big, type="scatter", x="x", y="y")
    assert "too large" in c._refused and c._columnar == {}


# --- live: Python changes the type / spec --------------------------------------------------
def test_real_browser_python_can_change_the_type_and_its_module_loads_on_demand():
    df = frame()
    g = LatticeGridWidget(df, offline=True)
    c = LatticeChart(g, type="bar", x="seg", y="v")
    assert sorted(c._chart_js) == ["charts"]
    with sync_playwright() as p:
        browser, page = open_page(p, [g, c])
        try:
            wait_chart(page)
            assert page.evaluate("[...window.__latticeChartModules.keys()]") == ["charts"]
            c.update(type="roc", label="churned", score="p")
            state = c.get_state()
            for k in ("type", "_spec", "_modules", "_chart_js", "_refused"):
                page.evaluate(f"window.__pyset(1, {json.dumps(k)}, {json.dumps(state[k])})")
            page.wait_for_selector("#root1 .lat-chartview__roc-auc", timeout=8000)
            assert page.evaluate("[...window.__latticeChartModules.keys()]") == ["charts", "chart-roc"]
        finally:
            browser.close()


# --- a windowed (large) grid ----------------------------------------------------------------
def big_frame(n=5000):
    rng = np.random.default_rng(7)
    return pd.DataFrame({"seg": rng.choice(SEGS, n), "v": rng.normal(10, 3, n), "w": rng.normal(0, 1, n)})


def test_a_windowed_grid_refuses_types_that_need_the_rows_by_name():
    g = LatticeGridWidget(big_frame(), windowed=True)
    with pytest.warns(LatticeGridWarning, match=r"chart\.windowed\.refused:scatter.*windowed"):
        c = LatticeChart(g, type="scatter", x="v", y="w")
    assert "windowed" in c._refused and c._windowed
    with pytest.warns(LatticeGridWarning, match=r"needs `x`"):
        LatticeChart(g, type="bar")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok = LatticeChart(g, type="bar", x="seg", y="v")  # grouped aggregates can draw this one
    assert ok._refused == "" and ok._windowed


def test_real_browser_windowed_chart_uses_the_grids_pushed_aggregates_not_the_rows():
    df = big_frame()
    g = LatticeGridWidget(df, windowed=True, offline=True)
    c = LatticeChart(g, type="bar", x="seg", y="v")
    with sync_playwright() as p:
        browser, page = open_page(p, [g, c])
        try:
            page.wait_for_selector("#root0 .lat-cell[role=gridcell]")
            wait_chart(page)
            pts = data(page)["series"][0]["points"]
            assert {q["x"]: q["y"] for q in pts} == pytest.approx(df.groupby("seg").v.sum().to_dict())
            sent = page.evaluate("window.__sent")
            assert any(m == "grouped" for _, m in sent), sent       # the engine's aggregate was asked
            # the windowed grid never held more than a window of rows
            assert page.evaluate("document.querySelector('#root0 div[style*=height]').__latticeGrid.rows.residentCount()") <= 400
            # a filter in the grid re-aggregates over the WHOLE frame, still without the rows
            before = draws(page)
            page.evaluate("document.querySelector('#root0 div[style*=height]').__latticeGrid.filters.quick('beta')")
            page.wait_for_function(f"{chart_js(1)}.__draws > {before}", timeout=10000)
            pts = data(page)["series"][0]["points"]
            assert [q["x"] for q in pts] == ["beta"]
            assert pts[0]["y"] == pytest.approx(df[df.seg == "beta"].v.sum())
        finally:
            browser.close()


def test_real_browser_windowed_refusal_draws_the_reason_and_asks_for_nothing():
    g = LatticeGridWidget(big_frame(), windowed=True, offline=True)
    with pytest.warns(LatticeGridWarning):
        c = LatticeChart(g, type="hexbin", x="v", y="w")
    with sync_playwright() as p:
        browser, page = open_page(p, [g, c])
        try:
            page.wait_for_selector("#root0 .lat-cell[role=gridcell]")
            page.wait_for_function("document.querySelector('#root1').innerText.includes('windowed')", timeout=8000)
            assert count(page, "svg") == 0
            assert not any(m == "grouped" for _, m in page.evaluate("window.__sent"))
        finally:
            browser.close()


# --- the docs run as written --------------------------------------------------------------
def test_every_python_block_in_docs_charts_runs_as_written():
    import pathlib
    import re
    text = (pathlib.Path(__file__).parent.parent / "docs" / "charts.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    ns: dict = {}
    for block in blocks:
        if block.lstrip().startswith("from ipywidgets import HBox\nfrom lattice_grid_jupyter") and "df" not in ns:
            continue  # the intro block names a user's own `df`
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exec(compile(block, "docs/charts.md", "exec"), ns)  # noqa: S102
    made = {k for k, v in ns.items() if isinstance(v, LatticeChart)}
    assert made == {"bar", "line", "pie", "scatter", "roc", "fan", "decomposition", "splom", "hexbin", "ridgeline"}
    assert all(ns[k]._refused == "" for k in made)
