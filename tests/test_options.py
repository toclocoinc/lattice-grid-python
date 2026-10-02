"""Card 1617: grid and per-column options from Python."""

import warnings

import pandas as pd
import pytest

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # noqa: E402

from _hosts import launch_browser  # noqa: E402
from _page import page_html, sets  # noqa: E402
from lattice_grid_jupyter import LatticeGridWidget, LatticeGridWarning  # noqa: E402


def _df():
    return pd.DataFrame({
        "name": ["Ada", "Grace", "Linus", "Ada"],
        "sales_total": [91, 88, 77, 50],
    })


# --- mapping ------------------------------------------------------------------
def test_snake_case_is_mapped_to_camel_case():
    w = LatticeGridWidget(_df(), options={"row_height": 30, "header_height": 40, "edit": True})
    assert w._grid_options == {"rowHeight": 30, "headerHeight": 40, "edit": True}


def test_camel_case_passes_through_and_mixed_nested_dicts_are_mapped():
    w = LatticeGridWidget(_df(), options={"rowHeight": 28, "tool_panel": {"panels": ["columns"]}})
    assert w._grid_options == {"rowHeight": 28, "toolPanel": {"panels": ["columns"]}}


def test_your_own_ids_are_never_rewritten():
    # `sales_total` is a column id, not an option; `context` is opaque user data
    w = LatticeGridWidget(_df(), options={"context": {"my_key": 1, "row_height": 2}})
    assert w._grid_options == {"context": {"my_key": 1, "row_height": 2}}
    w2 = LatticeGridWidget(_df(), columns=[{"field": "sales_total", "title": "Sales"}])
    assert {c["field"] for c in w2._columns} == {"name", "sales_total"}
    assert [c for c in w2._columns if c["field"] == "sales_total"][0]["title"] == "Sales"


def test_per_column_options_layer_over_the_built_columns():
    w = LatticeGridWidget(_df(), columns=[{"field": "sales_total", "width": 200, "title": "Sales"}])
    col = {c["field"]: c for c in w._columns}["sales_total"]
    assert col["width"] == 200 and col["type"] == "number" and col["edit"] is True
    w.set_data(_df())  # a data swap must not lose the per-column options
    assert {c["field"]: c for c in w._columns}["sales_total"]["width"] == 200


def test_functions_are_refused_not_dropped():
    with pytest.raises(TypeError, match="Options are data"):
        LatticeGridWidget(_df(), options={"row_class": lambda r: "x"})
    with pytest.raises(TypeError, match="Options are data"):
        LatticeGridWidget(_df(), columns=[{"field": "name", "compute": len}])


def test_python_owned_keys_are_ignored_with_a_warning():
    with pytest.warns(LatticeGridWarning, match="'rows'"):
        w = LatticeGridWidget(_df(), options={"rows": [1]})
    assert "rows" not in w._grid_options
    with pytest.warns(LatticeGridWarning, match="columns=\\[...\\]"):
        LatticeGridWidget(_df(), options={"columns": []})


def test_column_override_for_a_missing_field_warns_by_name():
    with pytest.warns(LatticeGridWarning, match="'nope'"):
        LatticeGridWidget(_df(), columns=[{"field": "nope", "width": 1}])


def test_convenience_flags():
    w = LatticeGridWidget(_df(), profile=True, histograms=True)
    assert w._grid_options["toolPanel"] == {"panels": ["columns", "statistics"]}
    assert w._grid_options["facets"] == {"enabled": True}
    # the user's own options win over a flag
    w = LatticeGridWidget(_df(), histograms=True, options={"facets": {"enabled": True, "height": 40}})
    assert w._grid_options["facets"] == {"enabled": True, "height": 40}
    # pivot by field name seeds the grid's pivot state; a dict is the pivot option
    assert LatticeGridWidget(_df(), pivot="name").state["pivot"] == {"enabled": True, "columns": ["name"]}
    assert LatticeGridWidget(_df(), pivot={"enabled": True})._grid_options["pivot"] == {"enabled": True}


# --- grid warnings -> warnings.warn --------------------------------------------
def test_grid_warning_ids_become_python_warnings_naming_the_option():
    w = LatticeGridWidget(_df())
    with pytest.warns(LatticeGridWarning, match=r"config\.unknown:row_hieght.*row_hieght"):
        w._grid_warnings = [{"id": "config.unknown:row_hieght",
                             "message": "'row_hieght' is not a configuration key this grid recognises"}]


# --- live update ---------------------------------------------------------------
def test_options_update_is_live_and_does_not_resend_the_data():
    w = LatticeGridWidget(_df(), options={"row_height": 30})
    sent = []
    real = w.send_state
    w.send_state = lambda key=None: (sent.append(key), real(key))[1]
    version = w._data_version

    w.options = {"row_height": 44, "stripe": True}

    flat = {k for entry in sent for k in ([entry] if isinstance(entry, str) else (entry or []))}
    assert flat == {"_grid_options"}, sent       # the options trait only
    assert "_columnar" not in flat and "_columns" not in flat and "_data_version" not in flat
    assert w._data_version == version
    assert w._grid_options["rowHeight"] == 44 and w._grid_options["stripe"] is True


# --- real browser ----------------------------------------------------------------
def _open(p, w, width=1100):
    browser = launch_browser(p)
    if browser is None:
        pytest.skip("no real browser available to launch")
    page = browser.new_page(viewport={"width": width, "height": 600})
    page.set_content(page_html(w))
    page.wait_for_function("window.__rendered === true || window.__error", timeout=30000)
    assert page.evaluate("window.__error") is None
    page.wait_for_selector(".lat-cell[role=gridcell]", timeout=10000)
    return browser, page


def test_real_browser_unknown_and_mistyped_options_reach_python_as_warnings():
    w = LatticeGridWidget(_df(), offline=True,
                          options={"row_hieght": 30, "row_height": "tall"})
    with sync_playwright() as p:
        browser, page = _open(p, w)
        try:
            page.wait_for_function("window.__sets.some(s => s[0]==='_grid_warnings')", timeout=4000)
            got = [x for batch in sets(page, "_grid_warnings") for x in batch]
        finally:
            browser.close()
    ids = {g["id"] for g in got}
    assert {"config.unknown:row_hieght", "config.value:rowHeight"} <= ids, ids
    with pytest.warns(LatticeGridWarning) as rec:
        w._grid_warnings = got
    text = " ".join(str(r.message) for r in rec)
    assert "row_hieght" in text and "rowHeight" in text and "[lattice]" in text


def test_real_browser_live_update_changes_the_grid_without_a_data_message():
    w = LatticeGridWidget(_df(), offline=True)
    with sync_playwright() as p:
        browser, page = _open(p, w)
        try:
            def row_h():
                return page.evaluate("document.querySelector('.lat-cell[role=gridcell]').getBoundingClientRect().height")
            before_h = row_h()
            before_msgs = len(sets(page))
            w.options = {"row_height": 60}
            page.evaluate("window.__pyset('_grid_options', %s)" % __import__("json").dumps(w._grid_options))
            page.wait_for_function(
                "document.querySelector('.lat-cell[role=gridcell]').getBoundingClientRect().height > %s" % before_h,
                timeout=4000)
            after_h = row_h()
            new_msgs = [k for k, _ in sets(page)[before_msgs:]]
            assert "_columnar" not in new_msgs and "_data_version" not in new_msgs
            assert page.locator(".lat-cell[role=gridcell]", has_text="Grace").count() == 1  # data intact
        finally:
            browser.close()
    assert after_h > before_h


def test_real_browser_histograms_render():
    w = LatticeGridWidget(_df(), offline=True, histograms=True)
    with sync_playwright() as p:
        browser, page = _open(p, w)
        try:
            page.wait_for_selector(".lat-facet .lat-facet__bar", timeout=6000)
            n = page.locator(".lat-facet").count()
        finally:
            browser.close()
    assert n >= 2  # one per column header


def test_real_browser_profile_panel_renders_the_statistics():
    w = LatticeGridWidget(_df(), offline=True, profile=True)
    with sync_playwright() as p:
        browser, page = _open(p, w)
        try:
            page.get_by_text("Statistics", exact=True).click()
            page.wait_for_selector(".lat-panel__stats", timeout=6000)
            page.locator(".lat-cell[role=gridcell]", has_text="91").first.click()
            page.wait_for_function(
                "document.querySelector('.lat-panel__body').innerText.includes('Present')", timeout=4000)
            text = page.evaluate("document.querySelector('.lat-panel__body').innerText")
        finally:
            browser.close()
    assert "Present" in text and "Distinct" in text


def test_real_browser_pivot_flag_pivots_the_grid():
    df = pd.DataFrame({"name": ["Ada", "Grace", "Linus", "Ada"], "region": list("nsns"),
                       "score": [91, 88, 77, 50]})
    w = LatticeGridWidget(df, offline=True, pivot="region",
                          columns=[{"field": "name", "group": True}, {"field": "score", "total": "sum"}])
    with sync_playwright() as p:
        browser, page = _open(p, w)
        try:
            heads = page.evaluate("[...document.querySelectorAll('[role=columnheader]')].map(e => e.innerText)")
        finally:
            browser.close()
    assert "n" in heads and "s" in heads  # one column per distinct region
