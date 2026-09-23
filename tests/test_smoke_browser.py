"""
REAL-BROWSER smoke test -- the one gap the headless Python/jsdom tests cannot cover.

Renders the widget's actual front end (widget.js + the vendored grid bundle) in a
real Chromium via Playwright, performs a REAL manual cell edit (double-click the
cell, type, Enter), and asserts:

  1. the grid actually rendered (cells in the DOM);
  2. the licence resolved to the free "localhost" state with no key;
  3. the manual edit fired cell:changed and produced the exact {key,colId,value}
     payload the widget's comm carries; and
  4. feeding that payload into the real Python widget updates the DataFrame,
     dtype-preserved -- i.e. the full browser-edit -> DataFrame loop.

If no real browser can be launched, the test is SKIPPED (never faked).
"""

import json
import pathlib

import pandas as pd
import pytest

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # noqa: E402

import lattice_grid_jupyter  # noqa: E402
from lattice_grid_jupyter import LatticeGridWidget  # noqa: E402

_STATIC = pathlib.Path(lattice_grid_jupyter.__file__).parent / "static"


def _launch(p):
    """Launch the first browser that works; return None if none can start."""
    for channel in ("chrome", "chromium", "msedge"):
        try:
            return p.chromium.launch(channel=channel, headless=True, args=["--no-sandbox"])
        except Exception:
            continue
    try:
        return p.chromium.launch(headless=True, args=["--no-sandbox"])  # bundled
    except Exception:
        return None


def _page_html(widget: LatticeGridWidget) -> str:
    widget_js = (_STATIC / "widget.js").read_text().replace("export default", "window.__widget =")
    state = {
        k: getattr(widget, k)
        for k in (
            "_columns", "_columnar", "_row_key", "licence", "height",
            "_grid_source", "_grid_js", "_grid_css", "_grid_version",
        )
    }
    state.update({"_data_version": 0, "_row_op": None, "_edit": None, "_licence_state": ""})
    tmpl = """<!doctype html><html><head><meta charset=utf-8></head><body>
<div id=root></div>
<script>
const state = __STATE__; const L = {};
window.__model = {
  get: k => state[k],
  set: (k, v) => { state[k] = v; (L['change:'+k]||[]).forEach(f=>f()); },
  save_changes: () => {},
  on: (e, cb) => { (L[e]=L[e]||[]).push(cb); },
  off: (e, cb) => { L[e] = (L[e]||[]).filter(f=>f!==cb); },
};
</script>
<script type=module>
__WIDGET__
;(async () => {
  await window.__widget.render({ model: window.__model, el: document.getElementById('root') });
  window.__rendered = true;
})();
</script>
</body></html>"""
    return tmpl.replace("__STATE__", json.dumps(state)).replace("__WIDGET__", widget_js)


def test_real_browser_manual_edit_round_trip():
    df = pd.DataFrame({
        "name": ["Ada", "Grace", "Linus"],
        "score": [91, 88, 77],
        "active": [True, False, True],
    })
    w = LatticeGridWidget(df, offline=True)  # vendored -> no network needed

    with sync_playwright() as p:
        browser = _launch(p)
        if browser is None:
            pytest.skip("no real browser available to launch")
        page = browser.new_page()
        try:
            page.set_content(_page_html(w))
            page.wait_for_function("window.__rendered === true", timeout=20000)

            # 1. grid rendered
            page.wait_for_selector(".lat-cell[role=gridcell]", timeout=8000)
            assert page.locator(".lat-cell[role=gridcell]").count() > 0

            # 2. free localhost licence, no key
            assert page.evaluate("window.__model.get('_licence_state')") == "localhost"

            # 3. REAL manual edit: dblclick the score=91 cell, type 99, Enter
            cell = page.locator(".lat-cell[role=gridcell]", has_text="91").first
            cell.dblclick()
            page.wait_for_selector("input.lat-editor__input", timeout=4000)
            page.keyboard.press("Control+A")
            page.keyboard.type("99")
            page.keyboard.press("Enter")

            page.wait_for_function("window.__model.get('_edit') !== null", timeout=4000)
            payload = page.evaluate("window.__model.get('_edit')")
        finally:
            browser.close()

    assert payload["key"] == "0"
    assert payload["colId"] == "score"
    assert payload["value"] == 99
    assert payload["old"] == 91

    # 4. the exact browser payload, pushed through the real widget, hits the df
    w._edit = {"key": payload["key"], "colId": payload["colId"], "value": payload["value"]}
    assert w.df.loc[0, "score"] == 99
    assert pd.api.types.is_integer_dtype(w.df["score"].dtype)


def test_real_browser_datetime_edit_round_trip():
    """BACKLOG-0001425, REAL browser: editing a ``timestamp`` cell must not
    write 1970.

    The grid's ``timestamp`` column type stores an epoch-MILLISECOND Number
    and its date/time editor emits exactly that Number on ``cell:changed`` --
    confirmed here by reading the actual payload, not assumed. Before the fix,
    ``cast_to_dtype`` ran that Number through ``pd.to_datetime`` with pandas'
    default nanosecond unit, landing on 1970 instead of the edited instant.

    The browser context timezone is pinned to UTC so the grid's date/time
    editor (which reads/writes wall-clock text in the browser's own zone
    when no column ``timeZone`` is configured) agrees with the UTC-declared
    naive column under this wrapper's contract -- see the pandas README.
    """
    df = pd.DataFrame({"when": pd.to_datetime(["2021-06-01T12:00:00"])})
    w = LatticeGridWidget(df, offline=True)

    with sync_playwright() as p:
        browser = _launch(p)
        if browser is None:
            pytest.skip("no real browser available to launch")
        context = browser.new_context(timezone_id="UTC")
        page = context.new_page()
        try:
            page.set_content(_page_html(w))
            page.wait_for_function("window.__rendered === true", timeout=20000)
            page.wait_for_selector(".lat-cell[role=gridcell]", timeout=8000)

            # REAL manual edit: focus the one cell, Enter opens the date/time
            # editor (a date input + a time input), retype both, Enter commits.
            page.locator(".lat-cell[role=gridcell]").first.click()
            page.keyboard.press("Enter")
            page.wait_for_selector("input[type=date]", timeout=4000)
            page.locator("input[type=date]").fill("2022-03-04")
            page.locator("input[type=time]").fill("15:30")
            page.keyboard.press("Enter")

            page.wait_for_function("window.__model.get('_edit') !== null", timeout=4000)
            payload = page.evaluate("window.__model.get('_edit')")
        finally:
            browser.close()

    # The grid's actual wire payload: a Number, epoch milliseconds.
    assert payload["colId"] == "when"
    assert isinstance(payload["value"], (int, float))
    expected_ms = pd.Timestamp("2022-03-04T15:30:00", tz="UTC").value // 1_000_000
    assert payload["value"] == expected_ms, f"{payload['value']} != {expected_ms}"

    # The exact browser payload, pushed through the real widget, must land on
    # the edited instant -- NOT 1970-01-01 (the pre-fix defect).
    w._edit = {"key": payload["key"], "colId": payload["colId"], "value": payload["value"]}
    assert w.df.loc[0, "when"] == pd.Timestamp("2022-03-04T15:30:00")
    assert w.df.loc[0, "when"].year == 2022  # the direct anti-regression check


def test_large_dataframe_renders_virtualized():
    """A 100k-row frame renders in a real browser without freezing (virtualized)."""
    import numpy as np
    n = 100_000
    df = pd.DataFrame({
        "id": np.arange(n),
        "val": np.random.rand(n).round(4),
        "flag": np.random.rand(n) > 0.5,
        "label": [f"r{i}" for i in range(n)],
    })
    w = LatticeGridWidget(df, offline=True)

    with sync_playwright() as p:
        browser = _launch(p)
        if browser is None:
            pytest.skip("no real browser available to launch")
        page = browser.new_page()
        try:
            page.set_content(_page_html(w), timeout=60000)
            page.wait_for_function("window.__rendered === true", timeout=60000)
            page.wait_for_selector(".lat-cell[role=gridcell]", timeout=15000)
            dom_cells = page.locator(".lat-cell[role=gridcell]").count()
        finally:
            browser.close()

    # Rendering virtualizes: only the visible window is in the DOM, not 100k rows.
    assert 0 < dom_cells < 5000, f"{dom_cells} DOM cells (virtualization not working?)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
