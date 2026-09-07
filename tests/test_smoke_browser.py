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
