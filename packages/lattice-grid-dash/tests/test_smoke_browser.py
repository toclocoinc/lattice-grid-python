"""REAL-BROWSER smoke tests for lattice-grid-dash (Playwright + system Chrome).

Two independent proofs, strongest first:

1. ``test_dash_server_manual_edit_reaches_python_callback`` -- boots a real Dash
   app (the component + a Python ``@callback`` on ``cellChanged``) in a background
   thread, drives it in system Chrome, performs a REAL manual cell edit
   (double-click, type, Enter), and asserts the Dash callback ran server-side with
   the edited value (its output text updates). This proves the full Dash
   round-trip: grid edit -> setProps -> Dash store -> Python callback.

2. ``test_bundle_renders_and_edit_fires_adapter_callback`` -- the lighter fallback
   called out in the task: renders the *built component bundle* in real Chrome
   (with Dash's React), performs a manual edit, and asserts the edit fires the
   adapter callback into the component's ``setProps`` (captured), i.e. the grid
   renders and edits surface -- no Dash server needed.

If no real browser can launch, tests SKIP (never faked). Chrome is driven via
channel='chrome' first, exactly like phase C1.
"""

import json
import pathlib
import socket
import threading
import time
import urllib.request

import pandas as pd
import pytest

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # noqa: E402

import lattice_grid_dash  # noqa: E402
from lattice_grid_dash import dataframe_to_data  # noqa: E402

_PKG = pathlib.Path(lattice_grid_dash.__file__).parent
_BUNDLE = _PKG / "lattice_grid_dash.min.js"
_CSS = _PKG / "lattice-grid.min.css"


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


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _react_umd() -> str:
    """Locate React/ReactDOM UMD text that Dash ships, for the bundle-only test."""
    import dash

    deps = pathlib.Path(dash.__file__).parent / "deps"
    react = sorted(deps.glob("react@18*.min.js"))
    reactdom = sorted(deps.glob("react-dom@18*.min.js"))
    if not react or not reactdom:
        return ""
    # exclude the profiling/development builds; prefer the plain min.js
    r = [f for f in react if "profiling" not in f.name][0]
    rd = [f for f in reactdom if "profiling" not in f.name][0]
    return r.read_text() + "\n" + rd.read_text()


# --------------------------------------------------------------------------- #
# 1. Full Dash server round-trip
# --------------------------------------------------------------------------- #
def test_dash_server_manual_edit_reaches_python_callback():
    from dash import Dash, Input, Output, html

    df = pd.DataFrame(
        {"name": ["Ada", "Grace", "Linus"], "score": [91, 88, 77], "active": [True, False, True]}
    )
    port = _free_port()
    app = Dash(__name__)
    app.layout = html.Div(
        [
            lattice_grid_dash.LatticeGrid(id="grid", data=dataframe_to_data(df)),
            html.Div(id="out", children="none"),
            html.Div(id="lic", children="none"),
        ]
    )

    @app.callback(Output("out", "children"), Input("grid", "cellChanged"))
    def _on_edit(edit):
        if not edit:
            return "none"
        return f"{edit['colId']}={edit['value']}@{edit['key']}"

    @app.callback(Output("lic", "children"), Input("grid", "licenceState"))
    def _on_lic(state):
        return state or "none"

    server_error = {}

    def _serve():
        try:
            app.run(port=port, debug=False, use_reloader=False)
        except Exception as e:  # noqa: BLE001
            server_error["e"] = e

    threading.Thread(target=_serve, daemon=True).start()

    base = f"http://127.0.0.1:{port}"
    ready = False
    for _ in range(100):
        if server_error:
            pytest.skip(f"dash server failed to start: {server_error['e']}")
        try:
            urllib.request.urlopen(base, timeout=0.5)
            ready = True
            break
        except Exception:
            time.sleep(0.1)
    if not ready:
        pytest.skip("dash server did not become ready")

    with sync_playwright() as p:
        browser = _launch(p)
        if browser is None:
            pytest.skip("no real browser available to launch")
        page = browser.new_page()
        try:
            page.goto(base, wait_until="networkidle")
            page.wait_for_selector(".lat-cell[role=gridcell]", timeout=20000)

            # REAL manual edit: dblclick the score=91 cell, type 99, Enter.
            cell = page.locator(".lat-cell[role=gridcell]", has_text="91").first
            cell.dblclick()
            page.wait_for_selector("input.lat-editor__input", timeout=5000)
            page.keyboard.press("Control+A")
            page.keyboard.type("99")
            page.keyboard.press("Enter")

            # The Python @callback only updates #out if cellChanged reached it.
            page.wait_for_function(
                "document.getElementById('out').textContent.indexOf('99') !== -1",
                timeout=10000,
            )
            out_text = page.text_content("#out")
            lic_text = page.text_content("#lic")
        finally:
            browser.close()

    assert out_text == "score=99@0", out_text
    # licence resolved client-side to the free localhost state (127.0.0.1).
    assert lic_text == "localhost", lic_text


# --------------------------------------------------------------------------- #
# 2. Bundle-only fallback (no Dash server)
# --------------------------------------------------------------------------- #
def test_bundle_renders_and_edit_fires_adapter_callback():
    if not _BUNDLE.exists():
        pytest.skip("component bundle not built (run: node build.mjs)")
    react = _react_umd()
    if not react:
        pytest.skip("could not locate React UMD from the installed dash package")

    df = pd.DataFrame({"name": ["Ada", "Grace"], "score": [91, 88]})
    data = dataframe_to_data(df)

    html_doc = """<!doctype html><html><head><meta charset=utf-8>
<style>__CSS__</style></head><body>
<div id=root></div>
<script>__REACT__</script>
<script>__BUNDLE__</script>
<script>
  window.__edits = [];
  const props = {
    id: 'grid',
    data: __DATA__,
    setProps: (u) => { window.__edits.push(u); Object.assign(props, u); },
  };
  const Comp = window.lattice_grid_dash.LatticeGrid;
  const root = window.ReactDOM.createRoot(document.getElementById('root'));
  root.render(window.React.createElement(Comp, props));
  window.__rendered = true;
</script>
</body></html>"""
    page_html = (
        html_doc.replace("__CSS__", _CSS.read_text() if _CSS.exists() else "")
        .replace("__REACT__", react)
        .replace("__BUNDLE__", _BUNDLE.read_text())
        .replace("__DATA__", json.dumps(data))
    )

    with sync_playwright() as p:
        browser = _launch(p)
        if browser is None:
            pytest.skip("no real browser available to launch")
        page = browser.new_page()
        try:
            page.set_content(page_html, timeout=30000)
            page.wait_for_function("window.__rendered === true", timeout=20000)
            page.wait_for_selector(".lat-cell[role=gridcell]", timeout=15000)
            assert page.locator(".lat-cell[role=gridcell]").count() > 0

            cell = page.locator(".lat-cell[role=gridcell]", has_text="91").first
            cell.dblclick()
            page.wait_for_selector("input.lat-editor__input", timeout=5000)
            page.keyboard.press("Control+A")
            page.keyboard.type("99")
            page.keyboard.press("Enter")

            page.wait_for_function(
                "window.__edits.some(u => u.cellChanged)", timeout=8000
            )
            edits = page.evaluate("window.__edits")
        finally:
            browser.close()

    changes = [u["cellChanged"] for u in edits if "cellChanged" in u]
    assert changes, f"no cellChanged captured; updates were {edits}"
    last = changes[-1]
    assert last["key"] == "0"
    assert last["colId"] == "score"
    assert last["value"] == 99
    assert last["old"] == 91


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
