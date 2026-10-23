"""BACKLOG-0004774 part 5: the customer path -- reference notebooks run top to bottom under Run All.

Three small notebooks in ``examples/customer-path/`` are run the way a customer runs them:

* ``small.ipynb``  -- a 200-row frame (small mode): render, sort, filter, then a later cell reads ``w.view``;
* ``large.ipynb``  -- a 200,000-row frame (large mode): render, group, then a later cell reads the view's
  row count and the group aggregates;
* ``router.ipynb`` -- a DuckDB relation: render, select rows, read ``w.selected``, then ``set_data`` with a
  new schema (one column added, one removed) and the grid shows the new columns with no stale sort or filter.

Two drivers, because ``nbclient`` alone has no front end:

* ``nbclient`` (kernel only, no browser) runs ``small`` and ``large`` -- the headless Run All that
  BACKLOG-0004772 broke (``w.view`` empty because the browser never got a turn);
* the repo's real-browser host harness (``tests/_hosts.py``: a real JupyterLab 4 in headless Chrome) runs all
  three with the menu's *Run All*. Selecting rows and painting the new columns need the grid's front end,
  so ``router.ipynb`` cannot run any other way. Each cell's printed ``KEY=value`` lines are read from the
  notebook's own outputs; a failed assertion in a cell is a failed test.
"""

import json
import pathlib
import re
import shutil
import textwrap
import time

import pytest

pytest.importorskip("duckdb")
nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # noqa: E402

from _hosts import JupyterServer, launch_browser  # noqa: E402

NOTEBOOKS = pathlib.Path(__file__).resolve().parent.parent / "examples" / "customer-path"
_KV = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$", re.M)


def _printed(outputs) -> dict:
    text = "".join(o.get("text", "") for o in outputs if o.get("output_type") == "stream")
    return {k: v for k, v in _KV.findall(text)}


def _expect(got: dict, want: dict) -> None:
    for k, v in want.items():
        assert got.get(k) == v, f"{k}: want {v!r}, notebook printed {got.get(k)!r} (all: {got})"


# What each notebook's cells print (the values are computed independently here).
SMALL = {"MODE": "client", "SMALL_ROWS": "50", "SMALL_TOP": "[196, 192, 188]", "SMALL_FIRST_ID": "108"}
LARGE = {"MODE": "windowed", "LARGE_ROWS": "100000", "LARGE_GROUPS": "4", "LARGE_T0_COUNT": "24000",
         "LARGE_T0_TOTAL": "1776000"}
ROUTER_BEFORE = {"MODE": "windowed", "ENGINE": "duckdb", "BEFORE_COLUMNS": "['id', 'who', 'day', 'amount']"}
ROUTER_AFTER = {"AFTER_COLUMNS": "['id', 'who', 'day', 'score']", "AFTER_ROWS": "300",
                "AFTER_SORT": "[] FILTERS=None"}


def _run_all_headless(name: str):
    nb = nbformat.read(str(NOTEBOOKS / f"{name}.ipynb"), 4)
    nbclient.NotebookClient(nb, kernel_name="python3", timeout=120).execute()
    return _printed([o for c in nb.cells for o in c.outputs])


def test_small_notebook_run_all_headless():
    _expect(_run_all_headless("small"), SMALL)


def test_large_notebook_run_all_headless():
    _expect(_run_all_headless("large"), LARGE)


# --- the real front end: JupyterLab 4 in headless Chrome ---------------------------------------------------
_CELLS_JS = "JSON.stringify(window.jupyterapp.shell.currentWidget.content.model.toJSON().cells)"
_IDLE_JS = ("(() => { const w = window.jupyterapp.shell.currentWidget; const k = w && w.sessionContext && "
            "w.sessionContext.session && w.sessionContext.session.kernel; return !!k && k.status === 'idle'; })()")


@pytest.fixture(scope="module")
def lab():
    pytest.importorskip("jupyterlab")
    server = JupyterServer("lab", [])
    for p in NOTEBOOKS.glob("*.ipynb"):
        shutil.copy(p, server.dir / p.name)
    with sync_playwright() as p:
        browser = launch_browser(p)
        if browser is None:
            server.close()
            pytest.skip("no real browser available to launch")
        try:
            yield server, browser
        finally:
            browser.close()
            server.close()


def _run_all_in_lab(lab, name: str, timeout_s: float = 90):
    """Open the notebook, click Run All (``notebook:run-all-cells``), wait for the last cell, return
    ``(page, cells)`` with the notebook's own outputs. The page stays open for DOM checks."""
    server, browser = lab
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.add_locator_handler(
        page.locator("dialog.jp-Dialog"),
        lambda d: d.get_by_role("button", name=re.compile("Cancel|Dismiss|OK")).first.click())
    page.goto(f"http://127.0.0.1:{server.port}/lab/tree/{name}.ipynb?token={server.token}&reset")
    page.wait_for_function("window.jupyterapp && window.jupyterapp.commands", timeout=60000)
    page.wait_for_selector(".jp-Notebook .jp-Cell", timeout=60000)
    page.wait_for_function(_IDLE_JS, timeout=120000)
    page.evaluate("window.jupyterapp.commands.execute('notebook:run-all-cells')")
    deadline = time.time() + timeout_s
    cells = []
    while time.time() < deadline:
        cells = json.loads(page.evaluate(_CELLS_JS))
        errored = any(o.get("output_type") == "error" for c in cells for o in c.get("outputs", []))
        done = all(c.get("execution_count") for c in cells) and page.evaluate(_IDLE_JS)
        if errored or done:
            break
        page.wait_for_timeout(100)
    errors = [o for c in cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, "a cell failed under Run All: " + "\n".join(
        f"{e['ename']}: {e['evalue']}" for e in errors)
    assert all(c.get("execution_count") for c in cells), "Run All did not reach the last cell"
    return page, cells


def _outputs(cells):
    return [o for c in cells for o in c.get("outputs", [])]


def _header_cells(page) -> list:
    """``[text, aria-sort]`` of each column header the grid is showing."""
    return page.evaluate("[...document.querySelectorAll('.lat-header-cell')]"
                         ".map(h => [h.innerText.trim(), h.getAttribute('aria-sort')])")


def _read_again(page, index: int, timeout_s: float = 30) -> dict:
    """The widget's documented pattern for what only the browser reports: run the reading cell again, later."""
    before = json.loads(page.evaluate(_CELLS_JS))[index]["execution_count"]
    page.locator(".jp-Notebook .jp-Cell").nth(index).locator(".jp-InputArea-editor").click()
    page.evaluate("window.jupyterapp.commands.execute('notebook:run-cell')")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        cell = json.loads(page.evaluate(_CELLS_JS))[index]
        if cell["execution_count"] != before and page.evaluate(_IDLE_JS):
            return _printed(cell["outputs"])
        page.wait_for_timeout(100)
    raise AssertionError("the reading cell did not run again")


def test_small_notebook_run_all_in_lab(lab):
    page, cells = _run_all_in_lab(lab, "small")
    try:
        _expect(_printed(_outputs(cells)), SMALL)
        # a list cell reaches the grid as an array ("3,4,0"), not as Python's text for a list ("[3, 4, 0]")
        page.wait_for_selector(".lat-cell[data-col=trend]", timeout=30000)
        texts = page.evaluate("[...document.querySelectorAll('.lat-cell[data-col=trend]')].map(c => c.innerText.trim())")
        assert texts and all(re.fullmatch(r"\d(,\d)*", t) for t in texts), texts
    finally:
        page.close()


def test_large_notebook_run_all_in_lab(lab):
    page, cells = _run_all_in_lab(lab, "large")
    try:
        _expect(_printed(_outputs(cells)), LARGE)
        page.wait_for_selector(".lat-cell[role=gridcell], .lat-group", timeout=30000)
    finally:
        page.close()


def test_router_notebook_run_all_in_lab(lab):
    page, cells = _run_all_in_lab(lab, "router")
    try:
        _expect(_printed(_outputs(cells)), {**ROUTER_BEFORE, **ROUTER_AFTER})
        # the front end reported the two rows selected with the grid's own selection API; the cell that
        # reads them is the notebook's last one, read again now (the grid reports between cells)
        _expect(_read_again(page, len(cells) - 1), {"SELECTED": "[3, 5]"})
        page.wait_for_selector(".lat-header-cell", timeout=30000)
        headers = _header_cells(page)
        assert [h[0] for h in headers] == ["id", "who", "day", "score"], headers
        assert all(h[1] in (None, "none") for h in headers), f"a stale sort is still drawn: {headers}"
    finally:
        page.close()
