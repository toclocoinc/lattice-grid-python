"""Card 1618: the 10M-row measurement, in real JupyterLab with a real kernel comm.

A 10,000,000 x 20 mixed frame (6 int, 6 float, 3 categorical, 2 text,
2 datetime, 1 bool) is built in the kernel, then the widget cell runs. Measured:

* **first paint** -- the widget cell is executed -> the first row's cells carry
  data on screen (budget 2 s);
* **scroll window** -- the grid's body is scrolled to a far, never-fetched
  position -> that row's data is on screen (budget 150 ms), twelve times at
  spread positions; the median is held to the budget and every sample is
  printed and written to ``LATTICE_LARGE_REPORT`` when set.

Needs ~6 GB of free memory and about a minute. Skipped by name when JupyterLab
or a browser is missing, never faked.
"""

import importlib.util
import json
import os
import re
import statistics
import time

import pytest

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # noqa: E402

from _hosts import JupyterServer, launch_browser  # noqa: E402

N = int(os.environ.get("LATTICE_LARGE_ROWS", "10000000"))
FIRST_PAINT_BUDGET_S = 2.0
SCROLL_BUDGET_MS = 150.0

BUILD = f'''
import numpy as np, pandas as pd, time
_t = time.perf_counter()
r = np.random.default_rng(0)
n = {N}
cols = {{}}
for i in range(6): cols[f"i{{i}}"] = r.integers(0, 1_000_000, n)
for i in range(6): cols[f"f{{i}}"] = r.random(n) * 1000
cats = np.array(["north", "south", "east", "west", "centre"])
for i in range(3): cols[f"c{{i}}"] = pd.Categorical.from_codes(r.integers(0, 5, n), cats)
ids = np.array([f"id-{{k:05d}}" for k in range(100000)], dtype=object)
for i in range(2): cols[f"s{{i}}"] = ids[r.integers(0, 100000, n)]
for i in range(2): cols[f"d{{i}}"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(r.integers(0, 10**8, n), unit="s")
cols["b0"] = r.random(n) > 0.5
df = pd.DataFrame(cols)
print("BUILT %d x %d in %.1fs" % (df.shape[0], df.shape[1], time.perf_counter() - _t))
'''
SHOW = '''
_t = time.perf_counter()
from lattice_grid_jupyter import LatticeGridWidget
w = LatticeGridWidget(df, offline=True, height=500)
print("WIDGET_PY_MS=%.0f windowed=%s" % ((time.perf_counter() - _t) * 1000, w.windowed))
w
'''

GRID = "[...document.querySelectorAll('div')].find(d => d.__latticeGrid)"

# A far row's data on screen: scroll the body, then wait (in the page, per
# animation frame) until that row is loaded and painted.
SCROLL_JS = """async (target) => {
  const host = %s; const g = host.__latticeGrid;
  const vp = host.querySelector('.lat-viewport') || host.querySelector('[class*=viewport]')
    || [...host.querySelectorAll('div')].find(d => d.scrollHeight > d.clientHeight + 1000);
  const rowH = vp.scrollHeight / g.rows.count();
  const t0 = performance.now();
  vp.scrollTop = Math.round(target * rowH);
  return await new Promise((resolve) => {
    const tick = () => {
      const r = g.rows.get(target);
      const painted = r && r.data && [...host.querySelectorAll('.lat-cell[role=gridcell]')]
        .some(c => c.textContent.includes(String(r.data.s0)));
      if (painted) resolve(performance.now() - t0);
      else if (performance.now() - t0 > 20000) resolve(-1);
      else requestAnimationFrame(tick);
    };
    tick();
  });
}""" % GRID


def test_10m_rows_first_paint_and_scroll_window_in_jupyterlab():
    if importlib.util.find_spec("jupyterlab") is None:
        pytest.skip("jupyterlab not installed")
    server = JupyterServer("lab", [BUILD, SHOW])
    report = {"rows": N, "columns": 20}
    try:
        with sync_playwright() as p:
            browser = launch_browser(p)
            if browser is None:
                pytest.skip("no real browser available to launch")
            try:
                page = browser.new_page(viewport={"width": 1400, "height": 1000})
                page.add_locator_handler(
                    page.locator("dialog.jp-Dialog"),
                    lambda d: d.get_by_role("button", name=re.compile("Cancel|Dismiss|OK")).first.click())
                page.goto(server.url)
                page.wait_for_function("window.jupyterapp && window.jupyterapp.commands", timeout=60000)
                page.wait_for_selector(".jp-Notebook .jp-Cell", timeout=60000)
                page.locator(".jp-Notebook .jp-Cell").first.click()
                page.evaluate("window.jupyterapp.commands.execute('notebook:run-cell-and-select-next')")
                page.wait_for_function("document.body.innerText.includes('BUILT')", timeout=300000)
                report["build"] = re.search(r"BUILT [^\n]*", page.evaluate("document.body.innerText")).group(0)

                t0 = time.perf_counter()
                page.evaluate("window.jupyterapp.commands.execute('notebook:run-cell-and-select-next')")
                page.wait_for_function(
                    "(() => { const h = %s; if (!h) return false; const r = h.__latticeGrid.rows.get(0);"
                    " return !!(r && r.data) && [...h.querySelectorAll('.lat-cell[role=gridcell]')]"
                    ".some(c => c.textContent.includes(String(r.data.s0))); })()" % GRID,
                    polling="raf", timeout=60000)
                report["first_paint_ms"] = round((time.perf_counter() - t0) * 1000)
                text = page.evaluate("document.body.innerText")
                m = re.search(r"WIDGET_PY_MS=(\d+) windowed=(\w+)", text)
                report["widget_python_ms"] = int(m.group(1)) if m else None
                assert m and m.group(2) == "True", text[-400:]
                report["total_rows_shown"] = page.evaluate(f"{GRID}.__latticeGrid.rows.count()")

                samples = []
                for k in range(12):
                    target = int(N * (0.04 + 0.08 * k)) % (N - 50)
                    samples.append(round(page.evaluate(SCROLL_JS, target), 1))
                report["scroll_ms"] = samples
                report["scroll_ms_median"] = statistics.median(samples)
                report["scroll_ms_max"] = max(samples)
            finally:
                browser.close()
    finally:
        server.close()
    print("LARGE_10M_REPORT " + json.dumps(report))
    out = os.environ.get("LATTICE_LARGE_REPORT")
    if out:
        with open(out, "w") as fh:
            json.dump(report, fh, indent=2)
    assert report["total_rows_shown"] == N
    assert all(s >= 0 for s in report["scroll_ms"]), report
    assert report["first_paint_ms"] < FIRST_PAINT_BUDGET_S * 1000, report
    assert report["scroll_ms_median"] < SCROLL_BUDGET_MS, report
