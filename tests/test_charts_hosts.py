"""HOST PROOF (card 1619): a grid and a chart, laid out with ipywidgets ``HBox``,
render side by side in real JupyterLab 4 and the chart follows the grid's filter.

The widget front ends, the vendored bundles and the comm are the real ones; only the
browser driving is scripted. Skipped by name when Lab or a browser is absent."""

import importlib.util
import re

import pytest

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # noqa: E402

from _hosts import JupyterServer, launch_browser  # noqa: E402

CELLS = [
    "import numpy as np, pandas as pd\n"
    "from ipywidgets import HBox\n"
    "from lattice_grid_jupyter import LatticeGridWidget, LatticeChart\n"
    "rng = np.random.default_rng(3)\n"
    "df = pd.DataFrame({'seg': rng.choice(['alpha', 'beta', 'gamma'], 90), 'v': rng.normal(10, 3, 90)})\n"
    "g = LatticeGridWidget(df, offline=True, height=300)\n"
    "c = LatticeChart(g, type='bar', x='seg', y='v', height=300)\n"
    "HBox([g, c])",
]


@pytest.mark.skipif(importlib.util.find_spec("jupyterlab") is None, reason="jupyterlab not installed")
def test_real_jupyterlab_hbox_lays_out_grid_and_chart_and_the_chart_follows_a_filter():
    server = JupyterServer("lab", CELLS)
    try:
        with sync_playwright() as p:
            browser = launch_browser(p)
            if browser is None:
                pytest.skip("no real browser available to launch")
            try:
                page = browser.new_page(viewport={"width": 1500, "height": 900})
                page.add_locator_handler(
                    page.locator("dialog.jp-Dialog"),
                    lambda d: d.get_by_role("button", name=re.compile("Cancel|Dismiss|OK")).first.click())
                page.goto(server.url)
                page.wait_for_function("window.jupyterapp && window.jupyterapp.commands", timeout=60000)
                page.wait_for_selector(".jp-Notebook .jp-Cell", timeout=60000)
                page.wait_for_function(
                    "(() => { const w = window.jupyterapp.shell.currentWidget; const k = w && "
                    "w.sessionContext && w.sessionContext.session && w.sessionContext.session.kernel;"
                    " return !!k && k.status === 'idle'; })()", timeout=120000)
                page.locator(".jp-Notebook .jp-Cell").first.click()
                page.evaluate("window.jupyterapp.commands.execute('notebook:run-cell')")
                page.wait_for_selector(".lat-cell[role=gridcell]", timeout=60000)
                page.wait_for_selector(".lat-chartview__bar", timeout=60000)
                bars = page.evaluate("document.querySelectorAll('.lat-chartview__bar').length")
                geo = page.evaluate("""(() => {
                    const g = document.querySelector('.lat-cell[role=gridcell]').closest('.jp-OutputArea-output').querySelector('.lat-grid, [class*=lat-]').getBoundingClientRect();
                    const c = document.querySelector('.lat-chartview__bar').closest('svg').getBoundingClientRect();
                    return {gridRight: g.right, chartLeft: c.left, gridTop: g.top, chartTop: c.top};
                })()""")
                before = page.evaluate("[...document.querySelectorAll('div')].find(d => d.__draws !== undefined).__draws")
                page.evaluate("[...document.querySelectorAll('div')].find(d => d.__latticeGrid).__latticeGrid.filters.quick('beta')")
                page.wait_for_function(
                    "[...document.querySelectorAll('div')].find(d => d.__draws !== undefined).__draws > " + str(before),
                    timeout=15000)
                after_bars = page.evaluate("document.querySelectorAll('.lat-chartview__bar').length")
            finally:
                browser.close()
    finally:
        server.close()
    assert bars == 3 and after_bars == 1
    assert geo["chartLeft"] >= geo["gridRight"] - 2, geo   # HBox: the chart sits to the right of the grid
