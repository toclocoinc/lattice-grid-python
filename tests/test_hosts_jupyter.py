"""HOST PROOF (card 1615): the widget renders and round-trips inside real
JupyterLab 4 and Notebook 7, driven headless by Chrome.

Flow per host: start a real server, open a notebook whose first cell builds the
widget, run it, wait for the grid to render in the output area, REALLY edit a cell
(double-click, type, Enter), run the next cell and read the kernel's DataFrame
back out of that cell's output. Skipped by name if the host or a browser is not
installed -- never faked.  VS Code and Colab cannot be driven here: see docs/hosts.md.
"""

import importlib.util
import re

import pytest

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # noqa: E402

from _hosts import JupyterServer, launch_browser  # noqa: E402

CELLS = [
    "import pandas as pd\n"
    "from lattice_grid_jupyter import LatticeGridWidget\n"
    "df = pd.DataFrame({'name': ['Ada', 'Grace', 'Linus'], 'score': [91, 88, 77]})\n"
    "w = LatticeGridWidget(df, offline=True)\n"
    "w",
    "print('SCORE0=%d dtype=%s' % (w.df.loc[0, 'score'], w.df['score'].dtype))",
]


def _installed(mod):
    return importlib.util.find_spec(mod) is not None


@pytest.mark.parametrize("flavour,module", [("lab", "jupyterlab"), ("notebook", "notebook")])
def test_widget_renders_and_edit_reaches_python(flavour, module):
    if not _installed(module):
        pytest.skip(f"{module} not installed")
    server = JupyterServer(flavour, CELLS)
    try:
        with sync_playwright() as p:
            browser = launch_browser(p)
            if browser is None:
                pytest.skip("no real browser available to launch")
            try:
                page = browser.new_page(viewport={"width": 1400, "height": 1000})
                # Lab may pop "Build Recommended" for unrelated installed extensions
                page.add_locator_handler(
                    page.locator("dialog.jp-Dialog"),
                    lambda d: d.get_by_role("button", name=re.compile("Cancel|Dismiss|OK")).first.click())
                page.goto(server.url)
                page.wait_for_function("window.jupyterapp && window.jupyterapp.commands", timeout=60000)
                # wait for the kernel to be idle-ish: run the first cell
                page.wait_for_selector(".jp-Notebook .jp-Cell", timeout=60000)
                # ...and really idle: a run issued before the session's kernel is
                # connected is dropped by the host, and the test then waits for a
                # grid that was never asked for (F-1618-2)
                if flavour == "lab":
                    page.wait_for_function(
                        "(() => { const w = window.jupyterapp.shell.currentWidget; const k = w && "
                        "w.sessionContext && w.sessionContext.session && w.sessionContext.session.kernel;"
                        " return !!k && k.status === 'idle'; })()", timeout=120000)
                page.locator(".jp-Notebook .jp-Cell").first.click()
                page.evaluate("window.jupyterapp.commands.execute("
                              "'notebook:run-cell-and-select-next')")
                page.wait_for_selector(".lat-cell[role=gridcell]", timeout=60000)

                cell = page.locator(".lat-cell[role=gridcell]", has_text="91").first
                cell.dblclick()
                page.wait_for_selector("input.lat-editor__input", timeout=5000)
                page.keyboard.press("Control+A")
                page.keyboard.type("99")
                page.keyboard.press("Enter")
                page.wait_for_timeout(800)  # comm round-trip

                page.locator(".jp-Notebook .jp-Cell").nth(1).locator(".jp-InputArea-editor").click()
                page.evaluate("window.jupyterapp.commands.execute('notebook:run-cell')")
                page.wait_for_function(
                    "document.body.innerText.includes('SCORE0=')", timeout=60000)
                text = page.evaluate("document.body.innerText")
            finally:
                browser.close()
    finally:
        server.close()
    m = re.search(r"SCORE0=(\d+) dtype=(\w+)", text)
    assert m, text[-500:]
    assert m.group(1) == "99" and m.group(2) == "int64"
