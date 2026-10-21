"""BACKLOG-0005167: LatticeChart follows the notebook host's theme like the grid.

The grid widget's detection (BACKLOG-0005025) is the one copy; the chart front
end gets it spliced in. A real Chromium renders a grid and a bound chart (and a
DataFrame chart) under a light and a dark host stylesheet, and the chart's
background and axis colours are read from REAL pixels / rendered text.
"""

import struct
import zlib

import pandas as pd
import pytest

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # noqa: E402

from _chartpage import bind_python, page_html  # noqa: E402
from _hosts import launch_browser  # noqa: E402
from lattice_grid_jupyter import LatticeChart, LatticeGridWidget  # noqa: E402

DARK_BG = (20, 24, 28)      # the grid's dark theme background (#14181c)
LIGHT_BG = (255, 255, 255)
DARK_STYLE = "<style>body { background: #14181c; }</style>"


def _df():
    return pd.DataFrame({"seg": ["a", "b", "c"], "v": [1, 2, 3]})


def _png_pixel(png: bytes, x: int, y: int):
    """One RGB pixel of an 8-bit RGB/RGBA PNG (no imaging library needed)."""
    pos, idat, w = 8, b"", 0
    while pos < len(png):
        n, kind = struct.unpack(">I4s", png[pos:pos + 8])
        body = png[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, _h, depth, ctype = struct.unpack(">IIBB", body[:10])
            assert depth == 8 and ctype in (2, 6)
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    bpp = 4 if ctype == 6 else 3
    raw, stride, prev = zlib.decompress(idat), w * bpp, bytearray(w * bpp)
    for row in range(y + 1):
        base = row * (stride + 1)
        f, cur = raw[base], bytearray(raw[base + 1:base + 1 + stride])
        for i in range(stride):
            a = cur[i - bpp] if i >= bpp else 0
            b, c = prev[i], (prev[i - bpp] if i >= bpp else 0)
            if f == 1:
                cur[i] = (cur[i] + a) & 255
            elif f == 2:
                cur[i] = (cur[i] + b) & 255
            elif f == 3:
                cur[i] = (cur[i] + (a + b) // 2) & 255
            elif f == 4:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                cur[i] = (cur[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        prev = cur
    return tuple(prev[x * bpp:x * bpp + 3])


PLOT = "document.querySelector('#root1 div[style*=height]')"


def _open(p, widgets, body_attrs="", head=""):
    browser = launch_browser(p)
    if browser is None:
        pytest.skip("no real browser available to launch")
    page = browser.new_page(viewport={"width": 1000, "height": 800})
    bind_python(page, widgets)
    html = page_html(widgets).replace("<body>", f"<body {body_attrs}>").replace(
        "<head><meta charset=utf-8>", f"<head><meta charset=utf-8>{head}")
    page.set_content(html)
    page.wait_for_function("window.__rendered === true || window.__error", timeout=30000)
    assert page.evaluate("window.__error") is None
    page.wait_for_function(
        f"(() => {{ const p = {PLOT}; return p && p.__latticeChart && p.__latticeChart.data(); }})()", timeout=15000)
    return browser, page


def _text_fill(page):
    return page.evaluate(
        f"(() => {{ const t = {PLOT}.querySelector('svg text'); return getComputedStyle(t).fill; }})()")


def _chart_pixel(page):
    """A background pixel inside the chart's box, away from marks and axes."""
    box = page.evaluate(f"(() => {{ const r = {PLOT}.getBoundingClientRect(); return [r.left, r.top]; }})()")
    png = page.screenshot()
    return _png_pixel(png, int(box[0]) + 40, int(box[1]) + 4)


def _dark(fill):
    r, g, b = (int(v) for v in fill[fill.index("(") + 1:-1].split(",")[:3])
    return (r + g + b) / 3 > 128      # a dark theme draws LIGHT text


def _pair(**chart_kw):
    g = LatticeGridWidget(_df(), offline=True, height=200)
    return g, LatticeChart(g, type="bar", x="seg", y="v", offline=True, height=260, **chart_kw)


@pytest.mark.parametrize("host", ["dark-stylesheet", "dark-attr", "dark-class"])
def test_chart_is_dark_under_a_dark_host(host):
    attrs = {"dark-stylesheet": "", "dark-attr": 'data-jp-theme-light="false"',
             "dark-class": 'class="dark-theme"'}[host]
    g, c = _pair()
    with sync_playwright() as p:
        browser, page = _open(p, [g, c], body_attrs=attrs, head=DARK_STYLE)
        try:
            assert page.evaluate(f"{PLOT}.getAttribute('data-theme')") == "dark"
            assert _dark(_text_fill(page))                       # axis labels are light on dark
            assert _chart_pixel(page) == DARK_BG
        finally:
            browser.close()


def test_chart_is_unchanged_under_a_light_host():
    g, c = _pair()
    with sync_playwright() as p:
        browser, page = _open(p, [g, c])
        try:
            assert page.evaluate(f"{PLOT}.getAttribute('data-theme')") == "light"
            assert not _dark(_text_fill(page))
            assert _chart_pixel(page) == LIGHT_BG
        finally:
            browser.close()


def test_an_explicit_chart_theme_wins_over_the_host_and_the_grid():
    g = LatticeGridWidget(_df(), offline=True, height=200, options={"theme": "dark"})
    c = LatticeChart(g, type="bar", x="seg", y="v", offline=True, height=260, theme="light")
    assert c._spec["theme"] == "light"
    with sync_playwright() as p:
        browser, page = _open(p, [g, c], body_attrs='data-jp-theme-light="false"', head=DARK_STYLE)
        try:
            assert page.evaluate(f"{PLOT}.getAttribute('data-theme')") == "light"
            assert not _dark(_text_fill(page))
            assert _chart_pixel(page) == LIGHT_BG
        finally:
            browser.close()


def test_a_chart_follows_the_grids_resolved_theme():
    # the grid says dark explicitly, the host is light: the chart goes with the grid
    g = LatticeGridWidget(_df(), offline=True, height=200, options={"theme": "dark"})
    c = LatticeChart(g, type="bar", x="seg", y="v", offline=True, height=260)
    with sync_playwright() as p:
        browser, page = _open(p, [g, c])
        try:
            assert page.evaluate(f"{PLOT}.getAttribute('data-theme')") == "dark"
            assert _chart_pixel(page) == DARK_BG
        finally:
            browser.close()


def test_a_chart_follows_a_live_host_switch():
    g, c = _pair()
    with sync_playwright() as p:
        browser, page = _open(p, [g, c])
        try:
            assert page.evaluate(f"{PLOT}.getAttribute('data-theme')") == "light"
            page.evaluate("document.body.setAttribute('data-jp-theme-light', 'false');"
                          "document.body.style.background = '#14181c'")
            page.wait_for_function(f"{PLOT}.getAttribute('data-theme') === 'dark'", timeout=5000)
            assert _chart_pixel(page) == DARK_BG
            page.evaluate("document.body.setAttribute('data-jp-theme-light', 'true');"
                          "document.body.style.background = ''")
            page.wait_for_function(f"{PLOT}.getAttribute('data-theme') === 'light'", timeout=5000)
            assert _chart_pixel(page) == LIGHT_BG
        finally:
            browser.close()


def test_a_chart_bound_to_a_dataframe_follows_the_host_theme():
    c = LatticeChart(_df(), type="bar", x="seg", y="v", offline=True, height=260)
    with sync_playwright() as p:
        browser = launch_browser(p)
        if browser is None:
            pytest.skip("no real browser available to launch")
        try:
            page = browser.new_page(viewport={"width": 1000, "height": 600})
            page.set_content(page_html([c]).replace("<body>", '<body data-jp-theme-light="false">'))
            page.wait_for_function("window.__rendered === true || window.__error", timeout=30000)
            assert page.evaluate("window.__error") is None
            sel = "document.querySelector('#root0 div[style*=height]')"
            page.wait_for_function(f"(() => {{ const p = {sel}; return p && p.__latticeChart && p.__latticeChart.data(); }})()")
            assert page.evaluate(f"{sel}.getAttribute('data-theme')") == "dark"
        finally:
            browser.close()
