"""BACKLOG-0005025: the widget's grid follows the notebook host's theme, and
the row-selection checkbox is clearly ticked vs unticked in every theme.

The host theme is detected in the browser (host attributes or the page's
computed background, never the OS scheme); an explicit ``theme`` option wins.
A real Chromium renders the front end under a light and a dark host, and the
checkbox is checked by REAL pixels (a native checkbox under a dark host used to
render solid black whether ticked or not, which computed styles could not see).
"""

import struct
import zlib

import pandas as pd
import pytest

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # noqa: E402

from _hosts import launch_browser  # noqa: E402
from _page import page_html  # noqa: E402
from lattice_grid_jupyter import LatticeGridWidget  # noqa: E402

DARK_BG = "rgb(20, 24, 28)"      # the grid's dark theme background (#14181c)
LIGHT_BG = "rgb(255, 255, 255)"  # the grid's light theme background


def _df():
    return pd.DataFrame({"name": ["Ada", "Grace", "Linus"], "score": [91, 88, 77]})


# --- option plumbing ---------------------------------------------------------
def test_theme_option_is_passed_through_and_wins_live():
    # no option: the browser auto-detects the host theme, so Python sends nothing
    assert "theme" not in LatticeGridWidget(_df())._grid_options
    w = LatticeGridWidget(_df(), options={"theme": "dark"})
    assert w._grid_options["theme"] == "dark"
    w.options = {"theme": "light"}            # live update, no data re-sent
    assert w._grid_options["theme"] == "light"
    w.options = {"row_height": 30}            # dropping it hands back to the host
    assert "theme" not in w._grid_options and w._grid_options["rowHeight"] == 30


# --- real browser ------------------------------------------------------------
def _open(p, w, body_attrs="", head=""):
    browser = launch_browser(p)
    if browser is None:
        pytest.skip("no real browser available to launch")
    page = browser.new_page(viewport={"width": 1000, "height": 600})
    page.set_content(page_html(w, body_attrs=body_attrs, head=head))
    page.wait_for_function("window.__rendered === true || window.__error", timeout=30000)
    assert page.evaluate("window.__error") is None
    page.wait_for_selector(".lat-cell[role=gridcell]", timeout=10000)
    return browser, page


def _grid_theme(page):
    return page.evaluate("document.querySelector('.lat-grid-root').getAttribute('data-theme')")


def _grid_background(page):
    return page.evaluate("getComputedStyle(document.querySelector('.lat-grid-root')).backgroundColor")


@pytest.mark.parametrize("host,theme,bg", [
    ("light", "light", LIGHT_BG),
    ("dark-attr", "dark", DARK_BG),       # JupyterLab: <body data-jp-theme-light="false">
    ("dark-class", "dark", DARK_BG),      # VS Code / Colab: a dark body class
    ("dark-stylesheet", "dark", DARK_BG),  # fallback: the page's computed background
])
def test_grid_follows_the_host_theme(host, theme, bg):
    attrs = {"light": "", "dark-attr": 'data-jp-theme-light="false"',
             "dark-class": 'class="dark-theme"', "dark-stylesheet": ""}[host]
    head = {"dark-stylesheet": "<style>body { background: #14181c; }</style>"}.get(host, "")
    w = LatticeGridWidget(_df(), offline=True)
    with sync_playwright() as p:
        browser, page = _open(p, w, body_attrs=attrs, head=head)
        try:
            assert _grid_theme(page) == theme
            assert _grid_background(page) == bg
        finally:
            browser.close()


def test_explicit_theme_option_wins_over_the_host():
    w = LatticeGridWidget(_df(), offline=True, options={"theme": "dark"})
    with sync_playwright() as p:
        browser, page = _open(p, w)  # a light host
        try:
            assert _grid_theme(page) == "dark" and _grid_background(page) == DARK_BG
        finally:
            browser.close()
    w = LatticeGridWidget(_df(), offline=True, options={"theme": "light"})
    with sync_playwright() as p:
        browser, page = _open(p, w, body_attrs='data-theme="dark"')
        try:
            assert _grid_theme(page) == "light" and _grid_background(page) == LIGHT_BG
        finally:
            browser.close()


def test_grid_follows_a_host_theme_switch():
    w = LatticeGridWidget(_df(), offline=True)
    with sync_playwright() as p:
        browser, page = _open(p, w, body_attrs='data-theme="light"')
        try:
            assert _grid_theme(page) == "light"
            page.evaluate("document.body.setAttribute('data-theme', 'dark')")
            page.wait_for_function(
                "document.querySelector('.lat-grid-root').getAttribute('data-theme') === 'dark'",
                timeout=4000)
            assert _grid_background(page) == DARK_BG
            page.evaluate("document.body.setAttribute('data-theme', 'light')")
            page.wait_for_function(
                "document.querySelector('.lat-grid-root').getAttribute('data-theme') === 'light'",
                timeout=4000)
            assert _grid_background(page) == LIGHT_BG
        finally:
            browser.close()


# --- checkbox pixels ---------------------------------------------------------
def _png_pixels(data):
    """Decode an 8-bit, non-interlaced PNG (what Chromium screenshots are) to RGB rows."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, width, height, idat = 8, 0, 0, b""
    channels = 3
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        if ctype == b"IHDR":
            width, height, _depth, colour = struct.unpack(">IIBB", chunk[:10])
            channels = {0: 1, 2: 3, 4: 2, 6: 4}[colour]
        elif ctype == b"IDAT":
            idat += chunk
        elif ctype == b"IEND":
            break
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = width * channels
    rows, prev, p = [], bytearray(stride), 0
    for _ in range(height):
        f = raw[p]
        p += 1
        line = bytearray(raw[p:p + stride])
        p += stride
        if f == 1:  # Sub
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif f == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif f == 3:  # Average
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif f == 4:  # Paeth
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b, c = prev[i], (prev[i - channels] if i >= channels else 0)
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        rows.append(bytes(line))
        prev = line
    return width, height, channels, rows


def _pixel(data, x, y):
    width, _height, channels, rows = _png_pixels(data)
    row = rows[y]
    if channels == 1:
        v = row[x]
        return (v, v, v)
    if channels == 2:
        v = row[x * 2]
        return (v, v, v)
    return tuple(row[x * channels:x * channels + 3])


def _lum(rgb):
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def _contrast(a, b):
    hi, lo = max(_lum(a), _lum(b)), min(_lum(a), _lum(b))
    return (hi + 0.05) / (lo + 0.05)


def _hex_rgb(h):
    h = h.strip()
    if not h.startswith("#") or len(h) != 7:
        return None
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _near(rgb, target, tol=42):
    return all(abs(rgb[i] - target[i]) <= tol for i in range(3))


def _checkbox_pixels(page):
    """The checkbox's rect plus the pixels we sample: border, interior, cell bg."""
    info = page.evaluate("""() => {
      const cb = document.querySelector('.lat-checkbox-input');
      const cell = document.querySelector('.lat-cell[data-col="__lattice_select"]');
      const r = cb.getBoundingClientRect(), c = cell.getBoundingClientRect();
      const cs = getComputedStyle(cb);
      return {
        x: r.x, y: r.y, w: r.width, h: r.height,
        cellX: c.x, cellY: c.y, cellW: c.width, cellH: c.height,
        accent: cs.getPropertyValue('--lattice-accent').trim(),
        accentContrast: cs.getPropertyValue('--lattice-accent-contrast').trim(),
      };
    }""")
    shot = page.screenshot()
    border = _pixel(shot, int(info["x"] + info["w"] / 2), int(info["y"] + 1))
    interior = _pixel(shot, int(info["x"] + info["w"] * 0.3), int(info["y"] + info["h"] * 0.35))
    cellbg = _pixel(shot, int(info["cellX"] + 6), int(info["cellY"] + info["cellH"] / 2))
    return info, shot, border, interior, cellbg


@pytest.mark.parametrize("theme,body_attrs", [("light", ""), ("dark", 'data-theme="dark"')])
def test_checkbox_ticked_and_unticked_differ_by_pixels(theme, body_attrs):
    w = LatticeGridWidget(_df(), offline=True, options={"selection": {"checkbox": True}})
    with sync_playwright() as p:
        browser, page = _open(p, w, body_attrs=body_attrs)
        try:
            info, shot, border, interior, cellbg = _checkbox_pixels(page)
            page.mouse.click(info["x"] + info["w"] / 2, info["y"] + info["h"] / 2)
            page.wait_for_function("document.querySelector('.lat-checkbox-input').checked", timeout=4000)
            _i2, shot2, border2, interior2, _cellbg2 = _checkbox_pixels(page)
        finally:
            browser.close()

    accent = _hex_rgb(info["accent"])
    tick = _hex_rgb(info["accentContrast"])
    assert accent and tick, (info["accent"], info["accentContrast"])

    # ticked vs unticked: the box fills with the theme's accent colour, a real
    # and clearly different state from the unticked surface/muted box (the bug:
    # ticked and unticked both drew as solid black squares).
    assert _near(border2, accent) and not _near(border, accent), (border, border2, accent)
    assert _near(interior2, accent) and not _near(interior, accent), (interior, interior2, accent)

    # the unticked box border keeps >= 3:1 against the cell behind it, and the
    # tick keeps >= 3:1 against the filled box it sits on
    assert _contrast(border, cellbg) >= 3.0, (border, cellbg)
    assert _contrast(tick, accent) >= 3.0, (tick, accent)
    width, height, channels, rows = _png_pixels(shot2)
    x0, y0 = int(info["x"]), int(info["y"])
    found_tick = False
    for yy in range(y0, y0 + int(info["h"])):
        for xx in range(x0, x0 + int(info["w"])):
            if channels == 1:
                v = rows[yy][xx]
                px = (v, v, v)
            elif channels == 2:
                v = rows[yy][xx * 2]
                px = (v, v, v)
            else:
                px = tuple(rows[yy][xx * channels:xx * channels + 3])
            if _near(px, tick):
                found_tick = True
                break
        if found_tick:
            break
    assert found_tick, f"no tick-coloured pixel ({tick}) in the ticked checkbox"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
