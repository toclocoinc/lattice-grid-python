"""Grid delivery + licence plumbing shared by every host wrapper.

The grid JS/CSS reaches the browser one of two ways, and every wrapper makes the
same choice with the same rules:

* **CDN** -- point at the published bundle on jsDelivr (needs network at render
  time; nothing is vendored). This is the default.
* **vendor** -- carry the grid bundle text in-process and inject it, so the page
  works fully offline (no network at render time). The wrapper is responsible for
  actually shipping the bundle files; this module just locates/loads them.

The **licence** itself is a single opaque string the wrapper hands straight to
``createGrid({ licence })``. The grid resolves it client-side (localhost origins
run free/unwatermarked with no key). There is deliberately no key parsing here --
the browser is the authority -- so "licence plumbing" for the Python side is just:
pass the string through, and surface the grid's resolved state back for debugging.
"""

from __future__ import annotations

import pathlib

# The grid version this generation of wrappers is built and verified against.
GRID_VERSION = "1.40.0"

# Published npm package, served file-for-file by jsDelivr.
_NPM = "@toclocoinc/lattice-grid"


def CDN_BASE(version: str = GRID_VERSION) -> str:
    """jsDelivr base URL for a given grid version."""
    return f"https://cdn.jsdelivr.net/npm/{_NPM}@{version}"


def cdn_urls(version: str = GRID_VERSION) -> dict[str, str]:
    """CDN URLs for the core ESM bundle, the React adapter module, and the CSS."""
    base = CDN_BASE(version)
    return {
        "core": f"{base}/lattice-grid.esm.min.js",
        "react": f"{base}/modules/react.esm.min.js",
        "css": f"{base}/lattice-grid.min.css",
    }


def load_vendored(static_dir: pathlib.Path, js_name: str, css_name: str) -> tuple[str, str]:
    """Read a vendored ``(js, css)`` bundle pair from ``static_dir``.

    Raises a clear error if the wrapper was asked for offline/vendor delivery but
    the bundle was not shipped with it.
    """
    js = static_dir / js_name
    css = static_dir / css_name
    if not js.exists() or not css.exists():
        raise FileNotFoundError(
            "vendor/offline delivery needs the grid bundle at "
            f"{js} and {css}. Reinstall the package (it ships them) or use CDN delivery."
        )
    return js.read_text(encoding="utf-8"), css.read_text(encoding="utf-8")


def resolve_source(offline: bool) -> str:
    """Map an ``offline`` flag to the delivery mode string wrappers sync to JS."""
    return "vendor" if offline else "cdn"
