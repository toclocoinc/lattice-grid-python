"""lattice-grid-dash -- a Dash component wrapping Lattice Grid.

Usage::

    import lattice_grid_dash
    from dash import Dash, callback, Input, Output, html
    import pandas as pd
    from lattice_grid_dash import dataframe_to_data

    df = pd.DataFrame({"name": ["Ada", "Grace"], "score": [91, 88]})
    app = Dash(__name__)
    app.layout = html.Div([
        lattice_grid_dash.LatticeGrid(id="grid", data=dataframe_to_data(df)),
        html.Div(id="out"),
    ])

    @callback(Output("out", "children"), Input("grid", "cellChanged"))
    def show(edit):
        return f"edited {edit}" if edit else "no edits yet"

The grid JS is *vendored* into this package (``lattice_grid_dash.min.js``), so it
works offline with no CDN at render time. See the README for the CDN variant.
"""

import json as _json
import os as _os
import sys as _sys

import dash as _dash

from ._imports_ import *  # noqa: F401,F403
from ._imports_ import __all__  # noqa: F401
from ._serialize_bridge import dataframe_to_data, apply_cell_edit  # noqa: F401

__all__ = list(__all__) + ["dataframe_to_data", "apply_cell_edit"]

if not hasattr(_dash, "development"):
    print(
        "Dash was not successfully imported. Make sure you don't have a file "
        'named \n"dash.py" in your current directory.',
        file=_sys.stderr,
    )
    _sys.exit(1)

_basepath = _os.path.dirname(__file__)
_filepath = _os.path.abspath(_os.path.join(_basepath, "package-info.json"))
with open(_filepath) as f:
    package = _json.load(f)

package_name = package["name"].replace(" ", "_").replace("-", "_")
__version__ = package["version"]

_current_path = _os.path.dirname(_os.path.abspath(__file__))

_this_module = _sys.modules[__name__]

_js_dist = [
    {
        "relative_package_path": "lattice_grid_dash.min.js",
        "namespace": package_name,
    }
]

_css_dist = [
    {
        "relative_package_path": "lattice-grid.min.css",
        "namespace": package_name,
    }
]

for _component in __all__:
    if hasattr(_this_module, _component):
        _comp = getattr(_this_module, _component)
        if isinstance(_comp, type):
            setattr(_comp, "_js_dist", _js_dist)
            setattr(_comp, "_css_dist", _css_dist)
