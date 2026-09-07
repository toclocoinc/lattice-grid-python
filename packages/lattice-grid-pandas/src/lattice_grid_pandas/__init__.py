"""lattice-grid-pandas -- framework-agnostic pandas <-> Lattice Grid glue.

This is the single source of truth for:

* the pandas-dtype -> grid-column-``type`` mapping,
* the columnar (column-major) DataFrame serialization the browser reconstructs,
* value/dtype coercion for the edit round-trip, and
* the grid delivery + licence plumbing shared by every host wrapper.

Both host packages consume it and add nothing of their own to this layer:

* ``lattice-grid-jupyter`` (phase C1, anywidget in a notebook), and
* ``lattice-grid-dash``    (phase C2, a Dash component).

Keeping it here means the dtype map and serialization exist in exactly one place
and are unit-tested once.
"""

from __future__ import annotations

from ._serialize import (
    ROW_KEY,
    build_columnar,
    build_columns,
    cast_to_dtype,
    column_values,
    data_fields,
    lattice_type,
)
from ._grid import CDN_BASE, GRID_VERSION, cdn_urls, load_vendored, resolve_source

__all__ = [
    "ROW_KEY",
    "lattice_type",
    "column_values",
    "build_columns",
    "build_columnar",
    "data_fields",
    "cast_to_dtype",
    "GRID_VERSION",
    "CDN_BASE",
    "cdn_urls",
    "load_vendored",
    "resolve_source",
    "__version__",
]

__version__ = "0.1.0"
