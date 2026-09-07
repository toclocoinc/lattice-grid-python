"""
DataFrame <-> Lattice grid serialization  (compatibility re-export).

The dtype map and the columnar serialization used to live here. They were
extracted verbatim into the framework-agnostic ``lattice-grid-pandas`` package so
the Jupyter widget (this package) and the Dash component share ONE implementation
of the dtype mapping and payload build -- no second copy to drift.

This module now just re-exports that single source of truth, so existing imports
(``from lattice_grid_jupyter import _serialize as S``; ``S.build_columnar`` etc.)
keep working unchanged.
"""

from __future__ import annotations

from lattice_grid_pandas._serialize import (  # noqa: F401
    ROW_KEY,
    _is_missing,
    _jsonable_scalar,
    _index_levels,
    build_columnar,
    build_columns,
    cast_to_dtype,
    column_values,
    data_fields,
    lattice_type,
)

__all__ = [
    "ROW_KEY",
    "lattice_type",
    "column_values",
    "build_columns",
    "build_columnar",
    "data_fields",
    "cast_to_dtype",
]
