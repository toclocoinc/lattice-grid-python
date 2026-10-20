"""Unit tests for lattice_grid_pandas itself (no UI framework, no browser).

``lattice_grid_jupyter`` re-exports this module's functions verbatim (see
``tests/test_serialize.py`` at the repo root), but the package is meant to
stand on its own -- these import it directly.
"""

import json

import numpy as np
import pandas as pd

from lattice_grid_pandas._serialize import _jsonable_scalar, column_values


# --- BACKLOG-0004772: list / tuple / ndarray cells -> real JSON arrays --------
def test_list_tuple_and_ndarray_cells_serialize_as_json_arrays_not_text():
    s = pd.Series([[1.0, 2.0, 1.5], (3, 4, 5), np.array([6, 7.5]), None], dtype="object")
    vals = column_values(s)
    assert vals[0] == [1.0, 2.0, 1.5] and isinstance(vals[0], list)
    assert vals[1] == [3, 4, 5] and isinstance(vals[1], list)
    assert vals[2] == [6.0, 7.5] and isinstance(vals[2], list)
    assert vals[3] is None
    assert json.loads(json.dumps(vals)) == vals  # real arrays, not str(list) text


def test_ndarray_cell_with_nan_and_numpy_scalars_coerces_per_element():
    assert _jsonable_scalar(np.array([1, np.nan, 3])) == [1.0, None, 3.0]


def test_nested_list_of_lists_coerces_recursively():
    assert _jsonable_scalar([np.int64(1), [np.float64(2.5), None]]) == [1, [2.5, None]]
