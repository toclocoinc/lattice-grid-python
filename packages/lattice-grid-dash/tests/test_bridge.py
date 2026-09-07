"""Unit tests for the Dash data-prop bridge (no browser).

These also prove the bridge reuses the shared `lattice-grid-pandas` serialization:
the dtype mapping and columnar build come from there, not from this package.
"""

import numpy as np
import pandas as pd

import lattice_grid_pandas as shared
from lattice_grid_dash import apply_cell_edit, dataframe_to_data
from lattice_grid_dash._serialize_bridge import ROW_KEY


def test_dataframe_to_data_shape():
    df = pd.DataFrame({"name": ["Ada", "Grace"], "score": [91, 88], "active": [True, False]})
    data = dataframe_to_data(df)
    assert set(data) == {"columns", "rowKey", "columnar"}
    assert data["rowKey"] == ROW_KEY
    assert data["columnar"][ROW_KEY] == ["0", "1"]
    assert data["columnar"]["score"] == [91, 88]
    # column types come from the shared dtype map
    types = {c["field"]: c["type"] for c in data["columns"]}
    assert types == {"name": "text", "score": "number", "active": "boolean"}


def test_bridge_uses_shared_serialization():
    # The bridge must not carry its own dtype map: it delegates to the shared pkg.
    df = pd.DataFrame({"when": pd.to_datetime(["2021-06-01", "2021-06-02"])})
    data = dataframe_to_data(df)
    assert data["columns"][0]["type"] == shared.lattice_type(df["when"].dtype) == "timestamp"


def test_apply_cell_edit_roundtrip_typed():
    df = pd.DataFrame({"name": ["Ada"], "score": [91]})
    apply_cell_edit(df, {"key": "0", "colId": "score", "value": "99"})
    assert df.loc[0, "score"] == 99
    assert pd.api.types.is_integer_dtype(df["score"].dtype)


def test_apply_cell_edit_ignores_unknown_col_and_row():
    df = pd.DataFrame({"a": [1, 2]})
    apply_cell_edit(df, {"key": "0", "colId": "nope", "value": 5})
    apply_cell_edit(df, {"key": "99", "colId": "a", "value": 5})
    assert df["a"].tolist() == [1, 2]
