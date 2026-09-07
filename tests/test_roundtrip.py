"""
Round-trip tests for the widget's PYTHON half (no browser).

Builds the widget from a DataFrame, asserts the grid-facing serialization, then
replays the EXACT comm payload the grid emits on ``cell:changed`` and asserts the
DataFrame updated and stayed typed -- plus the Python->grid live-update APIs
(set_data / append_rows / delete_rows).
"""

import numpy as np
import pandas as pd
import pytest

from lattice_grid_jupyter import LatticeGridWidget


@pytest.fixture
def df():
    return pd.DataFrame({
        "name": ["Ada", "Grace", "Linus"],
        "score": [91, 88, 77],          # int64
        "ratio": [0.91, 0.88, 0.77],    # float64
        "active": [True, False, True],  # bool
    })


def test_serialization(df):
    w = LatticeGridWidget(df)
    assert w._row_key == "__row_id__"
    types = {c["field"]: c["type"] for c in w._columns}
    assert types == {"name": "text", "score": "number", "ratio": "number", "active": "boolean"}
    assert all(c["edit"] for c in w._columns)
    assert w._columnar["__row_id__"] == ["0", "1", "2"]
    assert w._columnar["name"] == ["Ada", "Grace", "Linus"]
    assert w._columnar["score"] == [91, 88, 77]


def test_edit_roundtrip_is_dtype_preserving(df):
    w = LatticeGridWidget(df)
    w.apply_edit(key="1", col_id="score", value=99)
    w.apply_edit(key="0", col_id="name", value="Ada Lovelace")
    w.apply_edit(key="2", col_id="active", value=False)

    assert w.df.loc[1, "score"] == 99
    assert pd.api.types.is_integer_dtype(w.df["score"].dtype)
    assert w.df.loc[0, "name"] == "Ada Lovelace"
    assert w.df.loc[2, "active"] == False  # noqa: E712
    assert w.df.loc[2, "score"] == 77  # untouched


def test_string_int_edit_casts_back_to_int(df):
    w = LatticeGridWidget(df)
    w.apply_edit(key="0", col_id="score", value="42")  # grid may deliver a string
    assert w.df.loc[0, "score"] == 42
    assert pd.api.types.is_integer_dtype(w.df["score"].dtype)


def test_nondefault_index_edit_maps_by_position():
    df2 = pd.DataFrame({"x": [10, 20]}, index=["alpha", "beta"])
    w = LatticeGridWidget(df2)
    assert w._columnar["index"] == ["alpha", "beta"]
    w.apply_edit(key="1", col_id="x", value=222)  # key "1" == 2nd row == "beta"
    assert w.df.loc["beta", "x"] == 222


def test_duplicate_index_edit_hits_the_right_row():
    df3 = pd.DataFrame({"v": [1, 2, 3]}, index=["d", "d", "d"])
    w = LatticeGridWidget(df3)
    w.apply_edit(key="1", col_id="v", value=99)  # middle row only
    assert list(w.df["v"]) == [1, 99, 3]


def test_index_column_edits_are_ignored():
    df2 = pd.DataFrame({"x": [10, 20]}, index=pd.Index(["a", "b"], name="label"))
    w = LatticeGridWidget(df2)
    w.apply_edit(key="0", col_id="label", value="ZZZ")  # read-only index col
    assert list(w.df.index) == ["a", "b"]


def test_licence_wiring(df):
    w = LatticeGridWidget(df)
    assert w.licence == ""              # localhost free tier
    assert w._grid_source == "cdn"
    w3 = LatticeGridWidget(df, licence="LG-DEPLOYED-KEY")
    assert w3.licence == "LG-DEPLOYED-KEY"


def test_offline_carries_vendored_bundle(df):
    w = LatticeGridWidget(df, offline=True)
    assert w._grid_source == "vendor"
    assert "LatticeGrid" in w._grid_js and len(w._grid_js) > 100_000
    assert len(w._grid_css) > 1000


def test_set_data_replaces_and_bumps_version(df):
    w = LatticeGridWidget(df)
    v0 = w._data_version
    df_new = pd.DataFrame({"name": ["Zoe"], "score": [5], "ratio": [0.5], "active": [True]})
    w.set_data(df_new)
    assert w._data_version == v0 + 1
    assert list(w.df["name"]) == ["Zoe"]
    assert w._columnar["name"] == ["Zoe"]


def test_append_rows_incremental(df):
    w = LatticeGridWidget(df)
    keys = w.append_rows([{"name": "Zoe", "score": 5, "ratio": 0.5, "active": True}])
    assert len(w.df) == 4
    assert w.df.iloc[-1]["name"] == "Zoe"
    assert w._row_op["op"] == "add"
    assert w._row_op["rows"][0]["name"] == "Zoe"
    assert w._row_op["rows"][0]["__row_id__"] == keys[0]
    # appended key is distinct from the originals
    assert keys[0] not in {"0", "1", "2"}


def test_delete_rows_incremental_keeps_keys_aligned(df):
    w = LatticeGridWidget(df)
    w.delete_rows("1")  # remove Grace
    assert list(w.df["name"]) == ["Ada", "Linus"]
    assert w._row_op == {"op": "remove", "keys": ["1"]}
    # surviving keys are unchanged, so a later edit still lands correctly
    w.apply_edit(key="2", col_id="name", value="Linus T.")
    assert w.df.iloc[1]["name"] == "Linus T."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
