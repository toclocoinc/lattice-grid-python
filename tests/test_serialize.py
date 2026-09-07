"""Unit tests for the DataFrame <-> grid serialization layer (no browser needed)."""

import math

import numpy as np
import pandas as pd
import pytest

from lattice_grid_jupyter import _serialize as S


def test_dtype_mapping():
    assert S.lattice_type(pd.Series([1, 2]).dtype) == "number"
    assert S.lattice_type(pd.Series([1.0, 2.0]).dtype) == "number"
    assert S.lattice_type(pd.Series([True, False]).dtype) == "boolean"
    assert S.lattice_type(pd.Series(pd.to_datetime(["2020-01-01"])).dtype) == "timestamp"
    assert S.lattice_type(pd.Series(["a", "b"]).dtype) == "text"
    assert S.lattice_type(pd.Series(["a", "b"], dtype="string").dtype) == "text"
    assert S.lattice_type(pd.Series(["a", "b"]).astype("category").dtype) == "text"


def test_nullable_and_missing_normalize_to_none():
    s_int = pd.array([1, None, 3], dtype="Int64")
    assert S.column_values(pd.Series(s_int)) == [1, None, 3]

    s_float = pd.Series([1.5, np.nan, 3.5])
    vals = S.column_values(s_float)
    assert vals[0] == 1.5 and vals[1] is None and vals[2] == 3.5

    s_bool = pd.Series(pd.array([True, None, False], dtype="boolean"))
    assert S.column_values(s_bool) == [True, None, False]

    s_dt = pd.Series(pd.to_datetime(["2021-06-01", None, "2021-06-03"]))
    dv = S.column_values(s_dt)
    assert dv[1] is None and dv[0].startswith("2021-06-01")

    s_obj = pd.Series(["x", None, float("nan")], dtype="object")
    assert S.column_values(s_obj) == ["x", None, None]


def test_datetime_tz_isoformat():
    s = pd.Series(pd.to_datetime(["2021-06-01T12:00:00"]).tz_localize("UTC"))
    v = S.column_values(s)[0]
    assert v.startswith("2021-06-01T12:00:00") and ("+00:00" in v or "Z" in v)


def test_default_rangeindex_has_no_index_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    cols = S.build_columns(df)
    assert [c["field"] for c in cols] == ["a"]
    assert cols[0]["edit"] is True


def test_named_and_nondefault_index_becomes_readonly_column():
    df = pd.DataFrame({"x": [10, 20]}, index=pd.Index(["alpha", "beta"], name="label"))
    cols = S.build_columns(df)
    assert cols[0]["field"] == "label" and cols[0]["edit"] is False
    assert cols[1]["field"] == "x" and cols[1]["edit"] is True
    payload = S.build_columnar(df, ["0", "1"])
    assert payload["label"] == ["alpha", "beta"]
    assert payload["x"] == [10, 20]
    assert payload["__row_id__"] == ["0", "1"]


def test_multiindex_becomes_one_column_per_level():
    mi = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)], names=["grp", "sub"])
    df = pd.DataFrame({"v": [7, 8, 9]}, index=mi)
    cols = S.build_columns(df)
    fields = [c["field"] for c in cols]
    assert fields == ["grp", "sub", "v"]
    payload = S.build_columnar(df, ["0", "1", "2"])
    assert payload["grp"] == ["a", "a", "b"]
    assert payload["sub"] == [1, 2, 1]


def test_duplicate_index_labels_still_get_distinct_keys():
    df = pd.DataFrame({"v": [1, 2, 3]}, index=["dup", "dup", "dup"])
    keys = ["0", "1", "2"]
    payload = S.build_columnar(df, keys)
    # keys are unique even though the index label repeats
    assert payload["__row_id__"] == ["0", "1", "2"]
    assert payload["index"] == ["dup", "dup", "dup"]


def test_cast_to_dtype_preserves_types():
    assert isinstance(S.cast_to_dtype(np.dtype("int64"), "5"), int)
    assert isinstance(S.cast_to_dtype(np.dtype("float64"), "5"), float)
    assert S.cast_to_dtype(np.dtype("bool"), "true") is True
    assert S.cast_to_dtype(np.dtype("bool"), "false") is False
    assert S.cast_to_dtype(np.dtype("int64"), None) is None
    ts = S.cast_to_dtype(np.dtype("datetime64[ns]"), "2022-03-04")
    assert isinstance(ts, pd.Timestamp)


def test_large_frame_columnar_build_is_fast():
    import time
    n = 100_000
    df = pd.DataFrame({
        "id": np.arange(n),
        "val": np.random.rand(n),
        "flag": np.random.rand(n) > 0.5,
        "when": pd.date_range("2020-01-01", periods=n, freq="min"),
        "label": [f"r{i}" for i in range(n)],
    })
    keys = [str(i) for i in range(n)]
    t = time.time()
    payload = S.build_columnar(df, keys)
    elapsed = time.time() - t
    assert len(payload["__row_id__"]) == n
    # Vectorized, column-major: comfortably sub-second in practice; 5s is slack
    # for a loaded CI box. A naive per-cell iterrows build is ~15-20x slower.
    assert elapsed < 5.0, f"columnar build took {elapsed:.2f}s"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
