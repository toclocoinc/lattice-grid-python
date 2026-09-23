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


# --- BACKLOG-0001425: UTC end-to-end datetime contract ------------------------
#
# Three column kinds the contract names explicitly: a naive column (declared
# UTC), a UTC-aware column, and a non-UTC tz-aware column (Europe/London, to
# catch a DST-observing zone). Each must serialize to an offset-marked UTC ISO
# string (never a bare wall-clock string) and must read every shape the grid
# can hand back on an edit -- a Number (epoch ms), an ISO string with "Z", and
# an ISO string with a non-UTC offset -- back to the exact original instant.

_DT_KINDS = {
    "naive": None,
    "utc": "UTC",
    "london": "Europe/London",
}


def _dt_column(kind: str, values) -> pd.Series:
    s = pd.Series(pd.to_datetime(values))
    tz = _DT_KINDS[kind]
    return s.dt.tz_localize(tz) if tz else s


@pytest.mark.parametrize("kind", ["naive", "utc", "london"])
def test_datetime_column_values_are_never_bare_wall_clock_strings(kind):
    """Outbound: every column kind carries an explicit offset marker.

    A bare "YYYY-MM-DDTHH:MM:SS" (no "Z", no "+HH:MM") is parsed by the grid's
    ``timestamp`` type as a WALL-CLOCK time in the browser's own zone, not the
    UTC instant it was declared to be -- this is the outbound half of the
    corruption defect.
    """
    v = S.column_values(_dt_column(kind, ["2021-06-01T12:00:00"]))[0]
    assert v.endswith("Z") or v.endswith("+00:00"), f"{kind}: {v!r} has no offset marker"


@pytest.mark.parametrize("kind", ["naive", "utc", "london"])
def test_datetime_column_values_nat_is_none(kind):
    """NaT survives serialization as JSON null, for every column kind."""
    vals = S.column_values(_dt_column(kind, ["2021-06-01T12:00:00", None]))
    assert vals[1] is None


@pytest.mark.parametrize("kind", ["naive", "utc", "london"])
@pytest.mark.parametrize("edit_kind", ["number", "iso_z", "iso_offset"])
def test_cast_to_dtype_datetime_roundtrips_to_the_millisecond(kind, edit_kind):
    """The grid's actual edit payloads, cast back to every column kind.

    A normal grid edit on a ``timestamp`` column hands back a Number (epoch
    MILLISECONDS -- the grid's internal storage unit). Before this fix,
    ``cast_to_dtype`` ran that Number through ``pd.to_datetime`` with pandas'
    default nanosecond unit, landing on 1970 instead of the edited instant
    (BACKLOG-0001425). This asserts every column kind reads every payload
    shape the grid can send back to the exact original instant.
    """
    original = pd.Timestamp("2021-06-01T12:34:56.789", tz="UTC")
    dtype = _dt_column(kind, ["2020-01-01"]).dtype

    if edit_kind == "number":
        edit_value = original.value // 1_000_000  # epoch ms, as the grid sends it
    elif edit_kind == "iso_z":
        edit_value = original.isoformat().replace("+00:00", "Z")
    else:  # iso_offset: a non-UTC offset, not the column's own zone
        edit_value = original.tz_convert("America/New_York").isoformat()

    result = S.cast_to_dtype(dtype, edit_value)
    assert isinstance(result, pd.Timestamp)

    result_utc = result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")
    assert result_utc == original, f"{kind}/{edit_kind}: {result_utc} != {original}"

    if kind == "naive":
        assert result.tzinfo is None
    else:
        assert result.tzinfo is not None


@pytest.mark.parametrize("kind", ["naive", "utc", "london"])
def test_cast_to_dtype_datetime_offsetless_iso_string_is_utc(kind):
    """An offset-less ISO string from the grid is UTC under this contract,
    never the machine's local zone (BACKLOG-0001425 timezone contract)."""
    dtype = _dt_column(kind, ["2020-01-01"]).dtype
    result = S.cast_to_dtype(dtype, "2021-06-01T12:34:56")
    result_utc = result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")
    assert result_utc == pd.Timestamp("2021-06-01T12:34:56", tz="UTC")


@pytest.mark.parametrize("kind", ["naive", "utc", "london"])
def test_cast_to_dtype_datetime_none_stays_none(kind):
    """A cleared/NaT edit stays None for every column kind, so the DataFrame
    assignment lands on NaT rather than being coerced to some other sentinel."""
    dtype = _dt_column(kind, ["2020-01-01"]).dtype
    assert S.cast_to_dtype(dtype, None) is None


def test_datetime_full_roundtrip_through_a_dataframe_column_assignment():
    """End to end (no browser): outbound serialize -> grid Number edit ->
    cast_to_dtype -> DataFrame assignment reproduces the original instant,
    for a naive (UTC-declared), a UTC-aware, and an Europe/London-aware
    column, with a NaT row surviving untouched in each."""
    for kind in ("naive", "utc", "london"):
        original = pd.Timestamp("2021-06-01T12:34:56.789", tz="UTC")
        wall = pd.Series(pd.to_datetime([original.tz_localize(None), None]))
        if kind == "naive":
            col = wall
        elif kind == "utc":
            col = wall.dt.tz_localize("UTC")
        else:  # london
            col = wall.dt.tz_localize("UTC").dt.tz_convert("Europe/London")
        df = pd.DataFrame({"when": col})

        # NaT round-trips through serialization untouched.
        assert S.column_values(df["when"])[1] is None

        # Grid emits a Number (epoch ms) on edit; cast it back into row 0.
        edit_value = original.value // 1_000_000
        df.iloc[0, df.columns.get_loc("when")] = S.cast_to_dtype(df["when"].dtype, edit_value)

        got = df.iloc[0]["when"]
        got_utc = got.tz_localize("UTC") if got.tzinfo is None else got.tz_convert("UTC")
        assert got_utc == original, f"{kind}: {got_utc} != {original}"
        assert pd.isna(df.iloc[1]["when"])  # NaT row untouched


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
