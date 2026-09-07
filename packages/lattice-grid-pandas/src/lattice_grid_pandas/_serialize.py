"""
DataFrame <-> Lattice grid serialization.

All the pandas-facing logic lives here so each host wrapper (Jupyter widget, Dash
component, ...) stays thin, and so this can be unit-tested without any UI
framework or a browser.

Design decisions
----------------
* **Row identity is positional.** ``__row_id__`` is always the row's integer
  position (as a string), never the index label. This is bulletproof for every
  index shape the task calls out -- MultiIndex, non-default, and *duplicate*
  labels -- where a label-derived key would collide and corrupt the grid's
  one-key-one-row invariant. The real index is preserved and shown as leading,
  read-only column(s) so the user still sees their labels.

* **Columnar transfer.** The frame is serialized column-major
  (``{field: [v0, v1, ...]}``) rather than as a list-of-dicts. Field names are
  not repeated per row (roughly halves the payload for wide frames) and each
  column is built with a vectorized pass instead of ``DataFrame.iterrows()``
  (which is the real reason a naive list-of-dicts build freezes at 100k rows).
  The browser reconstructs row objects in a single loop and hands them to the
  grid's memory source, whose rendering is virtualized -- so only the visible
  window is ever in the DOM.

* **NaN / NaT / NA -> null.** Every missing value becomes JSON ``null`` (Python
  ``None``), regardless of the pandas sentinel.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from pandas.api import types as pdt

# Grid column types this wrapper emits. Verified to render against the published
# bundle (see tests/test_smoke_browser.py and the spike's jsdom test).
_T_NUMBER = "number"
_T_BOOL = "boolean"
_T_TIME = "timestamp"
_T_TEXT = "text"

ROW_KEY = "__row_id__"


def lattice_type(dtype) -> str:
    """Map a pandas dtype to a Lattice grid column ``type``."""
    if pdt.is_bool_dtype(dtype):
        return _T_BOOL
    if pdt.is_integer_dtype(dtype) or pdt.is_float_dtype(dtype):
        return _T_NUMBER
    if pdt.is_datetime64_any_dtype(dtype):
        return _T_TIME
    # category, object, string, timedelta and anything else render as text.
    return _T_TEXT


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    try:
        res = pd.isna(v)
    except (TypeError, ValueError):
        return False
    # pd.isna on an array-like (e.g. a list cell) returns an array; treat as present.
    return bool(res) if isinstance(res, bool) else False


def _jsonable_scalar(v: Any) -> Any:
    """Coerce one arbitrary cell value to a JSON-carriable scalar."""
    if _is_missing(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, (str, bool)):
        return v
    # numpy scalar -> python scalar
    item = getattr(v, "item", None)
    if callable(item):
        try:
            v = v.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, float):
        return None if math.isnan(v) else v
    return str(v)


def column_values(series: pd.Series) -> list:
    """JSON-safe, missing-normalized, *vectorized* value list for one column."""
    dtype = series.dtype

    if pdt.is_datetime64_any_dtype(dtype):
        # ISO 8601 strings; NaT -> None. tz-aware keeps its offset.
        return [None if pd.isna(x) else x.isoformat() for x in series]

    if pdt.is_bool_dtype(dtype):
        # covers numpy bool and pandas nullable boolean (pd.NA -> None)
        return series.astype(object).where(series.notna(), None).tolist()

    if pdt.is_integer_dtype(dtype):
        if series.isna().any():  # nullable Int with <NA>
            return series.astype(object).where(series.notna(), None).tolist()
        return series.tolist()  # fast path, plain python ints

    if pdt.is_float_dtype(dtype):
        out = series.to_numpy(dtype="float64", na_value=float("nan"))
        return [None if (x != x) else float(x) for x in out]  # x!=x catches NaN

    # object / string / category / timedelta / mixed -> per-element coerce
    return [_jsonable_scalar(v) for v in series.tolist()]


def _index_levels(df: pd.DataFrame) -> list[tuple[str, list]]:
    """Leading columns for the index, unless it is a plain default RangeIndex.

    Returns a list of (field_name, values) pairs -- one per MultiIndex level, or
    a single pair for a flat index. Empty when the index is the default 0..n-1.
    """
    idx = df.index
    # Default RangeIndex(0..n-1, step 1) carries no information the row already lacks.
    if isinstance(idx, pd.RangeIndex) and idx.start == 0 and idx.step == 1:
        return []

    levels: list[tuple[str, list]] = []
    if isinstance(idx, pd.MultiIndex):
        for i, name in enumerate(idx.names):
            field = str(name) if name is not None else f"index_{i}"
            vals = [None if _is_missing(v) else _jsonable_scalar(v)
                    for v in idx.get_level_values(i).tolist()]
            levels.append((field, vals))
    else:
        name = idx.name if idx.name is not None else "index"
        vals = [None if _is_missing(v) else _jsonable_scalar(v) for v in idx.tolist()]
        levels.append((str(name), vals))
    return levels


def build_columns(df: pd.DataFrame) -> list[dict]:
    """Grid column definitions: read-only index column(s) then editable data columns."""
    cols: list[dict] = []
    for field, _ in _index_levels(df):
        cols.append({"field": field, "title": field, "type": _T_TEXT, "edit": False})
    for name in df.columns:
        cols.append({
            "field": str(name),
            "title": str(name),
            "type": lattice_type(df[name].dtype),
            "edit": True,
        })
    return cols


def build_columnar(df: pd.DataFrame, keys: list[str]) -> dict:
    """Column-major payload: ``{field: [values...], "__row_id__": [keys...]}``.

    Data-column field names are the *editable* fields; index fields are added as
    read-only display columns. ``keys`` are the stable per-row ids, in row order.
    """
    payload: dict[str, list] = {ROW_KEY: list(keys)}
    for field, vals in _index_levels(df):
        payload[field] = vals
    for name in df.columns:
        payload[str(name)] = column_values(df[name])
    return payload


def data_fields(df: pd.DataFrame) -> list[str]:
    """The editable data field names (excludes index columns and the row key)."""
    return [str(name) for name in df.columns]


def cast_to_dtype(dtype, value):
    """Cast an edited value back to a column's dtype, so the DataFrame stays typed.

    Best-effort: on any failure the raw value is returned unchanged.
    """
    if value is None:
        return None
    try:
        if pdt.is_bool_dtype(dtype):
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes", "t")
            return bool(value)
        if pdt.is_integer_dtype(dtype):
            return int(value)
        if pdt.is_float_dtype(dtype):
            return float(value)
        if pdt.is_datetime64_any_dtype(dtype):
            return pd.to_datetime(value)
    except (TypeError, ValueError):
        return value
    return value
