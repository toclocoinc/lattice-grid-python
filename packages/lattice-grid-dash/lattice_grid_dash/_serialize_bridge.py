"""DataFrame <-> ``LatticeGrid`` data-prop bridge for Dash apps.

This is a *thin* adapter over ``lattice-grid-pandas`` -- the same package the
Jupyter widget (phase C1) uses. It adds ZERO serialization logic of its own: the
dtype mapping, the columnar payload build and the edit dtype-coercion all come
from the shared package, so there is one implementation across both wrappers.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import pandas as pd
from lattice_grid_pandas import (
    ROW_KEY,
    build_columnar,
    build_columns,
    cast_to_dtype,
)


def positional_keys(n: int) -> list[str]:
    """Stable per-row keys: the row's integer position, as a string ('0'..'n-1').

    Matches the Jupyter widget's key scheme, so both wrappers behave identically.
    """
    return [str(i) for i in range(n)]


def dataframe_to_data(
    df: pd.DataFrame,
    keys: Optional[Iterable[str]] = None,
) -> dict:
    """Build the ``data`` prop for ``LatticeGrid`` from a DataFrame.

    Returns ``{"columns": [...], "rowKey": "__row_id__", "columnar": {...}}`` where
    the browser reconstructs row objects from the column-major ``columnar`` block.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("dataframe_to_data expects a pandas DataFrame")
    key_list = list(keys) if keys is not None else positional_keys(len(df))
    if len(key_list) != len(df):
        raise ValueError("keys length must match the number of rows")
    return {
        "columns": build_columns(df),
        "rowKey": ROW_KEY,
        "columnar": build_columnar(df, key_list),
    }


def apply_cell_edit(
    df: pd.DataFrame,
    edit: Optional[dict],
    keys: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Apply one ``cellChanged`` payload back into a DataFrame, dtype-preserved.

    ``edit`` is the ``{"key", "colId", "value"}`` payload the component emits.
    Returns ``df`` mutated in place (and also returned, for convenience). Edits to
    read-only index columns or unknown rows are ignored, mirroring the widget.
    """
    if not edit:
        return df
    col = edit.get("colId")
    key = edit.get("key")
    value = edit.get("value")
    if col not in df.columns:
        return df  # index columns are read-only; ignore
    key_list = list(keys) if keys is not None else positional_keys(len(df))
    try:
        pos = key_list.index(str(key))
    except ValueError:
        return df
    value = cast_to_dtype(df[col].dtype, value)
    df.iat[pos, df.columns.get_loc(col)] = value
    return df


__all__ = ["dataframe_to_data", "apply_cell_edit", "positional_keys", "ROW_KEY"]
