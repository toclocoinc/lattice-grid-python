"""Large data that is not a pandas frame: a Parquet path or a Polars frame (card 1618).

The data is held as one Arrow table -- a Parquet file is read by DuckDB's
parallel Parquet reader (``duckdb.read_parquet(path).arrow()``), a Polars frame
is handed over without a copy (``to_arrow()``) -- and nothing is converted to
pandas up front. A column becomes a pandas Series only when a query touches it
(a sort key, a filter column, a histogram), and a window is ``table.take(rows)``
of the hundred rows on screen.

Every query is answered by the same kernels as :class:`PandasEngine` (one
implementation of the grid's filter / sort / histogram / profile semantics, so
the three inputs cannot disagree with each other or with the grid). Read-only:
there is no pandas frame to write an edit into.

Needs ``duckdb`` (Parquet) and ``pyarrow``: ``pip install "lattice-grid-jupyter[large]"``.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Iterable

import numpy as np
import pandas as pd

from lattice_grid_pandas._serialize import build_columnar, build_columns

from ._engine import PandasEngine, _Col


def is_polars_frame(data: Any) -> bool:
    """A ``polars.DataFrame`` (checked by type name: polars is optional)."""
    t = type(data)
    return t.__name__ == "DataFrame" and t.__module__.split(".")[0] == "polars"


def is_parquet_path(data: Any) -> bool:
    """A ``str`` / ``PathLike`` naming a ``.parquet`` file."""
    if not isinstance(data, (str, os.PathLike)):
        return False
    return str(data).lower().endswith((".parquet", ".pq"))


def _need(module: str):
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"[lattice] large data from a Parquet file or a Polars frame needs {module!r}: "
            'pip install "lattice-grid-jupyter[large]"') from exc


def load_table(data: Any):
    """The Arrow table behind a Polars frame or a Parquet path."""
    _need("pyarrow")
    if is_polars_frame(data):
        return data.to_arrow()
    path = pathlib.Path(data)
    if not path.exists():
        raise FileNotFoundError(f"[lattice] no Parquet file at {path}")
    duckdb = _need("duckdb")
    con = duckdb.connect()
    try:
        return con.read_parquet(str(path)).arrow()
    finally:
        con.close()


class ArrowEngine(PandasEngine):
    """The pandas engine's queries over an Arrow table, column by column."""

    name = "arrow"
    editable = False

    def __init__(self, data: Any):
        self.table = load_table(data)
        if hasattr(self.table, "read_all"):  # a RecordBatchReader
            self.table = self.table.read_all()
        self.source = "polars" if is_polars_frame(data) else "parquet"
        self._schema_frame = self.table.slice(0, 0).to_pandas(ignore_metadata=True)
        super().__init__(self._schema_frame)
        self._levels, self._show_index = [], False

    def __len__(self) -> int:
        return int(self.table.num_rows)

    def columns(self) -> list[dict]:
        # read-only: an edit would have nowhere to go
        return [{**c, "edit": False} for c in build_columns(self._schema_frame)]

    def fields(self) -> list[str]:
        return [str(c) for c in self._schema_frame.columns]

    def col(self, field: str) -> _Col:
        if field in self._cols:
            return self._cols[field]
        names = [str(n) for n in self.table.column_names]
        if field not in names:
            raise KeyError(f"[lattice] no column {field!r} in the data")
        series = self.table.column(names.index(field)).to_pandas()
        c = _Col(field, series)
        self._cols[field] = c
        return c

    def rows_at(self, positions: Iterable[int]) -> pd.DataFrame:
        idx = np.asarray(list(positions) if not isinstance(positions, np.ndarray) else positions,
                         dtype=np.int64)
        return self.table.take(idx).to_pandas(ignore_metadata=True)

    def window(self, positions: np.ndarray) -> dict:
        part = self.rows_at(positions)
        return build_columnar(part, [str(int(p)) for p in positions])

    def to_pandas(self) -> pd.DataFrame:
        return self.table.to_pandas(ignore_metadata=True)
