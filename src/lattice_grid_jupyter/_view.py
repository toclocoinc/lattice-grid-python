"""``widget.view`` over a large frame: a lazy handle, never ten million rows.

Below the threshold ``widget.view`` is a DataFrame (the browser reports the keys
it shows). Above it the grid holds only windows, so the view is defined by the
grid's ``state`` (filters, quick search, sort) and evaluated here, over the
whole frame, only when asked.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class LazyView:
    """The rows passing the grid's filters, in the grid's order -- unmaterialised.

    ``len(view)`` counts them, ``view.head(n)`` / ``view.slice(a, b)`` return a
    DataFrame of just those rows, ``view.to_pandas()`` materialises the whole
    view (explicitly; it may be large), and ``view.positions()`` gives the row
    positions in the source frame.
    """

    def __init__(self, engine: Any, state: dict | None):
        st = state or {}
        self._engine = engine
        self.filters = st.get("filters")
        self.quick = st.get("quick") or ""
        self.sort = st.get("sort") or []
        self._pos: np.ndarray | None = None

    def positions(self) -> np.ndarray:
        """Row positions (into the source frame) of the view, in grid order."""
        if self._pos is None:
            self._pos = self._engine.positions(self.filters, self.quick, self.sort)
        return self._pos

    def __len__(self) -> int:
        return int(len(self.positions()))

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self), len(self.columns))

    @property
    def columns(self) -> list[str]:
        return [c["field"] for c in self._engine.columns()]

    def slice(self, start: int, stop: int) -> pd.DataFrame:
        """Rows ``start:stop`` of the view as a DataFrame (original index and dtypes)."""
        return self._engine.rows_at(self.positions()[start:stop])

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.slice(0, n)

    def to_pandas(self) -> pd.DataFrame:
        """Materialise the whole view. Explicit, because it may be millions of rows."""
        return self._engine.rows_at(self.positions())

    def __repr__(self) -> str:
        total = len(self._engine)
        return (f"<LazyView {len(self):,} of {total:,} rows (lazy): "
                ".head(n), .slice(a, b), .to_pandas(), .positions()>")
