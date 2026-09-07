"""
LatticeGridWidget -- an editable Lattice grid over a pandas DataFrame in Jupyter.

    from lattice_grid_jupyter import LatticeGridWidget
    w = LatticeGridWidget(df)     # localhost notebook => free, unwatermarked
    w                             # renders the grid
    ...user edits cells...
    w.df                          # DataFrame reflecting the edits

Deployed on a non-localhost origin => pass a key:
    w = LatticeGridWidget(df, licence="LG-...")

Offline notebooks (no CDN at render time):
    w = LatticeGridWidget(df, offline=True)   # grid JS/CSS carried in the widget
"""

from __future__ import annotations

import pathlib
from typing import Any, Iterable

import anywidget
import pandas as pd
import traitlets

from . import _serialize as S

_HERE = pathlib.Path(__file__).parent
_STATIC = _HERE / "static"
GRID_VERSION = "1.40.0"


class LatticeGridWidget(anywidget.AnyWidget):
    _esm = _STATIC / "widget.js"

    # --- synced state ----------------------------------------------------------
    _columns = traitlets.List().tag(sync=True)
    _columnar = traitlets.Dict().tag(sync=True)
    _row_key = traitlets.Unicode(S.ROW_KEY).tag(sync=True)
    licence = traitlets.Unicode("").tag(sync=True)
    height = traitlets.Int(360).tag(sync=True)

    # bundle delivery
    _grid_source = traitlets.Unicode("cdn").tag(sync=True)   # "cdn" | "vendor"
    _grid_version = traitlets.Unicode(GRID_VERSION).tag(sync=True)
    _grid_js = traitlets.Unicode("").tag(sync=True)          # vendored UMD text
    _grid_css = traitlets.Unicode("").tag(sync=True)         # vendored CSS text

    # Python -> grid signals
    _data_version = traitlets.Int(0).tag(sync=True)
    _row_op = traitlets.Dict(allow_none=True).tag(sync=True)

    # grid -> Python
    _edit = traitlets.Dict(allow_none=True).tag(sync=True)
    _licence_state = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        df: pd.DataFrame,
        licence: str = "",
        height: int = 360,
        offline: bool = False,
        grid_version: str = GRID_VERSION,
        **kwargs: Any,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("LatticeGridWidget expects a pandas DataFrame")

        self._df = df.copy()
        # Stable, monotonic per-row keys. Never reused, never renumbered, so the
        # grid's key set and the DataFrame stay aligned across append/delete.
        self._next_id = 0
        self._keys: list[str] = self._fresh_keys(len(self._df))

        source = "vendor" if offline else "cdn"
        grid_js = grid_css = ""
        if offline:
            grid_js, grid_css = _load_vendored()

        super().__init__(
            licence=licence,
            height=height,
            _grid_source=source,
            _grid_version=grid_version,
            _grid_js=grid_js,
            _grid_css=grid_css,
            **kwargs,
        )
        self._row_key = S.ROW_KEY
        self._push_frame()
        self.observe(self._on_edit, names="_edit")

    # --- keys ------------------------------------------------------------------
    def _fresh_keys(self, n: int) -> list[str]:
        keys = [str(self._next_id + i) for i in range(n)]
        self._next_id += n
        return keys

    def _pos_of(self, key: str) -> int | None:
        try:
            return self._keys.index(str(key))
        except ValueError:
            return None

    # --- DataFrame -> grid -----------------------------------------------------
    def _push_frame(self) -> None:
        self._columns = S.build_columns(self._df)
        self._columnar = S.build_columnar(self._df, self._keys)

    @property
    def df(self) -> pd.DataFrame:
        """The live DataFrame, reflecting every edit made in the grid."""
        return self._df

    # --- grid edit -> DataFrame ------------------------------------------------
    def _on_edit(self, change: dict) -> None:
        edit = change["new"]
        if not edit:
            return
        col = edit.get("colId")
        key = edit.get("key")
        value = edit.get("value")
        if col not in self._df.columns:
            return  # index columns are read-only; ignore
        pos = self._pos_of(key)
        if pos is None:
            return
        value = S.cast_to_dtype(self._df[col].dtype, value)
        self._df.iat[pos, self._df.columns.get_loc(col)] = value

    def apply_edit(self, key: Any, col_id: str, value: Any) -> None:
        """Test/host hook: replay the exact comm payload the grid emits on an edit."""
        self._edit = {"key": str(key), "colId": col_id, "value": value}

    # --- Python -> grid: live updates -----------------------------------------
    def set_data(self, df: pd.DataFrame) -> None:
        """Replace the whole DataFrame and repaint the grid (fresh keys, full reload)."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("set_data expects a pandas DataFrame")
        self._df = df.copy()
        self._keys = self._fresh_keys(len(self._df))
        self._push_frame()
        self._data_version += 1  # -> grid.rows.load(...) in the browser

    def append_rows(self, rows: pd.DataFrame | Iterable[dict]) -> list[str]:
        """Append rows to the DataFrame and the grid (incremental). Returns new keys."""
        new = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        new = new.reindex(columns=self._df.columns)
        idx = self._df.index
        base_default = (
            isinstance(idx, pd.RangeIndex) and idx.start == 0 and idx.step == 1
        )
        if base_default:
            # Keep the combined frame a default 0..n-1 RangeIndex so no index
            # column silently appears mid-session.
            new.index = range(len(self._df), len(self._df) + len(new))
        keys = self._fresh_keys(len(new))
        self._df = pd.concat([self._df, new], axis=0, ignore_index=False)
        self._keys.extend(keys)
        # Rebuild the full columnar, then slice the tail so the appended records
        # carry EXACTLY the grid's field set (index columns included iff present).
        self._columnar = S.build_columnar(self._df, self._keys)
        tail = {f: v[-len(keys):] for f, v in self._columnar.items()}
        fields = [f for f in tail if f != S.ROW_KEY]
        records = [
            {S.ROW_KEY: keys[j], **{f: tail[f][j] for f in fields}}
            for j in range(len(keys))
        ]
        self._row_op = {"op": "add", "rows": records}  # -> grid.rows.apply({add})
        return keys

    def delete_rows(self, keys: str | Iterable[str]) -> list[str]:
        """Delete rows by their stable grid key (incremental). Returns removed keys."""
        if isinstance(keys, (str, int)):
            keys = [keys]
        keys = [str(k) for k in keys]
        positions = sorted(
            (p for p in (self._pos_of(k) for k in keys) if p is not None),
            reverse=True,
        )
        keep = [i not in set(positions) for i in range(len(self._df))]
        self._df = self._df.iloc[keep].copy()
        self._keys = [k for i, k in enumerate(self._keys) if keep[i]]
        self._columnar = S.build_columnar(self._df, self._keys)
        self._row_op = {"op": "remove", "keys": keys}  # -> grid.rows.apply({remove})
        return keys


def _load_vendored() -> tuple[str, str]:
    js = (_STATIC / "lattice-grid.min.js")
    css = (_STATIC / "lattice-grid.min.css")
    if not js.exists() or not css.exists():
        raise FileNotFoundError(
            "offline=True needs the vendored grid bundle at "
            f"{js} and {css}. Reinstall the wheel (it ships them) or use offline=False."
        )
    return js.read_text(encoding="utf-8"), css.read_text(encoding="utf-8")
