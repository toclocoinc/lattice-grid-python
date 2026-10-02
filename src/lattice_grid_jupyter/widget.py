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

Datetime columns are UTC end to end: a naive ``datetime64`` column is treated
as already being UTC (not the machine's local zone), and a tz-aware column is
normalized to UTC on the wire and converted back to its own zone on the way
in, so an edit round-trips to the exact original instant for every column
kind -- see the ``lattice-grid-pandas`` README for the full contract.
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Any, Iterable

import anywidget
import pandas as pd
import traitlets

from lattice_grid_pandas import GRID_VERSION  # noqa: F401  (re-exported)

from . import _options as O
from . import _serialize as S
from ._options import LatticeGridWarning

_HERE = pathlib.Path(__file__).parent
_STATIC = _HERE / "static"


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
    _selected_keys = traitlets.List().tag(sync=True)                  # keys only
    _view_keys = traitlets.List(allow_none=True, default_value=None).tag(sync=True)
    _grid_warnings = traitlets.List().tag(sync=True)                  # [{id, message}]

    # options / state, both directions
    _grid_options = traitlets.Dict().tag(sync=True)                   # camelCase, effective
    state = traitlets.Dict().tag(sync=True)
    """The grid's filter / sort / column / group / pivot state as a plain dict.
    Read it after the user changes the view; assign a dict to restore one."""
    options = traitlets.Dict()
    """Grid options as plain dicts (snake_case accepted). Assigning updates the
    live grid without re-sending the data."""

    def __init__(
        self,
        df: pd.DataFrame,
        licence: str = "",
        height: int = 360,
        offline: bool = False,
        grid_version: str = GRID_VERSION,
        options: dict | None = None,
        columns: list[dict] | None = None,
        profile: bool = False,
        histograms: bool = False,
        pivot: Any = None,
        state: dict | None = None,
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

        self._column_overrides = list(columns) if columns else []
        self._flags = self._flag_options(profile, histograms, pivot)
        self._flag_state = self._flag_state_for(pivot)
        if self._column_overrides:  # validate early, name what is wrong
            O.merge_columns(S.build_columns(self._df), self._column_overrides)
        eff = O.grid_options(options, self._flags)
        seed_state = {**self._flag_state, **(state or {})}

        super().__init__(
            licence=licence,
            height=height,
            _grid_source=source,
            _grid_version=grid_version,
            _grid_js=grid_js,
            _grid_css=grid_css,
            _grid_options=eff,
            options=dict(options or {}),
            state=seed_state,
            **kwargs,
        )
        self._row_key = S.ROW_KEY
        self._push_frame()
        self.observe(self._on_edit, names="_edit")
        self.observe(self._on_options, names="options")
        self.observe(self._on_grid_warnings, names="_grid_warnings")
        self.observe(self._on_selected_keys, names="_selected_keys")
        self.observe(self._on_view_keys, names="_view_keys")

    # --- convenience flags -----------------------------------------------------
    @staticmethod
    def _flag_options(profile: bool, histograms: bool, pivot: Any) -> dict:
        flags: dict = {}
        if profile:
            # the statistics tool panel: per-column figures + a histogram of its shape
            flags["toolPanel"] = {"panels": ["columns", "statistics"]}
        if histograms:
            flags["facets"] = {"enabled": True}  # header histograms
        flags.update(O.pivot_options(pivot))
        return flags

    @staticmethod
    def _flag_state_for(pivot: Any) -> dict:
        fields = [pivot] if isinstance(pivot, str) else (
            list(pivot) if isinstance(pivot, (list, tuple)) else [])
        return {"version": 1, "pivot": {"enabled": True, "columns": [str(f) for f in fields]}} if fields else {}

    def _on_options(self, change: dict) -> None:
        # push only the options trait; the data traits are not touched
        self._grid_options = O.grid_options(change["new"], self._flags)

    def _on_grid_warnings(self, change: dict) -> None:
        for w in change["new"] or []:
            warnings.warn(f"[lattice] {w.get('id')}: {w.get('message')}",
                          LatticeGridWarning, stacklevel=2)

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
        self._columns = O.merge_columns(S.build_columns(self._df), self._column_overrides)
        self._columnar = S.build_columnar(self._df, self._keys)

    @property
    def df(self) -> pd.DataFrame:
        """The live DataFrame, reflecting every edit made in the grid."""
        return self._df

    # --- selection / filtered view (grid -> Python, keys only) -----------------
    def _frame_for(self, keys: Iterable[str] | None) -> pd.DataFrame:
        """Rows of the live frame for these keys, in the given order; keeps the
        original index and dtypes. Keys no longer present are dropped."""
        if keys is None:
            return self._df
        pos = {k: i for i, k in enumerate(self._keys)}
        idx = [pos[str(k)] for k in keys if str(k) in pos]
        return self._df.iloc[idx]

    @property
    def selected(self) -> pd.DataFrame:
        """The selected rows as a DataFrame (original index and dtypes), in frame
        order. An empty frame -- never None -- when nothing is selected."""
        chosen = set(map(str, self._selected_keys))
        keys = [k for k in self._keys if k in chosen]
        return self._frame_for(keys)

    @property
    def view(self) -> pd.DataFrame:
        """The rows passing the grid's filters (quick search included), in the
        grid's sort order. Before the grid reports, every row."""
        return self._frame_for(self._view_keys)

    def _has_observers(self, name: str) -> bool:
        n = self._trait_notifiers.get(name, {}).get("change")
        a = self._trait_notifiers.get(traitlets.All, {}).get("change")
        return bool(n or a)

    def _notify_frame(self, name: str, new: pd.DataFrame) -> None:
        if self._has_observers(name):
            self.notify_change(traitlets.Bunch(
                name=name, old=None, new=new, owner=self, type="change"))

    def _on_selected_keys(self, change: dict) -> None:
        if self._has_observers("selected"):
            self._notify_frame("selected", self.selected)

    def _on_view_keys(self, change: dict) -> None:
        if self._has_observers("view"):
            self._notify_frame("view", self.view)

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
