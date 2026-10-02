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

Large frames (card 1618): above ``large_threshold`` rows (default 100,000) the
widget sends no rows up front. The grid asks Python for each window as the
person scrolls, and sort, filter, quick search, group-by subtotals, filter-menu
values, header histograms and column profiles are computed here over the whole
frame (pandas; a Parquet path read by DuckDB, or a Polars frame, held as Arrow); only results
cross the comm. See ``_engine.py`` and the README's "Large data" section.
"""

from __future__ import annotations

import pathlib
import uuid
import warnings
from typing import Any, Iterable

import anywidget
import numpy as np
import pandas as pd
import traitlets

from lattice_grid_pandas import GRID_VERSION  # noqa: F401  (re-exported)

from . import _options as O
from . import _serialize as S
from ._engine import AGGREGATES, OPERATORS, PandasEngine
from ._options import LatticeGridWarning
from ._view import LazyView

#: Row count at and above which the widget switches to the windowed source.
DEFAULT_LARGE_THRESHOLD = 100_000

_COPY_ON_WRITE = int(pd.__version__.split(".")[0]) >= 3

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

    # a stable id charts bind to (card 1619): the browser registers the live grid under it
    _uid = traitlets.Unicode("").tag(sync=True)

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

    # large frames (card 1618)
    _mode = traitlets.Unicode("client").tag(sync=True)                # "client" | "windowed"
    _page_size = traitlets.Int(100).tag(sync=True)                    # rows per window request
    _engine_caps = traitlets.Dict().tag(sync=True)                    # operators / aggregates

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
        large_threshold: int = DEFAULT_LARGE_THRESHOLD,
        windowed: bool | None = None,
        page_size: int = 100,
        **kwargs: Any,
    ):
        self.large_threshold = int(large_threshold)
        self._force_windowed = windowed
        self._engine = None
        frame, engine = self._adopt(df)
        self._df = frame
        # Stable, monotonic per-row keys. Never reused, never renumbered, so the
        # grid's key set and the DataFrame stay aligned across append/delete.
        # (Windowed: the key IS the row's position; no key list is held.)
        self._next_id = 0
        self._keys: list[str] = []
        self._engine = engine
        if not self._windowed:
            self._keys = self._fresh_keys(len(self._df))

        source = "vendor" if offline else "cdn"
        grid_js = grid_css = ""
        if offline:
            grid_js, grid_css = _load_vendored()

        self._column_overrides = list(columns) if columns else []
        self._flags = self._flag_options(profile, histograms, pivot)
        self._flag_state = self._flag_state_for(pivot)
        if self._column_overrides:  # validate early, name what is wrong
            O.merge_columns(self._built_columns(), self._column_overrides)
        eff = O.grid_options(options, self._flags)
        seed_state = {**self._flag_state, **(state or {})}

        super().__init__(
            _uid=uuid.uuid4().hex,
            licence=licence,
            height=height,
            _grid_source=source,
            _grid_version=grid_version,
            _grid_js=grid_js,
            _grid_css=grid_css,
            _grid_options=eff,
            options=dict(options or {}),
            state=seed_state,
            _page_size=int(page_size),
            **kwargs,
        )
        self._row_key = S.ROW_KEY
        self._push_frame()
        self.on_msg(self._on_custom)
        self.observe(self._on_state_for_view, names="state")
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

    # --- large frames: which mode, which engine -------------------------------
    def _adopt(self, data: Any):
        """Accept a pandas frame, a Polars frame or a Parquet path.

        Returns ``(pandas_frame_or_None, engine_or_None)``. A Polars frame or a
        Parquet path (read by DuckDB) is held as an Arrow table and is never
        copied into pandas when it is above the threshold.
        """
        if isinstance(data, pd.DataFrame):
            n = len(data)
            if self._decide_windowed(n):
                # pandas >= 3 is copy-on-write: a shallow copy costs nothing and an
                # edit still never reaches the caller's frame. Older pandas: copy.
                frame = data.copy(deep=not _COPY_ON_WRITE)
                return frame, PandasEngine(frame)
            return data.copy(), None
        from ._arrow_engine import ArrowEngine, is_parquet_path, is_polars_frame
        if is_polars_frame(data) or is_parquet_path(data):
            engine = ArrowEngine(data)
            if self._decide_windowed(len(engine)):
                return None, engine
            return engine.to_pandas(), None
        raise TypeError("LatticeGridWidget expects a pandas DataFrame, a Polars DataFrame "
                        "or the path of a Parquet file")

    def _decide_windowed(self, n: int) -> bool:
        if self._force_windowed is not None:
            return bool(self._force_windowed)
        return n >= self.large_threshold

    @property
    def _windowed(self) -> bool:
        return self._engine is not None

    def _built_columns(self) -> list[dict]:
        if self._windowed:
            return self._engine.columns()
        return S.build_columns(self._df)

    def _caps(self) -> dict:
        return {
            "operators": list(OPERATORS),
            "aggregates": list(AGGREGATES),
            "editable": bool(self._windowed and getattr(self._engine, "editable", False)),
        }

    # --- DataFrame -> grid -----------------------------------------------------
    def _push_frame(self) -> None:
        self._columns = O.merge_columns(self._built_columns(), self._column_overrides)
        if self._windowed:
            # No rows up front: the grid asks for each window as it scrolls.
            self._engine_caps = self._caps()
            self._columnar = {}
            self._mode = "windowed"
        else:
            self._columnar = S.build_columnar(self._df, self._keys)
            self._mode = "client"

    @property
    def df(self) -> pd.DataFrame:
        """The live DataFrame, reflecting every edit made in the grid.

        Over a Parquet path or a Polars frame above the threshold there is no
        pandas frame (the data stays where it is): use ``view`` instead."""
        if self._df is None:
            raise AttributeError(
                "this widget is backed by a Parquet file or a Polars frame (held as Arrow), "
                "so there is no pandas DataFrame to return; use widget.view (lazy) instead")
        return self._df

    @property
    def windowed(self) -> bool:
        """True when the grid is a windowed view answered by Python (large data)."""
        return self._windowed

    # --- comm: the grid's pushdown queries ---------------------------------------
    def _on_custom(self, _widget: Any, content: Any, _buffers: Any = None) -> None:
        if not isinstance(content, dict) or content.get("type") != "lg:req":
            return
        self.send(self._answer(content))

    def _answer(self, req: dict) -> dict:
        """Answer one request from the browser: ``{type: lg:res, id, ok, result|error}``."""
        rid = req.get("id")
        try:
            result = self._dispatch(str(req.get("method")), req.get("payload") or {})
            return {"type": "lg:res", "id": rid, "ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 -- surfaced to the grid as source:error
            return {"type": "lg:res", "id": rid, "ok": False, "error": f"[lattice] {type(exc).__name__}: {exc}"}

    def _dispatch(self, method: str, p: dict) -> Any:
        e = self._engine
        if e is None:
            raise RuntimeError("the grid asked for a window but this widget holds its rows client-side")
        if method == "execute":
            return e.execute(p.get("query") or {})
        if method == "grouped":
            return e.grouped_aggregates(p.get("query") or {}, p.get("groupBy") or [], p.get("aggregates") or [])
        if method == "aggregates":
            return e.aggregates(p.get("query") or {}, p.get("aggregates") or [])
        if method == "facet":
            return e.facet(p)
        if method == "profile":
            return e.profile(str(p.get("colId")), p.get("filters"), p.get("quick") or "")
        if method == "mutate":
            return self._mutate(p.get("op") or {})
        raise ValueError(f"unknown request {method!r}")

    def _mutate(self, op: dict) -> dict:
        """A cell edit over the windowed source (``adapter.mutate``, update only)."""
        if op.get("kind") != "update" or not getattr(self._engine, "editable", False):
            return {"ok": False, "reason": "this data source is read-only"}
        pos = int(op.get("key"))
        for col, value in (op.get("patch") or {}).items():
            if col not in map(str, self._df.columns):
                return {"ok": False, "reason": f"{col!r} is read-only"}
            label = next(c for c in self._df.columns if str(c) == col)
            self._df.iat[pos, self._df.columns.get_loc(label)] = S.cast_to_dtype(self._df[label].dtype, value)
        self._engine.invalidate()
        return {"ok": True}

    # --- selection / filtered view (grid -> Python, keys only) -----------------
    def _frame_for(self, keys: Iterable[str] | None) -> pd.DataFrame:
        """Rows of the live frame for these keys, in the given order; keeps the
        original index and dtypes. Keys no longer present are dropped."""
        if self._windowed:
            n = len(self._engine)
            pos = [int(k) for k in (keys or []) if str(k).isdigit() and int(k) < n]
            return self._engine.rows_at(pos)
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
        if self._windowed:
            return self._frame_for(sorted(chosen, key=lambda k: int(k) if k.isdigit() else -1))
        keys = [k for k in self._keys if k in chosen]
        return self._frame_for(keys)

    @property
    def view(self):
        """The rows passing the grid's filters (quick search included), in the
        grid's sort order. Before the grid reports, every row.

        Large data (windowed): a :class:`LazyView` -- nothing is materialised
        until you ask (``len(view)``, ``view.head(n)``, ``view.to_pandas()``)."""
        if self._windowed:
            return LazyView(self._engine, self.state)
        return self._frame_for(self._view_keys)

    def _on_state_for_view(self, change: dict) -> None:
        # Windowed: the view is defined by the state, so a state change that
        # moves the filter, quick search or sort is a view change.
        if not self._windowed or not self._has_observers("view"):
            return
        pick = lambda st: {k: (st or {}).get(k) for k in ("filters", "quick", "sort")}  # noqa: E731
        if pick(change.get("old")) != pick(change.get("new")):
            self._notify_frame("view", self.view)

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
        frame, engine = self._adopt(df)
        self._df = frame
        self._engine = engine
        # Crossing the threshold either way switches the grid's source.
        self._keys = [] if self._windowed else self._fresh_keys(len(self._df))
        self._push_frame()
        self._data_version += 1  # -> grid.rows.load(...) / a fresh windowed source

    def append_rows(self, rows: pd.DataFrame | Iterable[dict]) -> list[str]:
        """Append rows to the DataFrame and the grid (incremental). Returns new keys."""
        new = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        new = new.reindex(columns=self.df.columns)
        if self._windowed:
            return self._windowed_replace(pd.concat([self._df, new], axis=0, ignore_index=isinstance(
                self._df.index, pd.RangeIndex)), first_new=len(self._df))
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
        if self._windowed:
            n = len(self._df)
            keep = np.ones(n, dtype=bool)
            keep[[int(k) for k in keys if k.isdigit() and int(k) < n]] = False
            self._windowed_replace(self._df.iloc[keep])
            return keys
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


    def _windowed_replace(self, frame: pd.DataFrame, first_new: int | None = None) -> list[str]:
        """Windowed append/delete: a new frame, a new engine, a fresh source."""
        self._df = frame
        self._engine = PandasEngine(frame)
        self._selected_keys = []
        self._push_frame()
        self._data_version += 1
        if first_new is None:
            return []
        return [str(i) for i in range(first_new, len(frame))]


def _load_vendored() -> tuple[str, str]:
    js = (_STATIC / "lattice-grid.min.js")
    css = (_STATIC / "lattice-grid.min.css")
    if not js.exists() or not css.exists():
        raise FileNotFoundError(
            "offline=True needs the vendored grid bundle at "
            f"{js} and {css}. Reinstall the wheel (it ships them) or use offline=False."
        )
    return js.read_text(encoding="utf-8"), css.read_text(encoding="utf-8")
