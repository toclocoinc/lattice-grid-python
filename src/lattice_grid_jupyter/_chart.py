"""
LatticeChart -- a chart viewer bound to a Lattice grid in Jupyter (card 1619).

    from lattice_grid_jupyter import LatticeGridWidget, LatticeChart
    from ipywidgets import HBox

    grid = LatticeGridWidget(df)
    roc = LatticeChart(grid, type="roc", label="churned", score="p")
    HBox([grid, roc])           # filter the grid: the chart redraws in the browser

Every view is a viewer of a dataset. The chart is bound to the grid's own dataset
through the grid's own binding, so it follows the grid's filters, quick search and
selection live in the browser: no message goes to Python when you filter. A chart
can also take a DataFrame directly (``LatticeChart(df, type="scatter", ...)``).

The chart spec is plain data: ``snake_case`` keys are mapped to the grid's
camelCase, and a key the grid's ``ChartSpec`` does not declare is reported by name
(``config.unknown:<key>``, the grid's own warning id) and dropped. Functions are
refused: they cannot cross to the browser.

Chart code loads only when used: the base chart module plus the one module the
type needs (``roc`` -> ``chart-roc``), never a bundle with every chart.

Large (windowed) grids: the browser holds a window, not the rows. A chart over a
windowed grid draws from the grouped aggregates Python already computes for the
grid (``bar``, ``line``, ``pie`` ... with ``x`` and ``y``) and refuses, with a
named warning, every type that needs the rows themselves.
"""

from __future__ import annotations

import pathlib
import re
import warnings
from typing import Any

import anywidget
import pandas as pd
import traitlets

from lattice_grid_pandas import GRID_VERSION

from . import _options as O
from . import _serialize as S
from ._chart_names import CORE_TYPES, SPEC_KEYS
from ._options import LatticeGridWarning
from .widget import DEFAULT_LARGE_THRESHOLD, LatticeGridWidget

_STATIC = pathlib.Path(__file__).parent / "static"
_CHARTS = _STATIC / "charts"

#: Extension chart types shipped as their own module, with the spec keys each reads
#: beyond the grid's ``ChartSpec`` (from the modules' own documentation).
EXTENSION_TYPES: dict[str, frozenset[str]] = {
    "roc": frozenset({"label", "score", "curve", "positive", "bins"}),
    "fan": frozenset({"x", "y", "forecast", "lower", "upper"}),
    "decomposition": frozenset({"x", "observed", "trend", "seasonal", "residual"}),
    "splom": frozenset({"columns"}),
    "hexbin": frozenset({"x", "y", "radius"}),
    "ridgeline": frozenset({"x", "y", "overlap"}),
}

#: The grouped types a windowed grid's pushed aggregates can draw: one value per
#: category (per series) from ``x`` and ``y``.
WINDOWED_TYPES = frozenset({
    "bar", "horizontalBar", "line", "step", "area", "pie", "donut", "funnel", "waterfall",
})

#: Spec keys the chart reads that the grid's ``ChartSpec`` declarations omit (F-1619-1).
UNDECLARED_KEYS = frozenset({"from"})

#: Python owns these: the grid / data and the element the chart draws in.
RESERVED = ("grid", "container", "rows")

_SNAKE = re.compile(r"_([a-z0-9])")


def supported_types() -> list[str]:
    """Every chart type ``LatticeChart`` draws: the grid's core types and the shipped extensions."""
    return sorted(CORE_TYPES | set(EXTENSION_TYPES))


def map_spec(chart_type: str, user: dict | None) -> dict:
    """The effective chart spec: snake_case mapped, unknown keys reported by name.

    A key is accepted when it (or its camelCase form) is a property of the grid's
    ``ChartSpec`` or one the chart type's own module documents. Anything else is
    warned about as ``config.unknown:<key>`` and left out.
    """
    known = SPEC_KEYS | UNDECLARED_KEYS | EXTENSION_TYPES.get(chart_type, frozenset())
    out: dict = {}
    for key, value in dict(user or {}).items():
        if key in RESERVED:
            warnings.warn(
                f"[lattice] chart option {key!r} is supplied by the widget (the bound grid or DataFrame) "
                "and was ignored", LatticeGridWarning, stacklevel=4)
            continue
        camel = _SNAKE.sub(lambda m: m.group(1).upper(), key) if "_" in key and not key.startswith("_") else key
        if key in known:
            name = key
        elif camel in known:
            name = camel
        else:
            warnings.warn(
                f"[lattice] config.unknown:{key}: {key!r} is not a key a {chart_type!r} chart recognises "
                "and was ignored (check the spelling; snake_case is accepted)",
                LatticeGridWarning, stacklevel=4)
            continue
        out[name] = O.map_keys(value, name)
    return out


def _module_names(chart_type: str) -> list[str]:
    return [f"chart-{chart_type}"] if chart_type in EXTENSION_TYPES else []


def _read_modules(names: list[str]) -> dict[str, str]:
    """The vendored UMD text of the base chart module and each named type module."""
    out = {}
    for name in ["charts", *names]:
        path = _CHARTS / f"{name}.min.js"
        if not path.exists():
            raise FileNotFoundError(
                f"offline=True needs the vendored chart module {path}. Reinstall the wheel "
                "(it ships them) or use offline=False.")
        out[name] = path.read_text(encoding="utf-8")
    return out


class LatticeChart(anywidget.AnyWidget):
    """A chart viewer of a grid's dataset (or of a DataFrame). See the module docs."""

    _esm = _STATIC / "chart.js"

    type = traitlets.Unicode("bar").tag(sync=True)
    """The chart type. Assign to change it live (its module loads on demand)."""
    height = traitlets.Int(320).tag(sync=True)
    licence = traitlets.Unicode("").tag(sync=True)
    spec = traitlets.Dict()
    """The chart spec as plain data (snake_case accepted). Assigning redraws the chart."""
    _spec = traitlets.Dict().tag(sync=True)

    _grid_uid = traitlets.Unicode("").tag(sync=True)        # the bound grid widget's uid
    _mode = traitlets.Unicode("grid").tag(sync=True)        # "grid" | "frame"
    _windowed = traitlets.Bool(False).tag(sync=True)        # the bound grid holds a window
    _refused = traitlets.Unicode("").tag(sync=True)         # why nothing is drawn
    _columns = traitlets.List().tag(sync=True)              # frame mode only
    _columnar = traitlets.Dict().tag(sync=True)
    _row_key = traitlets.Unicode(S.ROW_KEY).tag(sync=True)
    _data_version = traitlets.Int(0).tag(sync=True)

    _grid_source = traitlets.Unicode("cdn").tag(sync=True)  # "cdn" | "vendor"
    _grid_version = traitlets.Unicode(GRID_VERSION).tag(sync=True)
    _grid_js = traitlets.Unicode("").tag(sync=True)         # frame mode + vendor only
    _grid_css = traitlets.Unicode("").tag(sync=True)
    _modules = traitlets.List().tag(sync=True)              # type modules this chart needs
    _chart_js = traitlets.Dict().tag(sync=True)             # vendor mode: {module: UMD text}

    _draws = traitlets.Int(0).tag(sync=True)                # browser -> Python: draws so far
    _chart_error = traitlets.Unicode("").tag(sync=True)     # browser -> Python: a load / draw failure

    def __init__(
        self,
        grid: LatticeGridWidget | pd.DataFrame,
        type: str = "bar",  # noqa: A002 -- the chart's own name for it
        *,
        height: int = 320,
        offline: bool | None = None,
        licence: str | None = None,
        grid_version: str = GRID_VERSION,
        **spec: Any,
    ):
        if type not in CORE_TYPES and type not in EXTENSION_TYPES:
            raise ValueError(f"unknown chart type {type!r}; use one of {supported_types()}")
        self._bound: LatticeGridWidget | None = None
        self._frame: pd.DataFrame | None = None
        self._keys: list[str] = []
        self._next_id = 0
        if isinstance(grid, LatticeGridWidget):
            self._bound = grid
            mode, uid = "grid", grid._uid
            vendor = grid._grid_source == "vendor" if offline is None else bool(offline)
            licence = grid.licence if licence is None else licence
        elif isinstance(grid, pd.DataFrame):
            self._frame = grid.copy()
            self._keys = self._fresh_keys(len(grid))
            mode, uid = "frame", ""
            vendor = bool(offline)
        else:
            raise TypeError("LatticeChart takes a LatticeGridWidget or a pandas DataFrame")
        grid_js = grid_css = ""
        if vendor and mode == "frame":  # the grid builds the hidden dataset; grid mode borrows the live grid
            from .widget import _load_vendored
            grid_js, grid_css = _load_vendored()

        super().__init__(
            type=type, height=int(height), licence=licence or "", _grid_uid=uid, _mode=mode,
            _grid_source="vendor" if vendor else "cdn", _grid_version=grid_version,
            _grid_js=grid_js, _grid_css=grid_css,
        )
        self.spec = dict(spec)  # runs the observer below
        self._sync_all()
        if mode == "frame":
            self._push_frame()
        self.observe(self._on_spec, names="spec")
        self.observe(self._on_type, names="type")

    # --- the spec, the type, what must load -------------------------------------
    def _sync_all(self) -> None:
        self._spec = map_spec(self.type, self.spec)
        names = _module_names(self.type)
        self._modules = names
        self._chart_js = _read_modules(names) if self._grid_source == "vendor" else {}
        self._windowed = bool(self._bound is not None and self._bound.windowed)
        self._refused = self._refusal()
        if self._refused:
            warnings.warn(f"[lattice] chart.windowed.refused:{self.type}: {self._refused}",
                          LatticeGridWarning, stacklevel=4)

    def _refusal(self) -> str:
        """Why this chart draws nothing (a windowed grid, or a frame too large to hold), or ``''``."""
        if self._frame is not None and len(self._frame) >= DEFAULT_LARGE_THRESHOLD:
            return (f"a DataFrame of {len(self._frame):,} rows is too large to chart directly: show it in a "
                    "windowed LatticeGridWidget and bind the chart to that grid")
        if not self._windowed:
            return ""
        n = len(self._bound._engine) if self._bound is not None and self._bound._engine is not None else 0
        if self.type not in WINDOWED_TYPES:
            return (f"the grid is windowed ({n:,} rows stay in Python; the browser holds one window), so a "
                    f"{self.type!r} chart, which needs the rows themselves, is not drawn rather than drawn "
                    f"from a window. Windowed grids chart {sorted(WINDOWED_TYPES)} from grouped aggregates.")
        spec = self._spec
        if not (isinstance(spec.get("x"), str) and isinstance(spec.get("y"), str)):
            return ("a chart over a windowed grid is drawn from grouped aggregates and needs `x` (the "
                    "category column) and `y` (the measure column)")
        return ""

    def update(self, type: str | None = None, **spec: Any) -> None:  # noqa: A002
        """Change the type and replace the spec in one step (one redraw, no half-way warnings).

        ``chart.update(type="roc", label="churned", score="p")`` -- the spec given here
        REPLACES the old one; the new type's module loads if it is not loaded yet."""
        new_type = self.type if type is None else type
        if new_type not in CORE_TYPES and new_type not in EXTENSION_TYPES:
            raise ValueError(f"unknown chart type {new_type!r}; use one of {supported_types()}")
        self._batching = True
        try:
            self.type = new_type
            self.spec = dict(spec)
        finally:
            self._batching = False
        self._sync_all()

    _batching = False

    def _on_spec(self, change: dict) -> None:
        if not self._batching:
            self._sync_all()

    def _on_type(self, change: dict) -> None:
        if self._batching:
            return
        if change["new"] not in CORE_TYPES and change["new"] not in EXTENSION_TYPES:
            self.type = change["old"]
            raise ValueError(f"unknown chart type {change['new']!r}; use one of {supported_types()}")
        self._sync_all()

    # --- DataFrame input ----------------------------------------------------------
    def _fresh_keys(self, n: int) -> list[str]:
        keys = [str(self._next_id + i) for i in range(n)]
        self._next_id += n
        return keys

    def _push_frame(self) -> None:
        self._columns = S.build_columns(self._frame)
        self._columnar = S.build_columnar(self._frame, self._keys) if len(self._frame) < DEFAULT_LARGE_THRESHOLD else {}

    def set_data(self, df: pd.DataFrame) -> None:
        """Replace a DataFrame-backed chart's data and redraw."""
        if self._frame is None:
            raise TypeError("this chart is bound to a grid: change the grid's data, not the chart's")
        self._frame = df.copy()
        self._keys = self._fresh_keys(len(df))
        self._refused = self._refusal()
        self._push_frame()
        self._data_version += 1

    @property
    def draws(self) -> int:
        """How many times the browser has drawn this chart (a filter in the bound grid adds one)."""
        return int(self._draws)
