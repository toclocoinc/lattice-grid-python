"""Large-frame engine: the grid's pushdown queries answered in Python (card 1618).

Above the widget's threshold the browser holds no data. It holds a *pushdown
source* whose adapter forwards every query over the widget comm, and this module
answers it over the WHOLE frame:

* ``execute``                  -- a row window, with the filter, quick search and
                                  sort applied, plus the matching-row total;
* ``grouped_aggregates``       -- group-by subtotals (and the distinct values a
                                  filter menu lists: a group-by with a count);
* ``aggregates``               -- whole-set statistics;
* ``facet``                    -- header-histogram buckets and counts;
* ``profile``                  -- the Statistics panel's per-column profile.

Only results cross the comm: a window of rows, a list of groups, a few numbers.

Semantics follow the grid's own client-side kernels (``compute/filter.js``,
``compute/sort.js``, ``compute/facet.js``, ``compute/statistics.js``) so that the
same frame answers the same way below the threshold (client) and above it
(here); ``tests/test_large_equivalence.py`` holds the two paths together.

Two engines share one interface:

* :class:`PandasEngine` -- a pandas DataFrame, computed with numpy;
* :class:`DuckDBEngine` -- (``_duckdb_engine.py``) a Parquet path or a Polars
  frame, queried through DuckDB.
"""

from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from lattice_grid_pandas._serialize import (
    ROW_KEY,
    _index_levels,
    _jsonable_scalar,
    build_columnar,
    build_columns,
)

# The comparison operators answered here (declared to the grid as the adapter's
# ``capabilities.operators``; anything else stays with the grid's planner).
OPERATORS = (
    "eq", "ne", "gt", "gte", "lt", "lte", "between", "notBetween", "in", "notIn",
    "contains", "notContains", "startsWith", "endsWith", "matches", "blank", "notBlank",
)

# The statistics answered here, by the grid's own names (``STAT_PUSHDOWN``).
AGGREGATES = (
    "count", "countValues", "sum", "avg", "min", "max", "range", "distinct", "median",
    "p25", "p75", "p90", "p95", "p99", "iqr", "variance", "varianceP", "stddev",
    "stddevP", "sumSquares",
)

DEFAULT_BUCKETS = 20
DEFAULT_CARDINALITY_LIMIT = 50
QUANTILE_SAMPLE = 10_000
PROFILE_TOP_VALUES = 5
_CACHE_SIZE = 8


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def query_key(*parts: Any) -> str:
    """A stable cache key for a query fragment."""
    return json.dumps(parts, sort_keys=True, default=str)


def is_group(node: Any) -> bool:
    """A filter node with children (``{op, conditions}``), as the grid tests it."""
    return isinstance(node, dict) and isinstance(node.get("conditions"), list)


def js_number_text(x: float) -> str:
    """``String(x)`` as JavaScript writes a finite number (``1`` not ``1.0``)."""
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if float(x).is_integer() and abs(x) < 1e21:
        return str(int(x))
    r = repr(float(x))
    if "e" in r:  # 1e-07 -> 1e-7, 1e+22 -> 1e+22
        mant, exp = r.split("e")
        sign = "-" if exp.startswith("-") else "+"
        r = f"{mant}e{sign}{int(exp.lstrip('+-'))}"
    return r


def _num_target(v: Any) -> float:
    """``Number(v)`` for a filter operand: NaN when it is not a number."""
    if v is None or isinstance(v, bool):
        return float("nan") if v is None else float(v)
    if isinstance(v, (int, float)):
        return float(v)
    try:
        s = str(v).strip()
        return float(s) if s else float("nan")
    except ValueError:
        return float("nan")


def _ts_target(v: Any) -> pd.Timestamp | None:
    """A timestamp operand: ISO text or epoch milliseconds."""
    if v is None or v == "":
        return None
    try:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return pd.Timestamp(int(v), unit="ms")
        return pd.Timestamp(str(v))
    except (ValueError, TypeError):
        return None


class _Col:
    """One field of the frame, read the ways the kernels need it.

    ``kind`` is ``number``, ``bool``, ``datetime``, ``category`` or ``text``.
    """

    def __init__(self, name: str, series: pd.Series):
        self.name = name
        s = series.reset_index(drop=True)
        self.series = s
        dt = s.dtype
        if pdt.is_bool_dtype(dt):
            self.kind = "bool"
        elif pdt.is_numeric_dtype(dt):
            self.kind = "number"
        elif pdt.is_datetime64_any_dtype(dt):
            self.kind = "datetime"
        elif isinstance(dt, pd.CategoricalDtype):
            self.kind = "category"
        else:
            self.kind = "text"
        self._missing = None
        self._float = None

    def __len__(self) -> int:
        return len(self.series)

    # --- missing / blank -----------------------------------------------------
    @property
    def missing(self) -> np.ndarray:
        """Null, NaN, NaT, NA -- what the grid receives as ``null``."""
        if self._missing is None:
            self._missing = self.series.isna().to_numpy(dtype=bool)
        return self._missing

    def blank(self) -> np.ndarray:
        """``isBlank``: missing, or the empty string."""
        m = self.missing.copy()
        if self.kind in ("text", "category"):
            m |= self.raw_text(lower=False) == ""
        return m

    # --- numbers -------------------------------------------------------------
    def floats(self) -> np.ndarray:
        """Numeric view, NaN where missing (bool as 0/1, datetime as epoch ms)."""
        if self._float is None:
            s = self.series
            if self.kind == "number":
                self._float = s.to_numpy(dtype="float64", na_value=np.nan)
            elif self.kind == "bool":
                self._float = s.astype("float64").to_numpy(dtype="float64", na_value=np.nan) \
                    if s.isna().any() else s.to_numpy(dtype="float64")
            elif self.kind == "datetime":
                self._float = self.millis()
            else:
                self._float = pd.to_numeric(s.astype(object).where(~self.missing, None),
                                            errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
        return self._float

    def millis(self) -> np.ndarray:
        """Epoch milliseconds (float, NaN for NaT) for a datetime column."""
        s = self.series
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        ns = s.to_numpy(dtype="datetime64[ns]").astype("int64").astype("float64")
        ns[self.missing] = np.nan
        return ns / 1e6

    # --- text ----------------------------------------------------------------
    def raw_text(self, lower: bool) -> np.ndarray:
        """``String(value)`` per row as the grid would write it; None where missing."""
        s = self.series
        if self.kind == "category":
            cats = s.cat.categories
            ctext = np.array([self._text_of(c, lower) for c in cats], dtype=object)
            codes = s.cat.codes.to_numpy()
            out = np.where(codes >= 0, ctext[np.clip(codes, 0, None)] if len(ctext) else None, None)
            return out.astype(object)
        if self.kind == "number":
            f = self.floats()
            out = np.empty(len(f), dtype=object)
            for i, x in enumerate(f):
                out[i] = None if x != x else js_number_text(x)
            return out
        if self.kind == "bool":
            vals = s.to_numpy(dtype=object)
            return np.array([None if m else ("true" if v else "false")
                             for v, m in zip(vals, self.missing)], dtype=object)
        if self.kind == "datetime":
            iso = [None if pd.isna(x) else x.isoformat() for x in s]
            if lower:
                iso = [None if x is None else x.lower() for x in iso]
            return np.array(iso, dtype=object)
        vals = s.to_numpy(dtype=object)
        out = np.empty(len(vals), dtype=object)
        miss = self.missing
        for i, v in enumerate(vals):
            out[i] = None if miss[i] else self._text_of(v, lower)
        return out

    @staticmethod
    def _text_of(v: Any, lower: bool) -> str:
        if isinstance(v, str):
            t = v
        elif isinstance(v, bool):
            t = "true" if v else "false"
        elif isinstance(v, (int, float, np.integer, np.floating)):
            t = js_number_text(float(v))
        else:
            t = str(_jsonable_scalar(v))
        return t.lower() if lower else t

    def text_values(self, lower: bool) -> pd.Series:
        """Text view as a Series (for the vectorised ``.str`` methods)."""
        return pd.Series(self.raw_text(lower), dtype=object)

    # --- values as the grid holds them ---------------------------------------
    def json_values(self, positions: np.ndarray | None = None) -> list:
        """JSON-safe values (the forward serialisation), optionally at positions."""
        from lattice_grid_pandas._serialize import column_values
        s = self.series if positions is None else self.series.iloc[positions]
        return column_values(s)


# ----------------------------------------------------------------------------
# filter predicates (compute/filter.js)
# ----------------------------------------------------------------------------
def _cmp_mask(values: np.ndarray, op: str, target: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        if op == "lt":
            return values < target
        if op == "lte":
            return values <= target
        if op == "gt":
            return values > target
        return values >= target


def condition_mask(col: _Col, cond: dict) -> np.ndarray:
    """Boolean mask for one leaf condition, matching ``compileValuePredicate``."""
    op = str(cond.get("op"))
    value = cond.get("value")
    cs = bool(cond.get("caseSensitive"))
    n = len(col)
    miss = col.missing

    if op in ("blank", "notBlank"):
        b = col.blank()
        return b if op == "blank" else ~b

    ordered = col.kind in ("number", "bool", "datetime")

    def ordered_vals_and_target(v):
        if col.kind == "datetime":
            t = _ts_target(v)
            return col.millis(), (np.nan if t is None else t.value / 1e6)
        if col.kind == "bool" and isinstance(v, str):
            v = {"true": 1.0, "false": 0.0}.get(v.strip().lower(), v)
        return col.floats(), _num_target(v)

    if op in ("eq", "ne"):
        if value is None:
            m = miss.copy()
        elif ordered:
            vals, t = ordered_vals_and_target(value)
            with np.errstate(invalid="ignore"):
                m = (vals == t) & ~miss
        else:
            text = col.text_values(lower=not cs)
            needle = col._text_of(value, lower=not cs)
            m = (text == needle).to_numpy(dtype=bool) & ~miss
        return m if op == "eq" else ~m

    if op in ("lt", "lte", "gt", "gte"):
        if ordered:
            vals, t = ordered_vals_and_target(value)
            if t != t:
                return np.zeros(n, dtype=bool)
            return _cmp_mask(vals, op, t) & ~miss
        # text: collator order, approximated by case-folded comparison
        if value is None:
            return np.zeros(n, dtype=bool)
        text = col.text_values(lower=True)
        needle = col._text_of(value, lower=True)
        ops = {"lt": text < needle, "lte": text <= needle, "gt": text > needle, "gte": text >= needle}
        return ops[op].fillna(False).to_numpy(dtype=bool) & ~miss

    if op in ("between", "notBetween"):
        pair = value if isinstance(value, (list, tuple)) else [value, value]
        lo_v, hi_v = (list(pair) + [None, None])[:2]
        bounds = str(cond.get("bounds") or "[]")
        lo_inc = bounds[0] != "("
        hi_inc = bounds[1:2] != ")"
        if ordered:
            vals, lo = ordered_vals_and_target(lo_v)
            _, hi = ordered_vals_and_target(hi_v)
            if lo != lo or hi != hi:
                return np.zeros(n, dtype=bool) if op == "between" else np.ones(n, dtype=bool)
            with np.errstate(invalid="ignore"):
                inside = (vals >= lo if lo_inc else vals > lo) & (vals <= hi if hi_inc else vals < hi)
        else:
            if lo_v is None or hi_v is None:
                return np.zeros(n, dtype=bool) if op == "between" else np.ones(n, dtype=bool)
            text = col.text_values(lower=True)
            lo_t, hi_t = col._text_of(lo_v, True), col._text_of(hi_v, True)
            a = (text >= lo_t) if lo_inc else (text > lo_t)
            b = (text <= hi_t) if hi_inc else (text < hi_t)
            inside = (a & b).fillna(False).to_numpy(dtype=bool)
        inside = inside & ~miss
        return inside if op == "between" else ~inside

    if op in ("in", "notIn"):
        entries = list(value) if isinstance(value, (list, tuple)) else [value]
        want_null = any(e is None for e in entries)
        if ordered:
            vals, _ = ordered_vals_and_target(None)
            targets = []
            for e in entries:
                if e is None:
                    continue
                _, t = ordered_vals_and_target(e)
                if t == t:
                    targets.append(t)
            m = np.isin(vals, np.array(targets, dtype="float64")) & ~miss if targets else np.zeros(n, bool)
        else:
            keys = set()
            for e in entries:
                if e is None:
                    continue
                keys.add(col._text_of(e, lower=not cs))
            text = col.text_values(lower=not cs)
            m = text.isin(keys).to_numpy(dtype=bool) & ~miss
        if want_null:
            m = m | miss
        return m if op == "in" else ~m

    if op in ("contains", "notContains", "startsWith", "endsWith", "matches"):
        if value is None:
            needle = ""
        else:
            needle = col._text_of(value, lower=False)
        text = col.text_values(lower=not cs)
        if op == "matches":
            try:
                re.compile(needle)
            except re.error:
                return np.zeros(n, dtype=bool)
            hit = text.str.contains(needle, regex=True, flags=0 if cs else re.IGNORECASE)
        else:
            nd = needle if cs else needle.lower()
            if op in ("contains", "notContains"):
                hit = text.str.contains(nd, regex=False)
            elif op == "startsWith":
                hit = text.str.startswith(nd)
            else:
                hit = text.str.endswith(nd)
        m = hit.fillna(False).to_numpy(dtype=bool) & ~miss
        return ~m if op == "notContains" else m

    raise ValueError(f"[lattice] filter operator {op!r} is not answered by the Python engine")


# ----------------------------------------------------------------------------
# the engine
# ----------------------------------------------------------------------------
class PandasEngine:
    """Every grid query, answered over a whole pandas DataFrame."""

    name = "pandas"
    #: cell edits reach the frame (``adapter.mutate`` update)
    editable = True

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._levels = [f for f, _ in _index_levels(df.iloc[:0])] if len(df) else []
        self._show_index = bool(self._levels)
        self._cols: dict[str, _Col] = {}
        self._positions: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._masks: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.version = 0

    # --- schema --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def columns(self) -> list[dict]:
        """The grid column definitions (index levels first, read-only)."""
        head = self.df.iloc[:0]
        return build_columns(head if self._show_index else head.reset_index(drop=True))

    def rows_at(self, positions: Iterable[int]) -> pd.DataFrame:
        """The frame's own rows at these positions (original index and dtypes)."""
        return self.df.iloc[list(positions)]

    def to_pandas(self) -> pd.DataFrame:
        return self.df

    def fields(self) -> list[str]:
        return [c["field"] for c in self.columns()]

    def col(self, field: str) -> _Col:
        """A field (data column or index level) by its grid field name."""
        if field in self._cols:
            return self._cols[field]
        series = None
        for name in self.df.columns:
            if str(name) == field:
                series = self.df[name]
                break
        if series is None and self._show_index:
            idx = self.df.index
            if isinstance(idx, pd.MultiIndex):
                for i, nm in enumerate(idx.names):
                    if (str(nm) if nm is not None else f"index_{i}") == field:
                        series = pd.Series(idx.get_level_values(i))
            elif str(idx.name if idx.name is not None else "index") == field:
                series = pd.Series(idx)
        if series is None:
            raise KeyError(f"[lattice] no column {field!r} in the frame")
        c = _Col(field, series)
        self._cols[field] = c
        return c

    def invalidate(self) -> None:
        """Forget every cached mask / order (the frame changed)."""
        self._cols.clear()
        self._positions.clear()
        self._masks.clear()
        self.version += 1

    # --- filter + quick + sort -> positions ------------------------------------
    def _node_mask(self, node: Any) -> np.ndarray:
        n = len(self)
        if not node:
            return np.ones(n, dtype=bool)
        if is_group(node):
            kids = [c for c in node["conditions"] if c is not None]
            op = node.get("op")
            if not kids:
                return np.ones(n, dtype=bool)
            acc = self._node_mask(kids[0])
            for k in kids[1:]:
                acc = (acc | self._node_mask(k)) if op == "or" else (acc & self._node_mask(k))
            return ~acc if op == "not" else acc
        return condition_mask(self.col(str(node.get("col"))), node)

    def quick_mask(self, quick: str) -> np.ndarray:
        """Free-text search: the needle (lower-cased) inside any field's text."""
        needle = str(quick or "").strip().lower()
        n = len(self)
        if not needle:
            return np.ones(n, dtype=bool)
        numeric_ok = re.fullmatch(r"[0-9.eE+\-infinity]+", needle) is not None
        hit = np.zeros(n, dtype=bool)
        for f in self.fields():
            c = self.col(f)
            if c.kind == "number" and not numeric_ok:
                continue  # a number's text cannot contain this needle
            if c.kind == "bool" and needle not in "true" and needle not in "false":
                continue
            text = c.text_values(lower=True)
            hit |= text.str.contains(needle, regex=False).fillna(False).to_numpy(dtype=bool)
        return hit

    def mask(self, filters: Any, quick: str = "") -> np.ndarray:
        key = query_key(filters, quick)
        m = self._masks.get(key)
        if m is None:
            m = self._node_mask(filters)
            if quick:
                m = m & self.quick_mask(quick)
            self._masks[key] = m
            while len(self._masks) > _CACHE_SIZE:
                self._masks.popitem(last=False)
        else:
            self._masks.move_to_end(key)
        return m

    def _sort_key(self, c: _Col, pos: np.ndarray, desc: bool) -> np.ndarray:
        """An ascending numeric rank per row (lexsort input) for one sort entry."""
        if c.kind in ("number", "bool", "datetime"):
            v = c.floats()[pos]
            v = np.where(np.isnan(v), 0.0, v)
            return -v if desc else v
        text = c.raw_text(lower=False)[pos]
        uniq = pd.unique(pd.Series(text, dtype=object).dropna())
        # collator order (base letters first, lower case before upper on a tie)
        ordered = sorted(uniq, key=lambda s: (s.casefold(), s.swapcase()))
        rank = {s: i for i, s in enumerate(ordered)}
        r = np.fromiter((rank.get(t, 0) if t is not None else 0 for t in text),
                        dtype=np.int64, count=len(text))
        return -r if desc else r

    def positions(self, filters: Any = None, quick: str = "", sort: Any = None) -> np.ndarray:
        """Row positions passing the filter + quick search, in the grid's sort order."""
        sort = [s for s in (sort or []) if s and s.get("col")]
        key = query_key(filters, quick, sort)
        hit = self._positions.get(key)
        if hit is not None:
            self._positions.move_to_end(key)
            return hit
        pos = np.flatnonzero(self.mask(filters, quick))
        if sort:
            keys = []
            for entry in reversed(sort):  # lexsort: last key is primary
                c = self.col(str(entry["col"]))
                desc = entry.get("dir") == "desc"
                keys.append(self._sort_key(c, pos, desc))
                absent = c.missing[pos]
                keys.append(~absent if entry.get("nullsFirst") else absent)
            order = np.lexsort(keys)
            pos = pos[order]
        self._positions[key] = pos
        while len(self._positions) > _CACHE_SIZE:
            self._positions.popitem(last=False)
        return pos

    # --- answers -------------------------------------------------------------
    def window(self, positions: np.ndarray) -> dict:
        """Columnar rows for these positions; ``__row_id__`` is the frame position."""
        part = self.df.iloc[positions]
        if not self._show_index:
            part = part.reset_index(drop=True)
        return build_columnar(part, [str(int(p)) for p in positions])

    def execute(self, query: dict) -> dict:
        """One window of the matching rows, and how many rows match."""
        pos = self.positions(query.get("filters"), query.get("quick") or "", query.get("sort"))
        rng = query.get("range")
        if rng:
            start = max(0, int(rng.get("start", 0)))
            end = max(start, int(rng.get("end", start)))
            win = pos[start:end]
        else:
            win = pos
        return {"columnar": self.window(win), "total": int(len(pos))}

    def _agg(self, c: _Col, fn: str, pos: np.ndarray) -> Any:
        if fn == "count":
            return int(len(pos))
        if fn == "countValues":
            return int((~c.blank()[pos]).sum())
        if fn == "distinct":
            keep = pos[~c.blank()[pos]]
            if c.kind in ("number", "bool", "datetime"):
                return int(len(np.unique(c.floats()[keep])))
            return int(len(pd.unique(pd.Series(c.raw_text(lower=False)[keep], dtype=object))))
        v = c.floats()[pos]
        v = v[~np.isnan(v)]
        if c.kind not in ("number", "bool") and fn not in ("min", "max", "range"):
            if c.kind != "datetime":
                return None
        if not len(v):
            return 0 if fn in ("sum", "sumSquares") else None
        if fn == "sum":
            return float(v.sum())
        if fn == "sumSquares":
            return float((v * v).sum())
        if fn == "avg":
            return float(v.mean())
        if fn == "min":
            return float(v.min())
        if fn == "max":
            return float(v.max())
        if fn == "range":
            return float(v.max() - v.min())
        q = {"median": 0.5, "p25": 0.25, "p75": 0.75, "p90": 0.9, "p95": 0.95, "p99": 0.99}
        if fn in q:
            return float(np.quantile(v, q[fn]))
        if fn == "iqr":
            return float(np.quantile(v, 0.75) - np.quantile(v, 0.25))
        if fn in ("variance", "stddev"):
            if len(v) < 2:
                return None
            var = float(v.var(ddof=1))
            return var if fn == "variance" else math.sqrt(var)
        if fn in ("varianceP", "stddevP"):
            var = float(v.var(ddof=0))
            return var if fn == "varianceP" else math.sqrt(var)
        raise ValueError(f"[lattice] statistic {fn!r} is not answered by the Python engine")

    def aggregates(self, query: dict, aggregates: list[dict]) -> dict:
        """Whole-set statistics over the matching rows, keyed by id."""
        pos = np.flatnonzero(self.mask(query.get("filters"), query.get("quick") or ""))
        return {a["id"]: self._agg(self.col(str(a["col"])), a["fn"], pos) for a in aggregates}

    def _group_codes(self, field: str, pos: np.ndarray) -> tuple[np.ndarray, list]:
        """Group codes for the rows at ``pos``, and each code's JSON key."""
        c = self.col(field)
        vals = c.json_values(pos)
        codes, uniques = pd.factorize(pd.Series(vals, dtype=object), use_na_sentinel=False)
        keys = [None if (u is None or (isinstance(u, float) and u != u)) else u for u in uniques]
        return codes, keys

    def grouped_aggregates(self, query: dict, group_by: list, aggregates: list[dict]) -> list[dict]:
        """ROLLUP over ``group_by``: one entry per group at every level + the grand row."""
        pos = np.flatnonzero(self.mask(query.get("filters"), query.get("quick") or ""))
        keys = [str(k) for k in group_by]
        per_key = [self._group_codes(k, pos) for k in keys]
        out: list[dict] = []
        for depth in range(len(keys), -1, -1):
            if depth == 0:
                groups = {(): np.arange(len(pos))}
            else:
                combo = np.zeros(len(pos), dtype=np.int64)
                for codes, ks in per_key[:depth]:
                    combo = combo * (len(ks) + 1) + codes
                order = np.argsort(combo, kind="stable")
                sc = combo[order]
                cuts = np.flatnonzero(np.diff(sc)) + 1
                groups = {}
                for chunk in np.split(order, cuts):
                    if not len(chunk):
                        continue
                    i0 = chunk[0]
                    groups[tuple(per_key[j][1][per_key[j][0][i0]] for j in range(depth))] = chunk
            for gkeys, local in groups.items():
                rows = pos[local]
                values = {a["id"]: self._agg(self.col(str(a["col"])), a["fn"], rows) for a in aggregates}
                out.append({
                    "keys": list(gkeys) + [None] * (len(keys) - depth),
                    "grouping": [0] * depth + [1] * (len(keys) - depth),
                    "level": depth,
                    "values": values,
                })
        return out

    # --- header histograms (compute/facet.js) ----------------------------------
    def facet_kind(self, c: _Col) -> str:
        return {"number": "numeric", "bool": "boolean", "datetime": "date"}.get(c.kind, "category")

    def facet(self, req: dict) -> dict:
        """Bucket bounds (placed over every row) and cross-filtered counts."""
        c = self.col(str(req["colId"]))
        kind = self.facet_kind(c)
        bounds = req.get("bounds")
        computed = False
        if not bounds:
            bounds = facet_bounds(c, kind, req)
            computed = True
        m = self.mask(req.get("filters"), req.get("quick") or "")
        counts = facet_counts(c, bounds, m)
        out = {"bounds": bounds, "counts": counts}
        if computed:
            out["unfiltered"] = facet_counts(c, bounds, None)
        return out

    # --- the Statistics panel's profile (grid.statistics.profile) --------------
    def profile(self, field: str, filters: Any = None, quick: str = "") -> dict:
        c = self.col(field)
        pos = np.flatnonzero(self.mask(filters, quick))
        return column_profile(c, pos)


# ----------------------------------------------------------------------------
# facet bounds / counts
# ----------------------------------------------------------------------------
def _floor_utc(ms: float, g: str) -> float:
    t = pd.Timestamp(int(ms), unit="ms")
    if g == "hour":
        t = t.floor("h")
    elif g == "day":
        t = t.normalize()
    elif g == "week":
        t = t.normalize() - pd.Timedelta(days=t.weekday())
    elif g == "month":
        t = t.normalize().replace(day=1)
    elif g == "quarter":
        t = t.normalize().replace(day=1, month=((t.month - 1) // 3) * 3 + 1)
    else:
        t = t.normalize().replace(day=1, month=1)
    return float(t.value // 1_000_000)


def _advance_utc(ms: float, g: str) -> float:
    t = pd.Timestamp(int(ms), unit="ms")
    step = {"hour": pd.Timedelta(hours=1), "day": pd.Timedelta(days=1), "week": pd.Timedelta(days=7),
            "month": pd.DateOffset(months=1), "quarter": pd.DateOffset(months=3)}.get(g, pd.DateOffset(years=1))
    return float((t + step).value // 1_000_000)


def _pick_granularity(span: float, target: int) -> str:
    ms = span if (math.isfinite(span) and span > 0) else 0
    wide = max(1, target) * 2
    hour, day = 3_600_000, 86_400_000
    if ms / hour <= wide:
        return "hour"
    if ms / day <= wide:
        return "day"
    if ms / (7 * day) <= wide:
        return "week"
    if ms / (30 * day) <= wide:
        return "month"
    if ms / (91 * day) <= wide:
        return "quarter"
    return "year"


def facet_bounds(c: _Col, kind: str, req: dict) -> dict:
    """``computeBounds`` over every row of the column."""
    wanted = max(1, int(req.get("buckets") or DEFAULT_BUCKETS))
    if kind == "boolean":
        b = [{"value": False, "label": "false"}, {"value": True, "label": "true"}]
        if c.missing.any():
            b.append({"null": True, "label": "Empty"})
        return {"kind": "boolean", "buckets": b}
    if kind == "category":
        text = c.raw_text(lower=False)
        s = pd.Series(text, dtype=object)
        blank = s.isna() | (s == "")
        nulls = int(blank.sum())
        present = s[~blank]
        # tally in first-seen order, then a stable sort by count (Map + Array.sort)
        counts = present.value_counts(sort=False)
        first = pd.unique(present)
        counts = counts.reindex(first)
        limit = DEFAULT_CARDINALITY_LIMIT
        if len(counts) > limit:
            return {"kind": "category", "buckets": [], "suppressed": "cardinality",
                    "cardinality": int(len(counts))}
        order = sorted(range(len(counts)), key=lambda i: -int(counts.iloc[i]))
        buckets = [{"value": _cat_value(c, counts.index[i]), "label": str(counts.index[i])} for i in order]
        if nulls:
            buckets.append({"null": True, "label": "Empty"})
        return {"kind": "category", "buckets": buckets, "cardinality": int(len(counts))}

    vals = c.floats()
    finite = vals[np.isfinite(vals)]
    nulls = int(len(vals) - len(finite))
    if not len(finite):
        return {"kind": kind, "buckets": [{"null": True, "label": "Empty"}] if nulls else [], "empty": True}
    vmin, vmax = float(finite.min()), float(finite.max())
    buckets: list[dict] = []
    if kind == "date":
        g = req.get("granularity")
        g = g if g in ("hour", "day", "week", "month", "quarter", "year") else _pick_granularity(vmax - vmin, wanted)
        edge = _floor_utc(vmin, g)
        while edge <= vmax and len(buckets) < 4096:
            nxt = _advance_utc(edge, g)
            if not nxt > edge:
                break
            buckets.append({"from": edge, "to": nxt})
            edge = nxt
        if nulls:
            buckets.append({"null": True, "label": "Empty"})
        return {"kind": "date", "buckets": buckets, "granularity": g}
    strategy = req.get("strategy") if req.get("strategy") in ("equal", "quantile", "log") else "equal"
    if strategy == "quantile":
        n = len(vals)
        step = n / QUANTILE_SAMPLE if n > QUANTILE_SAMPLE else 1
        picks = []
        s = 0.0
        while s < n:
            picks.append(int(math.floor(s)))
            s += step
        sample = vals[np.array(picks, dtype=np.int64)]
        sample = np.sort(sample[np.isfinite(sample)])
        if len(sample):
            edges = [float(sample[0])]
            for b in range(1, wanted):
                v = float(sample[min(len(sample) - 1, int(math.floor((b / wanted) * len(sample))))])
                if v > edges[-1]:
                    edges.append(v)
            edges.append(vmax)
            buckets = [{"from": edges[i], "to": edges[i + 1]} for i in range(len(edges) - 1)]
    elif strategy == "log" and vmin > 0:
        lo, hi = math.log10(vmin), math.log10(vmax)
        step = (hi - lo) / wanted or 1
        buckets = [{"from": 10 ** (lo + b * step), "to": 10 ** (lo + (b + 1) * step)} for b in range(wanted)]
    if not buckets:
        strategy = "equal" if strategy != "quantile" else strategy
        width = (vmax - vmin) / wanted or 1
        buckets = [{"from": vmin + b * width, "to": vmin + (b + 1) * width} for b in range(wanted)]
    buckets[-1]["to"] = vmax
    if nulls:
        buckets.append({"null": True, "label": "Empty"})
    return {"kind": kind, "buckets": buckets, "strategy": strategy, "min": vmin, "max": vmax}


def _cat_value(c: _Col, text: str) -> Any:
    """The bucket's value as the grid holds the cell (numbers stay numbers)."""
    return text


def facet_counts(c: _Col, bounds: dict, mask: np.ndarray | None) -> list[int]:
    """``countInto``: per-bucket counts over the rows in ``mask`` (all when None)."""
    buckets = bounds.get("buckets") or []
    if not buckets:
        return []
    sel = slice(None) if mask is None else mask
    has_null = bool(buckets[-1].get("null"))
    null_at = len(buckets) - 1
    counts = np.zeros(len(buckets), dtype=np.int64)
    kind = bounds.get("kind")
    if kind == "boolean":
        f = c.floats()[sel]
        miss = np.isnan(f)
        counts[0] = int(((f == 0) & ~miss).sum())
        counts[1] = int(((f != 0) & ~miss).sum())
        if has_null:
            counts[null_at] = int(miss.sum())
        return counts.tolist()
    if kind == "category":
        text = pd.Series(c.raw_text(lower=False)[sel], dtype=object)
        blank = text.isna() | (text == "")
        vc = text[~blank].value_counts()
        rem = next((i for i, b in enumerate(buckets) if b.get("remainder")), -1)
        seen = 0
        for i, b in enumerate(buckets):
            if b.get("null") or b.get("remainder"):
                continue
            k = int(vc.get(str(b.get("value")), 0))
            counts[i] = k
            seen += k
        if rem >= 0:
            counts[rem] = int(vc.sum()) - seen
        if has_null:
            counts[null_at] = int(blank.sum())
        return counts.tolist()
    f = c.floats()[sel]
    ordered = len(buckets) - 1 if has_null else len(buckets)
    if has_null:
        counts[null_at] = int((~np.isfinite(f)).sum())
    if not ordered:
        return counts.tolist()
    edges = np.array([b["from"] for b in buckets[:ordered]] + [buckets[ordered - 1]["to"]], dtype="float64")
    v = f[np.isfinite(f)]
    at = np.searchsorted(edges, v, side="right") - 1
    top = edges[ordered]
    at = np.where(v == top, ordered - 1, at)
    ok = (v >= edges[0]) & ((v < top) | (v == top))
    at = at[ok]
    counts[:ordered] += np.bincount(at, minlength=ordered)[:ordered]
    return counts.tolist()


# ----------------------------------------------------------------------------
# profile (grid.statistics.profile)
# ----------------------------------------------------------------------------
def _quantile7(sorted_v: np.ndarray, p: float) -> float:
    n = len(sorted_v)
    if not n:
        return float("nan")
    if n == 1:
        return float(sorted_v[0])
    h = (n - 1) * min(1.0, max(0.0, p))
    lo, hi = math.floor(h), math.ceil(h)
    if lo == hi:
        return float(sorted_v[lo])
    return float(sorted_v[lo] + (h - lo) * (sorted_v[hi] - sorted_v[lo]))


def stat_histogram(sorted_v: np.ndarray, q1: float, q3: float, cap: int = 20) -> list[dict]:
    """``statistics.histogram``: Freedman-Diaconis bins over Tukey's fences."""
    n = len(sorted_v)
    if not n:
        return []
    vmin, vmax = float(sorted_v[0]), float(sorted_v[-1])
    if vmax == vmin:
        return [{"from": vmin, "to": vmax, "count": n}]
    iqr = q3 - q1
    fence = 1.5 * iqr
    lo = max(vmin, q1 - fence) if iqr > 0 else vmin
    hi = min(vmax, q3 + fence) if iqr > 0 else vmax
    if not hi > lo:
        lo, hi = vmin, vmax
    width = (2 * iqr) / (n ** (1 / 3)) if iqr > 0 else (hi - lo) / (math.ceil(math.log2(n)) + 1)
    count = min(cap, max(1, math.ceil((hi - lo) / width))) if width > 0 else 1
    step = (hi - lo) / count
    at = np.clip(np.floor((sorted_v - lo) / step), 0, count - 1).astype(np.int64)
    tally = np.bincount(at, minlength=count)
    bins = [{"from": lo + i * step, "to": lo + (i + 1) * step, "count": int(tally[i])} for i in range(count)]
    bins[0]["from"] = vmin
    bins[-1]["to"] = vmax
    return bins


def column_profile(c: _Col, pos: np.ndarray) -> dict:
    """The grid's one-pass column profile, over the rows at ``pos``."""
    rows = int(len(pos))
    blank = c.blank()[pos]
    absent = int(blank.sum())
    numeric_col = c.kind in ("number",)
    if not numeric_col and c.kind not in ("bool",):
        # isNumericColumn: the first 200 present values decide it
        text = c.raw_text(lower=False)[pos]
        sample = [t for t in text[:20000] if t is not None and t != ""][:200]
        if c.kind != "datetime" and sample:
            good = sum(1 for t in sample if math.isfinite(_num_target(t)))
            numeric_col = good / len(sample) >= 0.9
    f = c.floats()[pos] if c.kind != "datetime" else np.full(rows, np.nan)
    if c.kind in ("text", "category"):
        f = np.array([_num_target(t) for t in c.raw_text(lower=False)[pos]], dtype="float64") \
            if numeric_col else np.full(rows, np.nan)
    vals = np.sort(f[np.isfinite(f)])
    n = int(len(vals))
    q1 = _quantile7(vals, 0.25)
    q3 = _quantile7(vals, 0.75)
    iqr = q3 - q1
    mean = float(vals.mean()) if n else None
    out = {
        "column": c.name,
        "rows": rows,
        "present": rows - absent,
        "missing": absent,
        "numeric": n,
        "distinct": _distinct_count(c, pos),
        "min": float(vals[0]) if n else None,
        "max": float(vals[-1]) if n else None,
        "mean": mean,
        "median": _quantile7(vals, 0.5) if n else None,
        "q1": q1 if n else None,
        "q3": q3 if n else None,
        "iqr": iqr if n else None,
        "stddev": float(vals.std(ddof=1)) if n > 1 else None,
        "outliers": int(((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum()) if n else 0,
        "histogram": stat_histogram(vals, q1, q3) if n else [],
    }
    if not numeric_col:
        out["topValues"] = _top_values(c, pos)
    return out


def _distinct_count(c: _Col, pos: np.ndarray) -> int:
    keep = pos[~c.blank()[pos]]
    if c.kind in ("number", "bool"):
        return int(len(np.unique(c.floats()[keep])))
    return int(len(pd.unique(pd.Series(c.raw_text(lower=False)[keep], dtype=object))))


def _top_values(c: _Col, pos: np.ndarray, k: int = PROFILE_TOP_VALUES) -> list[dict]:
    keep = pos[~c.blank()[pos]]
    if not len(keep):
        return []
    vals = c.json_values(keep)
    s = pd.Series(vals, dtype=object)
    vc = s.value_counts(sort=False)
    total = int(vc.sum())
    items = sorted(vc.items(), key=lambda kv: (-int(kv[1]), _default_order(kv[0])))
    return [{"value": v, "count": int(n), "share": int(n) / total} for v, n in items[:max(1, k)]]


def _default_order(v: Any):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return (0, float(v), "")
    return (1, 0.0, str(v))


def frame_positions_to_keys(positions: Iterable[int]) -> list[str]:
    return [str(int(p)) for p in positions]


__all__ = ["PandasEngine", "OPERATORS", "AGGREGATES", "ROW_KEY", "is_group", "query_key"]
