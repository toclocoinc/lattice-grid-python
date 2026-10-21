"""A DuckDB relation or connection as the grid's source: every query is SQL (BACKLOG-0005164).

The data stays in DuckDB. Where :class:`~lattice_grid_jupyter._arrow_engine.ArrowEngine`
holds the whole table as Arrow and filters, groups and profiles it in Python, this
engine turns each request of the windowed source into a SQL statement and runs it
on the connection, so the only things that ever reach Python are the answer: a
window of rows, a list of groups, a few bucket counts, a profile.

* ``execute``                -- ``SELECT ... WHERE ... ORDER BY ... LIMIT/OFFSET`` and a ``count(*)``;
* ``grouped_aggregates``     -- ``GROUP BY ROLLUP`` with the requested aggregates;
* ``aggregates``             -- one aggregate ``SELECT``;
* ``facet``                  -- bucket bounds and ``count(*) FILTER`` per bucket;
* ``profile``                -- the Statistics tab, from ``quantile_cont``, ``stddev_samp`` ... ;
* ``positions`` / ``rows_at`` / ``count`` -- what ``w.view`` (a lazy handle) needs.

**Values are always bound.** A filter value, a quick-search needle, a range bound, a
window offset reaches DuckDB as a prepared-statement parameter, never as text in the
statement. The only text built from the request is an identifier, and an identifier is
accepted only when it is one of the source's column names (then quoted). Over a
``DuckDBPyRelation`` (which has no ``execute(sql, params)``) the parameters travel as
typed constant columns of a one-row relation cross-joined to the source -- still values,
never SQL text.

Row identity is the row's position in the source (``row_number() OVER () - 1``), the
same key the Arrow path uses. DuckDB numbers the rows in scan order, which is stable for
a table or a file; give a ``sql=`` an ``ORDER BY`` if its row order must be repeatable.

The semantics are those of :mod:`._engine` (the grid's own kernels); the tests hold the two
paths equal on a seeded sample.
"""

from __future__ import annotations

import datetime
import math
import re
import threading
from collections import OrderedDict, deque
from typing import Any

import numpy as np
import pandas as pd

from lattice_grid_pandas._serialize import _jsonable_scalar, build_columnar, build_columns

from ._engine import (
    DEFAULT_BUCKETS,
    DEFAULT_CARDINALITY_LIMIT,
    PROFILE_TOP_VALUES,
    QUANTILE_SAMPLE,
    PandasEngine,
    _advance_utc,
    _default_order,
    _floor_utc,
    _num_target,
    _pick_granularity,
    _quantile7,
    _ts_target,
    is_group,
    query_key,
    stat_histogram,
)

POS = "__lg_pos"
_CACHE_SIZE = 8
_FLOAT_TYPES = {"FLOAT", "DOUBLE", "REAL"}
_INT_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
              "UINTEGER", "UBIGINT", "UHUGEINT"}
_TEXT_OPS = ("contains", "notContains", "startsWith", "endsWith", "matches")


def is_duckdb_relation(data: Any) -> bool:
    """A ``duckdb.DuckDBPyRelation`` (checked by type name: duckdb is optional)."""
    t = type(data)
    return t.__name__ == "DuckDBPyRelation" and t.__module__.split(".")[0] in ("duckdb", "_duckdb")


def is_duckdb_connection(data: Any) -> bool:
    t = type(data)
    return t.__name__ == "DuckDBPyConnection" and t.__module__.split(".")[0] in ("duckdb", "_duckdb")


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _quote_table(table: str) -> str:
    """``schema.table`` / ``catalog.schema.table`` -> quoted parts."""
    return ".".join(quote_ident(p) for p in str(table).split("."))


class _Q:
    """The parameters of one statement: ``p(value)`` returns its ``$name`` placeholder."""

    def __init__(self, inline_ints: bool = False) -> None:
        self.params: dict[str, Any] = {}
        #: over a relation the parameters are columns, which LIMIT/OFFSET cannot read
        self.inline_ints = inline_ints

    def i(self, n: int) -> str:
        """A LIMIT / OFFSET operand (an ``int()``, bound unless the statement cannot bind it)."""
        return str(int(n)) if self.inline_ints else self.p(int(n))

    def p(self, value: Any) -> str:
        if isinstance(value, np.generic):
            value = value.item()
        name = f"lgp{len(self.params)}"
        self.params[name] = value
        return "$" + name


class _SCol:
    """One column of the source and the SQL expressions the kernels need from it."""

    def __init__(self, name: str, dtype: str):
        self.name = name
        self.dtype = dtype.upper()
        base = re.split(r"[(\[ ]", self.dtype, maxsplit=1)[0]
        self.base = base
        self.is_float = base in _FLOAT_TYPES
        if self.dtype.endswith("[]") or self.dtype.startswith(("STRUCT", "MAP", "UNION")):
            self.kind = "text"
        elif base == "BOOLEAN":
            self.kind = "bool"
        elif base in _FLOAT_TYPES or base in _INT_TYPES or base == "DECIMAL":
            self.kind = "number"
        elif base in ("DATE",) or base.startswith("TIMESTAMP"):
            self.kind = "datetime"
        elif base == "ENUM":
            self.kind = "category"
        else:
            self.kind = "text"
        self.q = "b." + quote_ident(name)

    # --- value views ---------------------------------------------------------
    @property
    def ts(self) -> str:
        """The column as a naive UTC TIMESTAMP (a datetime column)."""
        if "TIME ZONE" in self.dtype:
            return f"CAST(timezone('UTC', {self.q}) AS TIMESTAMP)"
        return f"CAST({self.q} AS TIMESTAMP)"

    @property
    def num(self) -> str:
        """DOUBLE view, NULL where missing (NaN is missing; bool as 0/1, datetime as epoch ms)."""
        if self.kind == "number":
            c = f"CAST({self.q} AS DOUBLE)"
            return f"NULLIF({c}, 'NaN'::DOUBLE)" if self.is_float else c
        if self.kind == "bool":
            return f"CAST(CAST({self.q} AS INTEGER) AS DOUBLE)"
        if self.kind == "datetime":
            return f"(epoch_us({self.ts}) / 1000.0)"
        return f"NULLIF(TRY_CAST(CAST({self.q} AS VARCHAR) AS DOUBLE), 'NaN'::DOUBLE)"

    @property
    def txt(self) -> str:
        """``String(value)`` as the grid writes it; NULL where missing."""
        q = self.q
        if self.kind == "text" or self.kind == "category":
            return f"CAST({q} AS VARCHAR)"
        if self.kind == "bool":
            return f"CASE {q} WHEN true THEN 'true' WHEN false THEN 'false' END"
        if self.kind == "datetime":
            t = self.ts
            return (f"CASE WHEN {t} IS NULL THEN NULL WHEN date_trunc('second', {t}) = {t} "
                    f"THEN strftime({t}, '%Y-%m-%dT%H:%M:%SZ') "
                    f"ELSE strftime({t}, '%Y-%m-%dT%H:%M:%S.%f') || 'Z' END")
        if self.is_float:
            return f"regexp_replace(CAST({self.num} AS VARCHAR), '\\.0$', '')"
        return f"CAST({q} AS VARCHAR)"

    @property
    def miss(self) -> str:
        if self.is_float:
            return f"({self.q} IS NULL OR isnan({self.q}))"
        return f"({self.q} IS NULL)"

    @property
    def blank(self) -> str:
        """``isBlank``: missing, or the empty string."""
        if self.kind in ("text", "category"):
            return f"({self.q} IS NULL OR {self.txt} = '')"
        return self.miss

    @property
    def key(self) -> str:
        """The value as a group key / a distinct value (NaN is missing)."""
        if self.is_float:
            return f"NULLIF({self.q}, 'NaN'::{self.dtype})"
        if self.kind == "datetime":
            return self.ts
        if self.kind == "category":
            return f"CAST({self.q} AS VARCHAR)"
        return self.q

    @property
    def ordered(self) -> bool:
        return self.kind in ("number", "bool", "datetime")


def _text_of(v: Any, lower: bool) -> str:
    from ._engine import _Col
    return _Col._text_of(v, lower)


def _target(col: _SCol, v: Any) -> float:
    """The comparison operand for an ordered column, as a float (NaN = unusable)."""
    if col.kind == "datetime":
        t = _ts_target(v)
        return float("nan") if t is None else t.value / 1e6
    if col.kind == "bool" and isinstance(v, str):
        v = {"true": 1.0, "false": 0.0}.get(v.strip().lower(), v)
    return _num_target(v)


class DuckDBEngine(PandasEngine):
    """Every grid query, answered as SQL over a DuckDB relation or table."""

    name = "duckdb"
    editable = False

    def __init__(self, *, relation: Any = None, con: Any = None, table: str | None = None,
                 sql: str | None = None, keep_alive: Any = None):
        if relation is None and con is None:
            raise ValueError("[lattice] a DuckDB source needs a relation, or con= with table= or sql=")
        if relation is None:
            if bool(table) == bool(sql):
                raise ValueError("[lattice] con= needs exactly one of table= or sql=")
            src = _quote_table(table) if table else "(" + str(sql).strip().rstrip(";").strip() + ")"
            relation = con.sql(f"SELECT * FROM {src} AS s")
            self._src, self._con = src, con
        else:
            self._src, self._con = None, None
        self.relation = relation
        self._keep_alive = keep_alive  # a private connection that must outlive the engine
        self._lock = threading.RLock()
        #: ``(sql, params, rows_returned)`` of the latest statements (tests and debugging read it)
        self.sql_log: deque = deque(maxlen=2000)
        self._cols_by_name: dict[str, _SCol] = {}
        names = list(relation.columns)
        types = [str(t) for t in relation.types]
        if len(set(names)) != len(names):
            raise ValueError("[lattice] the DuckDB source has duplicate column names; alias them")
        self._order = names
        for n, t in zip(names, types):
            self._cols_by_name[n] = _SCol(n, t)
        self._schema_frame = self._fetch_df(
            f"SELECT {self._select_list()} FROM __BASE__ LIMIT 0", {}).iloc[:, :]
        self._count_cache: "OrderedDict[str, int]" = OrderedDict()
        self._pos_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._n: int | None = None
        self.version = 0
        self._levels, self._show_index = [], False
        self.df = self._schema_frame

    # --- running statements ----------------------------------------------------
    def _q(self) -> _Q:
        return _Q(inline_ints=self._con is None)

    def _select_list(self) -> str:
        return ", ".join(f"b.{quote_ident(n)}" for n in self._order)

    def _prepare(self, sql: str, params: dict) -> tuple[Any, dict]:
        """``(relation_or_connection_statement, params)`` with ``__BASE__`` expanded."""
        names = set(re.findall(r"\$(lgp\d+)\b", sql))
        used = [k for k in params if k in names]
        cols = ", ".join(quote_ident(n) for n in self._order)
        if self._con is not None:
            number = f", row_number() OVER () - 1 AS {POS}" if POS in sql else ""   # numbered only when asked
            base = f"(SELECT {cols}{number} FROM {self._src}) AS b"
            return sql.replace("__BASE__", base), {k: params[k] for k in used}
        # relation: parameters become typed constant columns of a one-row relation
        extra = "".join(", " + k for k in used)
        number = f", row_number() OVER () - 1 AS {POS}" if POS in sql else ""
        base = f"(SELECT {cols}{extra}{number} FROM v) AS b"
        text = sql.replace("__BASE__", base)
        text = re.sub(r"\$(lgp\d+)\b", lambda m: "b." + m.group(1), text)
        return text, {k: params[k] for k in used}

    def _run(self, sql: str, params: dict | None = None):
        params = params or {}
        text, used = self._prepare(sql, params)
        with self._lock:
            if self._con is not None:
                res = self._con.execute(text, used)
            else:
                rel = self.relation
                if used:
                    import duckdb
                    one = rel.query("v", "SELECT 1 AS lg_one")
                    consts = one.select(*[duckdb.ConstantExpression(v).alias(k) for k, v in used.items()])
                    rel = rel.cross(consts)
                res = rel.query("v", text)
            return res

    def _log(self, sql: str, params: dict, rows: int) -> None:
        self.sql_log.append((sql, dict(params), rows))

    def _fetch_all(self, sql: str, params: dict | None = None) -> list[tuple]:
        params = params or {}
        res = self._run(sql, params)
        rows = res.fetchall()
        self._log(sql, params, len(rows))
        return rows

    def _fetch_df(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        params = params or {}
        res = self._run(sql, params)
        # duckdb 2.x renamed fetch_arrow_table to to_arrow_table on a connection
        to_arrow = getattr(res, "to_arrow_table", None) or res.fetch_arrow_table
        df = to_arrow().to_pandas(ignore_metadata=True)
        self._log(sql, params, len(df))
        return df

    def _scalar(self, sql: str, params: dict | None = None) -> Any:
        return self._fetch_all(sql, params)[0][0]

    # --- schema ----------------------------------------------------------------
    def __len__(self) -> int:
        if self._n is None:
            self._n = int(self._scalar("SELECT count(*) FROM __BASE__"))
        return self._n

    def columns(self) -> list[dict]:
        return [{**c, "edit": False} for c in build_columns(self._schema_frame)]

    def fields(self) -> list[str]:
        return list(self._order)

    def col(self, field: str) -> _SCol:  # type: ignore[override]
        try:
            return self._cols_by_name[field]
        except KeyError:
            raise KeyError(f"[lattice] no column {field!r} in the data") from None

    def invalidate(self) -> None:
        self._count_cache.clear()
        self._pos_cache.clear()
        self._n = None
        self.version += 1

    # --- filter + quick -> SQL ---------------------------------------------------
    def _leaf(self, c: _SCol, cond: dict, q: _Q) -> str:
        op = str(cond.get("op"))
        value = cond.get("value")
        cs = bool(cond.get("caseSensitive"))
        if op in ("blank", "notBlank"):
            return c.blank if op == "blank" else f"(NOT {c.blank})"
        if c.ordered:
            num = c.num
            if op in ("eq", "ne"):
                if value is None:
                    m = c.miss
                else:
                    t = _target(c, value)
                    m = "FALSE" if t != t else f"COALESCE({num} = {q.p(t)}, FALSE)"
                return m if op == "eq" else f"(NOT {m})"
            if op in ("lt", "lte", "gt", "gte"):
                t = _target(c, value)
                if t != t:
                    return "FALSE"
                sym = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">="}[op]
                return f"COALESCE({num} {sym} {q.p(t)}, FALSE)"
            if op in ("between", "notBetween"):
                lo, hi = self._bounds(value)
                lo_t, hi_t = _target(c, lo), _target(c, hi)
                if lo_t != lo_t or hi_t != hi_t:
                    return "FALSE" if op == "between" else "TRUE"
                li, hi_inc = self._inclusive(cond)
                inside = (f"COALESCE({num} {'>=' if li else '>'} {q.p(lo_t)} AND "
                          f"{num} {'<=' if hi_inc else '<'} {q.p(hi_t)}, FALSE)")
                return inside if op == "between" else f"(NOT {inside})"
            if op in ("in", "notIn"):
                entries = list(value) if isinstance(value, (list, tuple)) else [value]
                targets = [t for t in (_target(c, e) for e in entries if e is not None) if t == t]
                m = f"COALESCE(list_contains({q.p(targets)}, {num}), FALSE)" if targets else "FALSE"
                if any(e is None for e in entries):
                    m = f"({m} OR {c.miss})"
                return m if op == "in" else f"(NOT {m})"
        txt = c.txt
        low = txt if cs else f"lower({txt})"
        if op in ("eq", "ne"):
            if value is None:
                m = c.miss
            else:
                m = f"COALESCE({low} = {q.p(_text_of(value, not cs))}, FALSE)"
            return m if op == "eq" else f"(NOT {m})"
        if op in ("lt", "lte", "gt", "gte"):
            if value is None:
                return "FALSE"
            sym = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">="}[op]
            return f"COALESCE(lower({txt}) {sym} {q.p(_text_of(value, True))}, FALSE)"
        if op in ("between", "notBetween"):
            lo, hi = self._bounds(value)
            if lo is None or hi is None:
                return "FALSE" if op == "between" else "TRUE"
            li, hi_inc = self._inclusive(cond)
            inside = (f"COALESCE(lower({txt}) {'>=' if li else '>'} {q.p(_text_of(lo, True))} AND "
                      f"lower({txt}) {'<=' if hi_inc else '<'} {q.p(_text_of(hi, True))}, FALSE)")
            return inside if op == "between" else f"(NOT {inside})"
        if op in ("in", "notIn"):
            entries = list(value) if isinstance(value, (list, tuple)) else [value]
            keys = list(dict.fromkeys(_text_of(e, not cs) for e in entries if e is not None))
            m = f"COALESCE(list_contains({q.p(keys)}, {low}), FALSE)" if keys else "FALSE"
            if any(e is None for e in entries):
                m = f"({m} OR {c.miss})"
            return m if op == "in" else f"(NOT {m})"
        if op in _TEXT_OPS:
            needle = "" if value is None else _text_of(value, False)
            if op == "matches":
                try:
                    re.compile(needle)
                except re.error:
                    return "FALSE"
                flag = "" if cs else ", 'i'"
                return f"COALESCE(regexp_matches({txt}, {q.p(needle)}{flag}), FALSE)"
            nd = needle if cs else needle.lower()
            fn = {"contains": "contains", "notContains": "contains", "startsWith": "starts_with",
                  "endsWith": "ends_with"}[op]
            m = f"COALESCE({fn}({low}, {q.p(nd)}), FALSE)"
            return f"(NOT {m})" if op == "notContains" else m
        raise ValueError(f"[lattice] filter operator {op!r} is not answered by the Python engine")

    @staticmethod
    def _bounds(value: Any) -> tuple[Any, Any]:
        pair = value if isinstance(value, (list, tuple)) else [value, value]
        lo, hi = (list(pair) + [None, None])[:2]
        return lo, hi

    @staticmethod
    def _inclusive(cond: dict) -> tuple[bool, bool]:
        bounds = str(cond.get("bounds") or "[]")
        return bounds[0] != "(", bounds[1:2] != ")"

    def _node(self, node: Any, q: _Q) -> str:
        if not node:
            return "TRUE"
        if is_group(node):
            kids = [c for c in node["conditions"] if c is not None]
            if not kids:
                return "TRUE"
            joiner = " OR " if node.get("op") == "or" else " AND "
            acc = "(" + joiner.join(self._node(k, q) for k in kids) + ")"
            return f"(NOT {acc})" if node.get("op") == "not" else acc
        return self._leaf(self.col(str(node.get("col"))), node, q)

    def _quick(self, quick: str, q: _Q) -> str:
        needle = str(quick or "").strip().lower()
        if not needle:
            return "TRUE"
        numeric_ok = re.fullmatch(r"[0-9.e+\-infinity]+", needle) is not None
        iso_ok = re.fullmatch(r"[0-9t:.+\-z]+", needle) is not None
        ph = q.p(needle)
        parts = []
        for f in self._order:
            c = self.col(f)
            if c.kind == "number" and not numeric_ok:
                continue
            if c.kind == "datetime" and not iso_ok:
                continue
            if c.kind == "bool" and needle not in "true" and needle not in "false":
                continue
            parts.append(f"COALESCE(contains(lower({c.txt}), {ph}), FALSE)")
        return "(" + " OR ".join(parts) + ")" if parts else "FALSE"

    def _where(self, filters: Any, quick: str, q: _Q) -> str:
        w = self._node(filters, q)
        if quick:
            w = f"({w} AND {self._quick(quick, q)})"
        return w

    def _order_by(self, sort: Any) -> str:
        parts = []
        for entry in (s for s in (sort or []) if s and s.get("col")):
            c = self.col(str(entry["col"]))
            desc = entry.get("dir") == "desc"
            nulls = "FIRST" if entry.get("nullsFirst") else "LAST"
            d, rd = ("DESC", "ASC") if desc else ("ASC", "DESC")
            if c.ordered:
                parts.append(f"{c.num} {d} NULLS {nulls}")
            else:
                # collator order: base letters first, lower case before upper on a tie
                parts.append(f"lower({c.txt}) {d} NULLS {nulls}")
                parts.append(f"{c.txt} {rd}")
        parts.append(POS + " ASC")
        return ", ".join(parts)

    # --- the matching rows -------------------------------------------------------
    def count(self, filters: Any = None, quick: str = "") -> int:
        """How many rows pass the filter and the quick search."""
        if not filters and not quick:
            return len(self)
        key = query_key(filters, quick)
        hit = self._count_cache.get(key)
        if hit is None:
            q = self._q()
            hit = int(self._scalar(f"SELECT count(*) FROM __BASE__ WHERE {self._where(filters, quick, q)}",
                                   q.params))
            self._count_cache[key] = hit
            while len(self._count_cache) > _CACHE_SIZE:
                self._count_cache.popitem(last=False)
        else:
            self._count_cache.move_to_end(key)
        return hit

    def _window_positions(self, filters: Any, quick: str, sort: Any, start: int, stop: int) -> np.ndarray:
        """Source positions of view rows ``start:stop`` (only those cross to Python)."""
        sort = [s for s in (sort or []) if s and s.get("col")]
        if stop <= start:
            return np.empty(0, dtype=np.int64)
        if not filters and not quick and not sort:
            return np.arange(start, min(stop, len(self)), dtype=np.int64)
        q = self._q()
        sql = (f"SELECT {POS} FROM __BASE__ WHERE {self._where(filters, quick, q)} "
               f"ORDER BY {self._order_by(sort)} LIMIT {q.i(stop - start)} OFFSET {q.i(start)}")
        return np.array([r[0] for r in self._fetch_all(sql, q.params)], dtype=np.int64)

    def positions(self, filters: Any = None, quick: str = "", sort: Any = None) -> np.ndarray:
        """Every view position, in grid order (explicitly materialised: the whole view)."""
        total = self.count(filters, quick)
        return self._window_positions(filters, quick, sort, 0, total)

    def view_rows(self, filters: Any, quick: str, sort: Any, start: int, stop: int) -> pd.DataFrame:
        return self.rows_at(self._window_positions(filters, quick, sort, start, stop))

    def rows_at(self, positions) -> pd.DataFrame:
        idx = np.asarray(list(positions) if not isinstance(positions, np.ndarray) else positions,
                         dtype=np.int64)
        if not len(idx):
            return self._schema_frame.iloc[:0].copy()
        q = self._q()
        start = int(idx[0])
        if np.array_equal(idx, np.arange(start, start + len(idx))):
            # a contiguous run (the unsorted, unfiltered window): a plain LIMIT/OFFSET, no numbering
            df = self._fetch_df(f"SELECT {self._select_list()} FROM __BASE__ LIMIT {q.i(len(idx))} OFFSET {q.i(start)}",
                                q.params)
            return df.reset_index(drop=True)
        sql = f"SELECT {self._select_list()}, {POS} FROM __BASE__ WHERE list_contains({q.p(idx.tolist())}, {POS})"
        df = self._fetch_df(sql, q.params)
        by_pos = dict(zip(df[POS].tolist(), range(len(df))))
        out = df.drop(columns=[POS]).iloc[[by_pos[int(p)] for p in idx if int(p) in by_pos]]
        return out.reset_index(drop=True)

    def to_pandas(self) -> pd.DataFrame:
        return self._fetch_df(f"SELECT {self._select_list()} FROM __BASE__ ORDER BY {POS}")

    def window(self, positions: np.ndarray) -> dict:
        return build_columnar(self.rows_at(positions), [str(int(p)) for p in positions])

    def execute(self, query: dict) -> dict:
        filters, quick, sort = query.get("filters"), query.get("quick") or "", query.get("sort")
        total = self.count(filters, quick)
        rng = query.get("range")
        if rng:
            start = max(0, int(rng.get("start", 0)))
            end = max(start, int(rng.get("end", start)))
        else:
            start, end = 0, total
        pos = self._window_positions(filters, quick, sort, start, min(end, total))
        return {"columnar": self.window(pos), "total": total}

    # --- aggregates --------------------------------------------------------------
    def _agg_sql(self, c: _SCol, fn: str) -> str:
        num = c.num
        if fn == "count":
            return "count(*)"
        if fn == "countValues":
            return f"count(*) FILTER (WHERE NOT {c.blank})"
        if fn == "distinct":
            x = num if c.ordered else c.txt
            return f"count(DISTINCT {x}) FILTER (WHERE NOT {c.blank})"
        if c.kind not in ("number", "bool", "datetime") and fn not in ("min", "max", "range"):
            return "NULL"
        quant = {"median": 0.5, "p25": 0.25, "p75": 0.75, "p90": 0.9, "p95": 0.95, "p99": 0.99}
        if fn == "sum":
            return f"COALESCE(sum({num}), 0)"
        if fn == "sumSquares":
            return f"COALESCE(sum({num} * {num}), 0)"
        if fn == "avg":
            return f"avg({num})"
        if fn == "min":
            return f"min({num})"
        if fn == "max":
            return f"max({num})"
        if fn == "range":
            return f"(max({num}) - min({num}))"
        if fn in quant:
            return f"quantile_cont({num}, {quant[fn]})"
        if fn == "iqr":
            return f"(quantile_cont({num}, 0.75) - quantile_cont({num}, 0.25))"
        if fn == "variance":
            return f"var_samp({num})"
        if fn == "stddev":
            return f"stddev_samp({num})"
        if fn == "varianceP":
            return f"var_pop({num})"
        if fn == "stddevP":
            return f"stddev_pop({num})"
        raise ValueError(f"[lattice] statistic {fn!r} is not answered by the Python engine")

    @staticmethod
    def _py(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return int(v)
        f = float(v)
        return None if math.isnan(f) else f

    def aggregates(self, query: dict, aggregates: list[dict]) -> dict:
        q = self._q()
        where = self._where(query.get("filters"), query.get("quick") or "", q)
        if not aggregates:
            return {}
        items = [f"{self._agg_sql(self.col(str(a['col'])), a['fn'])} AS a{i}" for i, a in enumerate(aggregates)]
        row = self._fetch_all(f"SELECT {', '.join(items)} FROM __BASE__ WHERE {where}", q.params)[0]
        return {a["id"]: self._py(v) for a, v in zip(aggregates, row)}

    def grouped_aggregates(self, query: dict, group_by: list, aggregates: list[dict]) -> list[dict]:
        keys = [str(k) for k in group_by]
        cols = [self.col(k) for k in keys]
        q = self._q()
        where = self._where(query.get("filters"), query.get("quick") or "", q)
        aggs = [f"{self._agg_sql(self.col(str(a['col'])), a['fn'])} AS a{i}" for i, a in enumerate(aggregates)]
        if not keys:
            row = self._fetch_all(f"SELECT {', '.join(aggs) or '1'} FROM __BASE__ WHERE {where}", q.params)[0]
            return [{"keys": [], "grouping": [], "level": 0,
                     "values": {a["id"]: self._py(v) for a, v in zip(aggregates, row)}}]
        kexp = [c.key for c in cols]
        sel = ([f"{e} AS k{i}" for i, e in enumerate(kexp)]
               + [f"GROUPING({e}) AS g{i}" for i, e in enumerate(kexp)]
               + aggs + [f"min({POS}) AS fp"])
        sql = (f"SELECT {', '.join(sel)} FROM __BASE__ WHERE {where} "
               f"GROUP BY ROLLUP({', '.join(kexp)})")
        df = self._fetch_df(sql, q.params)
        n = len(keys)
        out = []
        for row in df.itertuples(index=False):
            flags = [int(row[n + i]) for i in range(n)]
            level = n - sum(flags)
            out.append((-level, int(row[-1]), {
                "keys": [None if flags[i] else _jsonable_scalar(row[i]) for i in range(n)],
                "grouping": flags,
                "level": level,
                "values": {a["id"]: self._py(row[2 * n + j]) for j, a in enumerate(aggregates)},
            }))
        out.sort(key=lambda t: (t[0], t[1]))
        return [o[2] for o in out]

    # --- header histograms (compute/facet.js) --------------------------------------
    def facet_kind(self, c: _SCol) -> str:  # type: ignore[override]
        return {"number": "numeric", "bool": "boolean", "datetime": "date"}.get(c.kind, "category")

    def facet(self, req: dict) -> dict:
        c = self.col(str(req["colId"]))
        kind = self.facet_kind(c)
        bounds = req.get("bounds")
        computed = False
        if not bounds:
            bounds = self._facet_bounds(c, kind, req)
            computed = True
        q = self._q()
        where = self._where(req.get("filters"), req.get("quick") or "", q)
        out = {"bounds": bounds, "counts": self._facet_counts(c, bounds, where, q)}
        if computed:
            out["unfiltered"] = (list(out["counts"]) if where == "TRUE"
                                 else self._facet_counts(c, bounds, "TRUE", self._q()))
        return out

    def _category_tally(self, c: _SCol, where: str, q: _Q) -> tuple[dict, int, list]:
        """Per-text counts (first-seen order) and the blank count, over ``where``."""
        t = c.txt
        rows = self._fetch_all(
            f"SELECT {t} AS t, count(*) AS n FROM __BASE__ WHERE {where} AND {t} IS NOT NULL AND {t} <> '' "
            f"GROUP BY {t} ORDER BY min({POS})", q.params)
        nulls = int(self._scalar(f"SELECT count(*) FROM __BASE__ WHERE {where} AND {c.blank}", q.params))
        return {r[0]: int(r[1]) for r in rows}, nulls, [r[0] for r in rows]

    def _facet_bounds(self, c: _SCol, kind: str, req: dict) -> dict:
        wanted = max(1, int(req.get("buckets") or DEFAULT_BUCKETS))
        if kind == "boolean":
            b = [{"value": False, "label": "false"}, {"value": True, "label": "true"}]
            if self._scalar(f"SELECT count(*) FROM __BASE__ WHERE {c.miss}"):
                b.append({"null": True, "label": "Empty"})
            return {"kind": "boolean", "buckets": b}
        if kind == "category":
            card = int(self._scalar(
                f"SELECT count(DISTINCT {c.txt}) FILTER (WHERE {c.txt} <> '') FROM __BASE__"))
            if card > DEFAULT_CARDINALITY_LIMIT:
                return {"kind": "category", "buckets": [], "suppressed": "cardinality", "cardinality": card}
            tally, nulls, first = self._category_tally(c, "TRUE", self._q())
            order = sorted(first, key=lambda t: -tally[t])
            buckets = [{"value": t, "label": t} for t in order]
            if nulls:
                buckets.append({"null": True, "label": "Empty"})
            return {"kind": "category", "buckets": buckets, "cardinality": len(tally)}

        num = c.num
        total, finite, vmin, vmax = self._fetch_all(
            f"SELECT count(*), count(*) FILTER (WHERE isfinite({num})), "
            f"min({num}) FILTER (WHERE isfinite({num})), max({num}) FILTER (WHERE isfinite({num})) "
            f"FROM __BASE__")[0]
        nulls = int(total - finite)
        if not finite:
            return {"kind": kind, "buckets": [{"null": True, "label": "Empty"}] if nulls else [], "empty": True}
        vmin, vmax = float(vmin), float(vmax)
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
            n = int(total)
            step = n / QUANTILE_SAMPLE if n > QUANTILE_SAMPLE else 1
            picks, s = [], 0.0
            while s < n:
                picks.append(int(math.floor(s)))
                s += step
            q = self._q()
            rows = self._fetch_all(
                f"SELECT {num} AS v FROM __BASE__ WHERE list_contains({q.p(picks)}, {POS}) AND isfinite({num}) "
                f"ORDER BY v", q.params)
            sample = [float(r[0]) for r in rows]
            if sample:
                edges = [sample[0]]
                for b in range(1, wanted):
                    v = sample[min(len(sample) - 1, int(math.floor((b / wanted) * len(sample))))]
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

    def _facet_counts(self, c: _SCol, bounds: dict, where: str, q: _Q) -> list[int]:
        buckets = bounds.get("buckets") or []
        if not buckets:
            return []
        has_null = bool(buckets[-1].get("null"))
        null_at = len(buckets) - 1
        counts = [0] * len(buckets)
        kind = bounds.get("kind")
        if kind == "boolean":
            n0, n1, nm = self._fetch_all(
                f"SELECT count(*) FILTER (WHERE {c.num} = 0), count(*) FILTER (WHERE {c.num} <> 0), "
                f"count(*) FILTER (WHERE {c.miss}) FROM __BASE__ WHERE {where}", q.params)[0]
            counts[0], counts[1] = int(n0), int(n1)
            if has_null:
                counts[null_at] = int(nm)
            return counts
        if kind == "category":
            tally, nulls, _ = self._category_tally(c, where, q)
            rem = next((i for i, b in enumerate(buckets) if b.get("remainder")), -1)
            seen = 0
            for i, b in enumerate(buckets):
                if b.get("null") or b.get("remainder"):
                    continue
                k = int(tally.get(str(b.get("value")), 0))
                counts[i] = k
                seen += k
            if rem >= 0:
                counts[rem] = sum(tally.values()) - seen
            if has_null:
                counts[null_at] = nulls
            return counts
        num = c.num
        ordered = len(buckets) - 1 if has_null else len(buckets)
        edges = [float(b["from"]) for b in buckets[:ordered]] + ([float(buckets[ordered - 1]["to"])] if ordered else [])
        sel = [f"count(*) FILTER (WHERE NOT COALESCE(isfinite({num}), FALSE))"]
        strict = all(edges[i] < edges[i + 1] for i in range(len(edges) - 1))
        if ordered and strict:
            top = edges[-1]
            for i in range(ordered):
                lo = q.p(edges[i])
                if i < ordered - 1:
                    sel.append(f"count(*) FILTER (WHERE {num} >= {lo} AND {num} < {q.p(edges[i + 1])})")
                else:
                    sel.append(f"count(*) FILTER (WHERE {num} >= {lo} AND {num} <= {q.p(top)})")
            row = self._fetch_all(f"SELECT {', '.join(sel)} FROM __BASE__ WHERE {where}", q.params)[0]
            counts[:ordered] = [int(x) for x in row[1:]]
        elif ordered:
            # edges that repeat: the bucket is the last edge not above the value (searchsorted, side=right)
            # a sum of comparisons, not a SQL lambda: the lambda syntax differs between duckdb 1.x and 2.x
            below = " + ".join(f"CASE WHEN {q.p(x)} <= {num} THEN 1 ELSE 0 END" for x in edges)
            top = q.p(edges[-1])
            rows = self._fetch_all(
                f"SELECT CASE WHEN {num} = {top} THEN {ordered - 1} ELSE ({below}) - 1 END AS bin_ix, "
                f"count(*) FROM __BASE__ WHERE {where} AND {num} >= {q.p(edges[0])} AND {num} <= {top} "
                f"GROUP BY bin_ix", q.params)
            row = [0]
            for at, n in rows:
                if 0 <= at < ordered:
                    counts[int(at)] += int(n)
            row = [int(self._scalar(f"SELECT {sel[0]} FROM __BASE__ WHERE {where}", q.params))]
        else:
            row = [int(self._scalar(f"SELECT {sel[0]} FROM __BASE__ WHERE {where}", q.params))]
        if has_null:
            counts[null_at] = int(row[0])
        return counts

    # --- the Statistics panel's profile ---------------------------------------------
    def profile(self, field: str, filters: Any = None, quick: str = "") -> dict:
        c = self.col(field)
        q = self._q()
        where = self._where(filters, quick, q)
        rows, absent = self._fetch_all(
            f"SELECT count(*), count(*) FILTER (WHERE {c.blank}) FROM __BASE__ WHERE {where}", q.params)[0]
        rows, absent = int(rows), int(absent)
        numeric_col = c.kind == "number"
        if c.kind in ("text", "category"):
            # isNumericColumn: the first 200 present values decide it
            head = self._fetch_all(
                f"SELECT {c.txt} FROM __BASE__ WHERE {where} ORDER BY {POS} LIMIT 20000", q.params)
            sample = [r[0] for r in head if r[0] is not None and r[0] != ""][:200]
            if sample:
                good = sum(1 for t in sample if math.isfinite(_num_target(t)))
                numeric_col = good / len(sample) >= 0.9
        # a datetime column's numbers are not profiled (the grid profiles dates as text)
        has_num = c.kind in ("number", "bool") or (c.kind in ("text", "category") and numeric_col)
        n = 0
        stats: dict[str, Any] = {}
        if has_num:
            num = c.num
            fin = f"isfinite({num})"
            # one list-form quantile shares a single sort (three separate calls hold three copies)
            n, vmin, vmax, mean, qs, sd = self._fetch_all(
                f"SELECT count(*), min({num}), max({num}), avg({num}), quantile_cont({num}, [0.25, 0.5, 0.75]), "
                f"stddev_samp({num}) FROM __BASE__ WHERE {where} AND {fin}", q.params)[0]
            n = int(n)
            q1, med, q3 = qs if qs else (None, None, None)
            stats = dict(vmin=vmin, vmax=vmax, mean=mean, med=med, q1=q1, q3=q3, sd=sd)
        distinct = int(self._scalar(
            f"SELECT count(DISTINCT {c.num if c.ordered and c.kind != 'datetime' else c.txt}) "
            f"FROM __BASE__ WHERE {where} AND NOT {c.blank}", q.params))
        out: dict[str, Any] = {
            "column": c.name, "rows": rows, "present": rows - absent, "missing": absent, "numeric": n,
            "distinct": distinct,
        }
        if n:
            vmin, vmax = float(stats["vmin"]), float(stats["vmax"])
            q1, q3 = float(stats["q1"]), float(stats["q3"])
            iqr = q3 - q1
            outliers = self._outliers(c, where, q, q1, q3)
            out.update({
                "min": vmin, "max": vmax, "mean": float(stats["mean"]), "median": float(stats["med"]),
                "q1": q1, "q3": q3, "iqr": iqr,
                "stddev": float(stats["sd"]) if n > 1 and stats["sd"] is not None else None,
                "outliers": outliers, "histogram": self._histogram(c, where, q, n, vmin, vmax, q1, q3),
            })
        else:
            out.update({"min": None, "max": None, "mean": None, "median": None, "q1": None, "q3": None,
                        "iqr": None, "stddev": None, "outliers": 0, "histogram": []})
        if not numeric_col:
            out["topValues"] = self._top_values(c, where, q)
        return out

    def _outliers(self, c: _SCol, where: str, q: _Q, q1: float, q3: float) -> int:
        iqr = q3 - q1
        lo, hi = q.p(q1 - 1.5 * iqr), q.p(q3 + 1.5 * iqr)
        num = c.num
        return int(self._scalar(
            f"SELECT count(*) FROM __BASE__ WHERE {where} AND isfinite({num}) AND ({num} < {lo} OR {num} > {hi})",
            q.params))

    def _histogram(self, c: _SCol, where: str, q: _Q, n: int, vmin: float, vmax: float,
                   q1: float, q3: float, cap: int = 20) -> list[dict]:
        """``statistics.histogram`` (Freedman-Diaconis bins over Tukey's fences), counted in SQL."""
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
        num = c.num
        rows = self._fetch_all(
            f"SELECT CAST(least(greatest(floor(({num} - {q.p(lo)}) / {q.p(step)}), 0), {count - 1}) AS BIGINT) AS bin, "
            f"count(*) FROM __BASE__ WHERE {where} AND isfinite({num}) GROUP BY bin", q.params)
        tally = [0] * count
        for b, k in rows:
            tally[int(b)] += int(k)
        bins = [{"from": lo + i * step, "to": lo + (i + 1) * step, "count": tally[i]} for i in range(count)]
        bins[0]["from"] = vmin
        bins[-1]["to"] = vmax
        return bins

    def _top_values(self, c: _SCol, where: str, q: _Q, k: int = PROFILE_TOP_VALUES) -> list[dict]:
        key = c.key
        total = int(self._scalar(f"SELECT count(*) FROM __BASE__ WHERE {where} AND NOT {c.blank}", q.params))
        if not total:
            return []
        rows = self._fetch_all(
            f"SELECT {key} AS v, count(*) AS n FROM __BASE__ WHERE {where} AND NOT {c.blank} GROUP BY v "
            f"ORDER BY n DESC, v ASC LIMIT {q.i(max(1, k))}", q.params)
        items = [(_jsonable_scalar(pd.Timestamp(v) if isinstance(v, datetime.datetime) else v), int(n))
                 for v, n in rows]
        items.sort(key=lambda kv: (-kv[1], _default_order(kv[0])))
        return [{"value": v, "count": n, "share": n / total} for v, n in items]


def from_parquet(path: Any) -> DuckDBEngine:
    """A Parquet file queried in place through a private DuckDB connection (``pushdown=True``)."""
    import pathlib
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError('[lattice] a DuckDB source needs "duckdb": pip install "lattice-grid-jupyter[large]"') from exc
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[lattice] no Parquet file at {p}")
    con = duckdb.connect()
    con.read_parquet(str(p)).create_view("lg_parquet")
    return DuckDBEngine(con=con, table="lg_parquet", keep_alive=con)


__all__ = ["DuckDBEngine", "from_parquet", "is_duckdb_relation", "is_duckdb_connection", "quote_ident"]
