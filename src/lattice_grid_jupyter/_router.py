"""
LatticeRouter -- the grid's data router, declared from Python (card 1620).

    from lattice_grid_jupyter import LatticeRouter, LatticeGridWidget

    router = LatticeRouter(
        sources={
            "customer": {"data": df_c, "key": "customer_id",
                         "fields": {"signup": "date"},
                         "join": [
                             {"from": "txn", "many": True, "foreign_key": "customer_id",
                              "aggregate": {"n_txn": {"fn": "count"},
                                            "spend": {"fn": "sum", "field": "amount"}}},
                             {"from": "label", "local_key": "customer_id", "fields": ["churned"]},
                         ]},
            "txn": {"data": df_t, "key": "txn_id"},
            "label": {"data": df_l, "key": "customer_id"},
        },
        routes={"customer": "customer"},
    )
    w = LatticeGridWidget(router, route="customer")   # the shaped rows
    ...
    router.update("txn", df_t2)    # a keyed diff: only the affected customers re-emit
    w.view                         # the shaped rows, as a DataFrame (a feature table)

The router runs in the browser on the grid's own ``data-router`` module: join,
rollup-onto-parent, spread, unnest and field coercion are that module's semantics,
not a Python reimplementation. This file only declares the specs as data
(``snake_case`` accepted, an unknown key reported by name, functions refused),
validates them, and computes the keyed diff ``update`` sends.
"""

from __future__ import annotations

import re
import warnings
import weakref
from typing import Any

import pandas as pd

from . import _serialize as S
from ._options import LatticeGridWarning

_SNAKE = re.compile(r"_([a-z0-9])")

# the router's own spec vocabulary (data-router.d.ts, grid 1.84.0)
SOURCE_KEYS = frozenset({"data", "key", "join", "unnest", "fields"})
JOIN_KEYS = frozenset({
    "from", "localKey", "on", "foreignKey", "fromKey", "many", "as", "aggregate", "fields", "select",
    "distinct", "sort", "missing", "onParentDelete", "spread",
})
AGGREGATE_KEYS = frozenset({"fn", "field", "orderBy"})
SPREAD_KEYS = frozenset({"name", "value", "prefix", "include", "type"})
UNNEST_KEYS = frozenset({"path", "as", "key", "parentKey", "keep", "onParentDelete", "join", "unnest"})
FIELD_KEYS = frozenset({"type", "format", "decimal", "true", "false", "nulls"})
ROUTE_KEYS = frozenset({"source"})
FIELD_TYPES = ("text", "number", "integer", "boolean", "date", "json")


def _camel(key: str) -> str:
    return _SNAKE.sub(lambda m: m.group(1).upper(), key) if "_" in key and not key.startswith("_") else key


def _refuse_function(value: Any, path: str) -> None:
    if callable(value):
        raise TypeError(
            f"router spec {path!r} is a function ({value!r}). Specs are data: a function cannot be sent to "
            "the browser. Use the declarative form (a field list, an aggregate map, a type string).")


def _map(spec: dict, known: frozenset, path: str) -> dict:
    """snake_case -> camelCase for one level; an unknown key is warned by name and left out."""
    out: dict = {}
    for key, value in spec.items():
        _refuse_function(value, f"{path}.{key}")
        name = key if key in known else _camel(key)
        if name not in known:
            warnings.warn(
                f"[lattice] router.unknown:{path}.{key}: {key!r} is not a key a router {path.split('.')[-1]} "
                f"spec recognises ({', '.join(sorted(known))}) and was ignored",
                LatticeGridWarning, stacklevel=5)
            continue
        out[name] = value
    return out


def _plain(value: Any, path: str) -> Any:
    """A value that must be plain data all the way down (no functions)."""
    if isinstance(value, dict):
        return {k: _plain(v, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v, f"{path}[]") for v in value]
    _refuse_function(value, path)
    return value


def _join_spec(raw: dict, path: str) -> dict:
    j = _map(raw, JOIN_KEYS, path)
    if "aggregate" in j:
        agg = j["aggregate"]
        if not isinstance(agg, dict):
            raise TypeError(f"{path}.aggregate takes a dict of name -> {{'fn', 'field'}}")
        j["aggregate"] = {name: _map(dict(a), AGGREGATE_KEYS, f"{path}.aggregate.{name}") for name, a in agg.items()}
    if "spread" in j:
        j["spread"] = _map(dict(j["spread"]), SPREAD_KEYS, f"{path}.spread")
    for k in ("fields", "select"):
        if k in j:
            j[k] = _plain(j[k], f"{path}.{k}")
    return j


def _joins(raw: Any, path: str) -> Any:
    if isinstance(raw, dict):
        return _join_spec(raw, path)
    return [_join_spec(dict(r), path) for r in raw]


def _unnests(raw: Any, path: str) -> Any:
    def one(u: dict) -> dict:
        m = _map(dict(u), UNNEST_KEYS, path)
        if "join" in m:
            m["join"] = _joins(m["join"], f"{path}.join")
        if "unnest" in m:
            m["unnest"] = _unnests(m["unnest"], f"{path}.unnest")
        return m
    return one(raw) if isinstance(raw, dict) else [one(dict(u)) for u in raw]


def _fields(raw: dict, path: str) -> dict:
    out = {}
    for name, spec in raw.items():
        if isinstance(spec, str):
            if spec not in FIELD_TYPES:
                warnings.warn(f"[lattice] router.unknown:{path}.{name}: {spec!r} is not a field type "
                              f"({', '.join(FIELD_TYPES)}); the field passes through uncoerced",
                              LatticeGridWarning, stacklevel=4)
                continue
            out[name] = spec
        else:
            out[name] = _map(dict(spec), FIELD_KEYS, f"{path}.{name}")
    return out


def _unnest_ids(raw: Any) -> list[str]:
    ids: list[str] = []
    for u in ([raw] if isinstance(raw, dict) else list(raw or [])):
        if "as" in u:
            ids.append(u["as"])
        ids += _unnest_ids(u.get("unnest"))
    return ids


class _Source:
    """One declared source: its frame, key column and the router options (camelCase)."""

    def __init__(self, name: str, value: Any):
        if isinstance(value, pd.DataFrame):
            value = {"data": value}
        elif not isinstance(value, dict):
            raise TypeError(f"source {name!r} must be a DataFrame or a dict with 'data'")
        spec = dict(value)
        _refuse_function(spec.get("map"), f"sources.{name}.map")
        spec = _map(spec, SOURCE_KEYS, f"sources.{name}")
        df = spec.get("data")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"source {name!r} needs a pandas DataFrame under 'data'")
        self.name = name
        self.key = self._key_column(name, df, spec.get("key"))
        self.frame = self._frame(df)
        opts: dict = {}
        if "join" in spec:
            opts["join"] = _joins(spec["join"], f"sources.{name}.join")
            joins = [opts["join"]] if isinstance(opts["join"], dict) else opts["join"]
            for j in joins:  # a collect / rollup / spread join matches on THIS source's key column
                if (j.get("many") or j.get("spread")) and "localKey" not in j and "on" not in j:
                    j["localKey"] = self.key
        if "unnest" in spec:
            opts["unnest"] = _unnests(spec["unnest"], f"sources.{name}.unnest")
        if "fields" in spec:
            opts["fields"] = _fields(dict(spec["fields"]), f"sources.{name}.fields")
        self.options = opts

    @staticmethod
    def _key_column(name: str, df: pd.DataFrame, key: str | None) -> str:
        if key is None:
            for cand in ("id", f"{name}_id"):
                if cand in df.columns:
                    return cand
            raise ValueError(f"source {name!r}: name its key column, e.g. {{'data': df, 'key': '{name}_id'}} "
                             "(a column, or the name of the index)")
        if key not in df.columns and df.index.name != key:
            raise ValueError(f"source {name!r}: key {key!r} is not a column (or the index name) of its DataFrame")
        return key

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.reset_index() if self.key not in df.columns else df.copy()
        if frame[self.key].duplicated().any():
            dup = frame[self.key][frame[self.key].duplicated()].iloc[0]
            raise ValueError(f"source {self.name!r}: key {self.key!r} is not unique (e.g. {dup!r}); "
                             "a keyed diff and the router's identity need one row per key")
        return frame.reset_index(drop=True)

    def keys(self, frame: pd.DataFrame | None = None) -> list[str]:
        return [str(k) for k in (self.frame if frame is None else frame)[self.key]]

    def columnar(self, frame: pd.DataFrame | None = None) -> dict:
        f = self.frame if frame is None else frame
        out = S.build_columnar(f, self.keys(f))
        for col in f.columns:  # a nested list / dict column (for an unnest) travels as data, not as text
            if f[col].dtype == object and any(isinstance(v, (list, dict)) for v in f[col]):
                out[str(col)] = [_nested(v) for v in f[col]]
        return out


def _rows_of(columnar: dict, keys: list[str]) -> dict:
    return {k: [v[i] for i in keys] for k, v in columnar.items()}


class LatticeRouter:
    """A set of DataFrame sources shaped by the grid's data router, shown through
    ``LatticeGridWidget(router, route=...)``. See the module docs."""

    def __init__(self, sources: dict, routes: dict | None = None):
        if not isinstance(sources, dict) or not sources:
            raise TypeError("LatticeRouter(sources={...}) takes a dict of source name -> DataFrame or spec dict")
        self._sources = {str(n): _Source(str(n), v) for n, v in sources.items()}
        known = set(self._sources)
        for s in self._sources.values():
            known |= set(_unnest_ids(s.options.get("unnest")))
        for name, s in self._sources.items():
            for j in ([s.options["join"]] if isinstance(s.options.get("join"), dict) else s.options.get("join") or []):
                if j.get("from") not in known:
                    raise ValueError(f"source {name!r}: join from {j.get('from')!r}, which is not a source "
                                     f"(sources: {sorted(known)})")
        self._routes = self._routes_from(routes, known)
        self._widgets: list[weakref.ref] = []
        self.updates = 0

    @staticmethod
    def _routes_from(routes: dict | None, known: set) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, value in (routes or {n: n for n in known}).items():
            if isinstance(value, dict):
                value = _map(dict(value), ROUTE_KEYS, f"routes.{name}").get("source", name)
            value = name if value is None else str(value)
            if value not in known:
                raise ValueError(f"route {name!r} reads {value!r}, which is not a source or an unnest 'as' "
                                 f"(known: {sorted(known)})")
            out[str(name)] = value
        return out

    # --- what the widgets read -------------------------------------------------------
    @property
    def sources(self) -> list[str]:
        """The declared source names."""
        return list(self._sources)

    @property
    def routes(self) -> dict[str, str]:
        """Route name -> the source (or unnest ``as``) whose shaped rows it shows."""
        return dict(self._routes)

    def frame(self, source: str) -> pd.DataFrame:
        """The current DataFrame of a source (after ``update``)."""
        return self._sources[source].frame.copy()

    def _load_order(self) -> list[str]:
        """Sources other sources join FROM first, so the browser settles each parent once."""
        order: list[str] = []

        def visit(n: str) -> None:
            if n in order or n not in self._sources:
                return
            for j in ([self._sources[n].options["join"]] if isinstance(self._sources[n].options.get("join"), dict)
                      else self._sources[n].options.get("join") or []):
                visit(j.get("from"))
            order.append(n)

        for n in self._sources:
            visit(n)
        return order

    def _payload(self, route: str) -> dict:
        return {
            "route": self._routes[route],
            "loadOrder": self._load_order(),
            "sources": {n: {"columnar": s.columnar(), "options": s.options} for n, s in self._sources.items()},
        }

    def _date_fields(self) -> set[str]:
        out: set[str] = set()
        for s in self._sources.values():
            for f, spec in (s.options.get("fields") or {}).items():
                if spec == "date" or (isinstance(spec, dict) and spec.get("type") == "date"):
                    out.add(f)
        return out

    def _attach(self, widget: Any) -> None:
        self._widgets = [w for w in self._widgets if w() is not None] + [weakref.ref(widget)]

    # --- a keyed diff --------------------------------------------------------------
    def update(self, source: str, df: pd.DataFrame) -> dict:
        """Replace one source's rows and send the browser the KEYED DIFF: rows added or
        changed (upserts) and keys removed. Only the parents the diff reaches re-emit.

        Returns ``{"added": n, "updated": n, "removed": n}`` (the diff, as counted here)."""
        if source not in self._sources:
            raise KeyError(f"unknown source {source!r}; the router has {self.sources}")
        s = self._sources[source]
        new = s._frame(df)
        if list(new.columns) != list(s.frame.columns):
            raise ValueError(f"source {source!r}: update() takes the same columns as the source was declared with "
                             f"({list(s.frame.columns)}), got {list(new.columns)}")
        old_c, new_c = s.columnar(), s.columnar(new)
        old_keys, new_keys = s.keys(), s.keys(new)
        fields = [f for f in new_c if f != S.ROW_KEY]
        old_row = {k: tuple(map(_hashable, (old_c[f][i] for f in fields))) for i, k in enumerate(old_keys)}
        new_pos = {k: i for i, k in enumerate(new_keys)}
        added = [k for k in new_keys if k not in old_row]
        changed = [k for k in new_keys if k in old_row
                   and tuple(map(_hashable, (new_c[f][new_pos[k]] for f in fields))) != old_row[k]]
        removed = [k for k in old_keys if k not in new_pos]
        s.frame = new
        upsert = added + changed
        diff = {"added": len(added), "updated": len(changed), "removed": len(removed)}
        self.updates += 1
        live = [w() for w in self._widgets if w() is not None]
        for w in live:
            # a view that renders LATER reads the synced state: keep it current without
            # re-sending the whole payload (the diff below is what the live views get)
            w._trait_values["_router"] = self._payload(w.route)
        if upsert or removed:
            msg = {"type": "lg:router", "source": source,
                   "upsert": _rows_of(new_c, [new_pos[k] for k in upsert]) if upsert else {S.ROW_KEY: []},
                   "remove": removed}
            for w in live:
                w.send(msg)
        return diff


def _nested(v: Any) -> Any:
    """A nested value made JSON-safe (numpy scalars and arrays, NaN -> null)."""
    if isinstance(v, dict):
        return {str(k): _nested(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)) or hasattr(v, "tolist") and not isinstance(v, (str, bytes)) and hasattr(v, "__len__"):
        return [_nested(x) for x in (v.tolist() if hasattr(v, "tolist") else v)]
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and v != v:
        return None
    return v


def _hashable(v: Any) -> Any:
    if isinstance(v, (list, dict)):
        return repr(v)
    if v != v:  # NaN
        return "__nan__"
    return v


def shaped_frame(columnar: dict, date_fields: set[str]) -> pd.DataFrame:
    """The route's shaped rows (as the browser reports them) as a typed DataFrame."""
    data = {k: v for k, v in (columnar or {}).items() if k != S.ROW_KEY}
    df = pd.DataFrame(data)
    for col in df.columns:
        if col in date_fields:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
        elif df[col].dtype == object and not any(isinstance(v, (list, dict)) for v in df[col].head(50)):
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
    return df
