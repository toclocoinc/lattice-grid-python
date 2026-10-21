"""
The data router's shaped rows, computed in the kernel (BACKLOG-0005044).

The grid's ``data-router`` module shapes a ``LatticeRouter``'s sources in the browser:
field coercion, lookup / collect / rollup / spread joins and unnest. ``widget.view``
used to need the browser's report of those rows, which a cell cannot wait for under
Run All, in nbconvert/papermill or with no front end at all. This module computes the
same rows from what the widget already holds -- the source frames and the router
declaration -- so the view is synchronous and needs no browser.

It follows the module's own semantics, written against that module's behaviour rather
than a looser reading of it: JavaScript's ``String()`` / ``Number()`` conversions, the
join key matching on ``String(value)``, ``null`` versus ``undefined``, the order rows
leave the router. ``tests/test_router_parity.py`` runs one declaration through the
vendored bundle under node and through this module and compares row for row, so a
difference is a failing test rather than a surprise.

What it does not reproduce raises :class:`RouterEngineUnsupported`, and the widget
falls back to waiting for the browser's report: today that is a date string only the
browser's own ``Date`` parser reads (a non-ISO text date with no ``format``).
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
from decimal import Decimal
from functools import cmp_to_key
from typing import Any

ROW_ID = "__row_id__"
SOURCE = "__source"


class RouterEngineUnsupported(Exception):
    """A declaration whose result only the browser's engine can produce."""


class _Undefined:
    """JavaScript ``undefined`` -- a field a row does not have, distinct from ``null``."""

    def __repr__(self) -> str:
        return "undefined"


UNDEF = _Undefined()


# --- JavaScript value semantics -----------------------------------------------------
_JS_WS = " \t\n\r\x0b\x0c             " \
         "    　﻿"
_NUM_RE = re.compile(r"^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
_RADIX_RE = re.compile(r"^0(?:[xX][0-9a-fA-F]+|[oO][0-7]+|[bB][01]+)$")
NAN = float("nan")


def _nullish(v: Any) -> bool:
    return v is None or v is UNDEF


def _num_to_str(x: float) -> str:
    """``String(number)`` -- the shortest round-trip digits, JavaScript's layout."""
    if x != x:
        return "NaN"
    if x in (math.inf, -math.inf):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0:
        return "0"
    if x < 0:
        return "-" + _num_to_str(-x)
    if x == int(x) and x < 2 ** 53:
        return str(int(x))
    _sign, dig, exp = Decimal(repr(x)).as_tuple()
    digits = "".join(map(str, dig)).rstrip("0") or "0"
    point = len(dig) + exp
    k, n = len(digits), point
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + digits
    e = n - 1
    sign = "+" if e >= 0 else "-"
    return (digits if k == 1 else digits[0] + "." + digits[1:]) + "e" + sign + str(abs(e))


def js_str(v: Any) -> str:
    """JavaScript ``String(v)``."""
    if v is None:
        return "null"
    if v is UNDEF:
        return "undefined"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return _num_to_str(v)
    if isinstance(v, (list, tuple)):
        return ",".join("" if _nullish(x) else js_str(x) for x in v)
    return "[object Object]"


def js_number(v: Any) -> float:
    """JavaScript ``Number(v)`` (``NaN`` where it is NaN)."""
    if v is UNDEF:
        return NAN
    if v is None:
        return 0.0
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip(_JS_WS)
        if s == "":
            return 0.0
        if _RADIX_RE.match(s):
            return float(int(s[2:], {"x": 16, "o": 8, "b": 2}[s[1].lower()]))
        if s in ("Infinity", "+Infinity"):
            return math.inf
        if s == "-Infinity":
            return -math.inf
        if _NUM_RE.match(s):
            return float(s)
        return NAN
    if isinstance(v, (list, tuple)):
        return 0.0 if not v else js_number(v[0]) if len(v) == 1 and not isinstance(v[0], (list, dict)) else NAN
    return NAN


def _finite(x: float) -> bool:
    return x == x and x not in (math.inf, -math.inf)


def _strict_eq(a: Any, b: Any) -> bool:
    """JavaScript ``===`` over JSON values (a number is a number, ``true`` is not ``1``)."""
    if a is b:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    return False


def _js_compare(a: Any, b: Any) -> float:
    """The module's ``$n`` comparator: nullish first, numbers by value, else by ``String()``."""
    if _strict_eq(a, b):
        return 0
    if _nullish(a):
        return -1
    if _nullish(b):
        return 1
    if _is_num(a) and _is_num(b):
        return a - b
    x, y = js_str(a), js_str(b)
    return -1 if x < y else 1 if x > y else 0


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _clean(v: Any) -> Any:
    """What survives ``JSON.parse(JSON.stringify(v))``: no ``undefined`` in an object, ``null`` in an
    array, NaN/Infinity as ``null`` and a whole-valued number as an integer."""
    if v is UNDEF or v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        if not _finite(v):
            return None
        return int(v) if v == int(v) and abs(v) < 1e21 else v
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items() if x is not UNDEF}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def _json_text(v: Any) -> str:
    return json.dumps(_clean(v), separators=(",", ":"), ensure_ascii=False)


# --- dates: the module's toIsoDate and the pattern reader -------------------------------
_ISO_PREFIX = re.compile(r"^([0-9]{4}-[0-9]{2}-[0-9]{2})")
_TOKENS = ["yyyy", "yy", "MMMM", "MMM", "MM", "M", "dd", "d", "EEEE", "EEE", "HH", "H", "hh", "h",
           "mm", "m", "ss", "s", "SSS", "a"]
_PATTERN_RE = {"yyyy": "([0-9]{4})", "yy": "([0-9]{2})", "MM": "([0-9]{1,2})", "M": "([0-9]{1,2})",
               "dd": "([0-9]{1,2})", "d": "([0-9]{1,2})", "HH": "([0-9]{1,2})", "mm": "([0-9]{1,2})",
               "ss": "([0-9]{1,2})"}


def _scan_pattern(pattern: str) -> list[tuple[str | None, str]]:
    out: list[tuple[str | None, str]] = []
    text, i = "", 0

    def flush() -> None:
        nonlocal text
        if text:
            out.append((None, text))
            text = ""

    while i < len(pattern):
        ch = pattern[i]
        if ch == "'":
            if pattern[i + 1:i + 2] == "'":
                text += "'"
                i += 2
                continue
            end = pattern.find("'", i + 1)
            if end == -1:
                text += pattern[i + 1:]
                i = len(pattern)
                continue
            text += pattern[i + 1:end]
            i = end + 1
            continue
        token = next((t for t in _TOKENS if pattern.startswith(t, i)), None)
        if token:
            flush()
            out.append((token, token))
            i += len(token)
            continue
        text += ch
        i += 1
    flush()
    return out


def _iso_of_date(d: _dt.datetime) -> str:
    return f"{d.year}-{d.month:02d}-{d.day:02d}"


def _iso_date(value: Any) -> str | None:
    """The module's ``toIsoDate``: the ``yyyy-mm-dd`` of a string, or of a number read as epoch ms."""
    if _nullish(value) or value == "":
        return None
    if isinstance(value, str):
        m = _ISO_PREFIX.match(value.strip(_JS_WS))
        if m:
            return m.group(1)
        # any other text goes through the browser's own Date parser, which is not reproduced
        raise RouterEngineUnsupported(
            f"the date text {value!r} is read by the browser's Date parser; give the field a "
            "'format' (e.g. 'dd/MM/yyyy') or use ISO yyyy-mm-dd text")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _iso_of_epoch_ms(float(value))
    return None


def _iso_of_epoch_ms(ms: float) -> str | None:
    if not _finite(ms) or abs(ms) > 8.64e15:
        return None
    try:
        return _iso_of_date(_dt.datetime.fromtimestamp(ms / 1000.0))
    except (OverflowError, OSError, ValueError):
        raise RouterEngineUnsupported(f"the date {ms:g} (epoch ms) is outside the range this kernel reads") from None


def _date_from_pattern(value: Any, pattern: str) -> str | None:
    if value is None or value is UNDEF:
        return None
    text = js_str(value).strip(_JS_WS)
    if text == "":
        return None
    rx, order = "^", []
    for token, raw in _scan_pattern(pattern):
        if token and token in _PATTERN_RE:
            rx += _PATTERN_RE[token]
            order.append(token)
        elif token:
            return None
        else:
            rx += re.escape(raw)
    m = re.match(rx + r"\Z", text)
    if not m:
        return None
    c = {"year": 0, "month": 1, "day": 1, "hour": 0, "minute": 0, "second": 0}
    for i, token in enumerate(order):
        g = int(m.group(i + 1))
        if token == "yyyy":
            c["year"] = g
        elif token == "yy":
            c["year"] = 2000 + g
        elif token in ("MM", "M"):
            c["month"] = g
        elif token in ("dd", "d"):
            c["day"] = g
        elif token == "HH":
            c["hour"] = g
        elif token == "mm":
            c["minute"] = g
        elif token == "ss":
            c["second"] = g
    year = c["year"] + 1900 if 0 <= c["year"] <= 99 else c["year"]  # new Date(y, ...) reads 0-99 as 19yy
    months = year * 12 + (c["month"] - 1)
    y, mo = divmod(months, 12)
    try:
        d = _dt.datetime(y, mo + 1, 1) + _dt.timedelta(days=c["day"] - 1, hours=c["hour"],
                                                       minutes=c["minute"], seconds=c["second"])
    except (OverflowError, ValueError):
        return None
    return _iso_of_date(d)


# --- field coercion (the module's `fields`) -----------------------------------------------
_FIELD_TYPES = ("text", "number", "integer", "boolean", "date", "json")
_SPREAD_TYPES = ("number", "date", "boolean", "text")
_STRIP_NUM = re.compile(r"[\s   ']")
_EXPRESSION = re.compile(r"[0-9]\s*[*/^]\s*[0-9.(]")
_NON_NUM = re.compile(r"[^0-9eE+\-.]")


def _field_spec(name: str, spec: Any, types: tuple, nulls_ok: bool) -> dict | None:
    if isinstance(spec, str):
        return {"name": name, "type": spec, "format": None, "decimal": None, "true": None, "false": None,
                "nulls": None} if spec in types else None
    if isinstance(spec, dict):
        t = None if spec.get("type") is None else str(spec["type"])
        if t is not None and t not in types:
            return None
        if t is None and not nulls_ok:
            return None

        def as_set(key: str):
            v = spec.get(key)
            return {js_str(x) for x in v} if isinstance(v, list) and v else None

        return {"name": name, "type": t, "format": spec["format"] if isinstance(spec.get("format"), str) else None,
                "decimal": spec["decimal"] if spec.get("decimal") in (",", ".") else None,
                "true": as_set("true"), "false": as_set("false"), "nulls": as_set("nulls") if nulls_ok else None}
    return None


def _field_specs(fields: Any) -> list[dict]:
    if not isinstance(fields, dict):
        return []
    out = []
    for name, spec in fields.items():
        s = _field_spec(str(name), spec, _FIELD_TYPES, True)
        if s:
            out.append(s)
    return out


def _parse_number(value: Any, decimal: str = ".") -> float | None:
    """The module's number reader: tolerant of thousands separators, refuses expressions."""
    if _is_num(value):
        return float(value) if _finite(float(value)) else None
    if not isinstance(value, str):
        return None
    n = value.strip(_JS_WS)
    if n == "" or _EXPRESSION.search(n):
        return None
    n = _STRIP_NUM.sub("", n)
    n = n.replace(".", "").replace(",", ".", 1) if decimal == "," else n.replace(",", "")
    n = _NON_NUM.sub("", n)
    if n in ("", "-", "+", ".", "-.", "+."):
        return None
    x = js_number(n)
    return x if _finite(x) else None


def _tri_state(v: Any) -> bool | None:
    if _nullish(v) or v == "":
        return None
    if isinstance(v, bool):
        return v
    if _is_num(v):
        return v != 0
    s = js_str(v).strip(_JS_WS).lower()
    if s in ("true", "yes", "y", "1"):
        return True
    if s in ("false", "no", "n", "0"):
        return False
    return None


_FAIL = object()


def _coerce(spec: dict, v: Any) -> Any:
    """The module's per-type reader: the value, or ``_FAIL`` (stored as null)."""
    t = spec["type"]
    if t == "text":
        return v if isinstance(v, str) else js_str(v)
    if t in ("number", "integer"):
        n = _parse_number(v, "," if spec["decimal"] == "," else ".")
        if n is None or (t == "integer" and n != int(n)):
            return _FAIL
        return n
    if t == "boolean":
        if isinstance(v, bool):
            return v
        if spec["true"] or spec["false"]:
            s = js_str(v)
            if spec["true"] and s in spec["true"]:
                return True
            if spec["false"] and s in spec["false"]:
                return False
            return _FAIL
        b = _tri_state(v)
        return _FAIL if b is None else b
    if t == "date":
        fmt = spec["format"] or "iso"
        if fmt in ("epoch-s", "epoch-ms"):
            x = js_number(v)
            if not _finite(x):
                return _FAIL
            if fmt == "epoch-s":
                x = x * 1e3
                if not _finite(x):
                    return _FAIL
            d = _iso_of_epoch_ms(x)
        elif fmt == "iso":
            d = _iso_date(v)
        else:
            d = _date_from_pattern(v, fmt)
        return _FAIL if d is None else d
    if t == "json":
        if not isinstance(v, str):
            return v
        try:
            return json.loads(v, parse_constant=_no_constant)
        except ValueError:
            return _FAIL
    return v


def _no_constant(name: str) -> Any:
    raise ValueError(name)


def _coerce_fields(specs: list[dict], row: dict) -> None:
    for spec in specs:
        raw = row.get(spec["name"], UNDEF)
        if _nullish(raw) or (isinstance(raw, float) and raw != raw):
            out = None
        elif spec["nulls"] and js_str(raw) in spec["nulls"]:
            out = None
        elif spec["type"] is None:
            out = raw
        elif spec["type"] != "text" and isinstance(raw, str) and raw.strip(_JS_WS) == "":
            out = None
        else:
            out = _coerce(spec, raw)
            out = None if out is _FAIL else out
        row[spec["name"]] = out


# --- the declaration: sources, joins, unnests ------------------------------------------------
class _Join:
    def __init__(self, kind: str, spec: dict, src: str, local_key: str | None, fk: str):
        self.kind, self.spec, self.src, self.local_key, self.fk = kind, spec, src, local_key, fk
        self.missing = "passthrough"
        self.index: dict[str, list[str]] | None = None


class _Src:
    def __init__(self, sid: str, ns: bool, fields: list[dict]):
        self.id, self.ns, self.fields = sid, ns, fields
        self.joins: list[_Join] = []
        self.unnests: list[dict] = []
        self.live: dict[str, dict] = {}
        self.out: dict[str, dict] = {}
        self.done = False


def _aggregates(spec: Any) -> list[tuple[str, dict]] | None:
    if not isinstance(spec, dict):
        return None
    out = []
    for name, a in spec.items():
        if not isinstance(a, dict) or not isinstance(a.get("fn"), str):
            continue
        fn = a["fn"]
        if fn not in ("count", "sum", "avg", "min", "max", "distinctCount", "first", "last"):
            continue
        field = a["field"] if isinstance(a.get("field"), str) and a["field"] else None
        if fn in ("sum", "avg", "min", "max", "distinctCount", "first", "last") and not field:
            continue
        order = (a["orderBy"] if isinstance(a.get("orderBy"), str) and a["orderBy"] else field) \
            if fn in ("first", "last") else None
        out.append((name, {"fn": fn, "field": field, "orderBy": order}))
    return out or None


def _parse_join(e: Any, owner_key: str) -> _Join | None:
    """The module's join reader; a join it would ignore is ignored here too."""
    if not isinstance(e, dict) or e.get("from") is None:
        return None
    src = str(e["from"])
    many = e.get("many") is True or e.get("spread") is not None
    lk = e.get("localKey") if e.get("localKey") is not None else e.get("on")
    if lk is None and many:
        lk = ROW_ID
    if not isinstance(lk, str):
        return None
    fk = e.get("foreignKey") if e.get("foreignKey") is not None else e.get("fromKey")
    if fk is None:
        fk = lk
    if not isinstance(fk, str):
        return None
    j: _Join
    if many:
        if e.get("spread") is not None:
            sp = e["spread"] if isinstance(e["spread"], dict) else {}
            name, value = sp.get("name"), sp.get("value")
            if not (isinstance(name, str) and name) or not (isinstance(value, str) and value):
                return None
            j = _Join("spread", e, src, lk, fk)
            inc = sp.get("include")
            j.name, j.value = name, value
            j.prefix = sp["prefix"] if isinstance(sp.get("prefix"), str) else ""
            j.include = {js_str(x) for x in inc} if isinstance(inc, list) else None
            j.types = {}
            if isinstance(sp.get("type"), dict):
                for k, t in sp["type"].items():
                    s = _field_spec(str(k), t, _SPREAD_TYPES, False)
                    if s:
                        j.types[str(k)] = s
            return j
        missing = e.get("missing") if e.get("missing") in ("hold", "passthrough", "null") else "passthrough"
        if e.get("aggregate") is not None:
            aggs = _aggregates(e["aggregate"])
            if not aggs:
                return None
            j = _Join("rollup", e, src, lk, fk)
            j.aggregates, j.missing = aggs, missing
            return j
        if not isinstance(e.get("as"), str) or not e["as"]:
            return None
        j = _Join("collect", e, src, lk, fk)
        j.as_, j.missing = e["as"], missing
        j.distinct, j.sort = e.get("distinct") is True, e.get("sort") is True
        sel = e.get("select") if e.get("select") is not None else e.get("fields")
        j.select = sel
        return j
    missing = e.get("missing") if e.get("missing") in ("hold", "passthrough", "null") else "passthrough"
    j = _Join("lookup", e, src, lk, fk)
    j.missing = missing
    g = e.get("fields") if e.get("fields") is not None else e.get("select")
    j.pairs = []
    if isinstance(g, list):
        j.pairs = [(str(x), str(x)) for x in g]
    elif isinstance(g, dict):
        j.pairs = [(str(k), str(v)) for k, v in g.items()]
    return j


class _Engine:
    def __init__(self, payload: dict):
        self.sources: dict[str, _Src] = {}
        raw = payload.get("sources") or {}
        for sid, entry in raw.items():
            self._register(str(sid), entry.get("options") or {}, True)
        self.raw = raw
        self.route = str(payload.get("route"))

    # registration mirrors addSource: joins are checked against the sources registered so far
    def _joins(self, spec: Any, owner: str) -> list[_Join]:
        out = []
        for e in ([] if spec is None else spec if isinstance(spec, list) else [spec]):
            j = _parse_join(e, owner)
            if j is None:
                continue
            if j.src == owner or self._reaches(j.src, owner):
                continue
            out.append(j)
        return out

    def _reaches(self, start: str, target: str) -> bool:
        seen, stack = set(), [start]
        while stack:
            cur = stack.pop()
            if cur == target:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            s = self.sources.get(cur)
            if s:
                stack.extend(j.src for j in s.joins)
        return False

    def _register(self, sid: str, options: dict, ns: bool, parent: str | None = None) -> _Src:
        src = _Src(sid, ns, _field_specs(options.get("fields")) if ns else [])
        src.joins = self._joins(options.get("join"), sid)
        self.sources[sid] = src
        src.unnests = self._unnests(options.get("unnest"), src)
        return src

    def _unnests(self, spec: Any, owner: _Src) -> list[dict]:
        out = []
        for u in ([] if spec is None else spec if isinstance(spec, list) else [spec]):
            if not isinstance(u, dict):
                continue
            path = u.get("path") if isinstance(u.get("path"), str) and u.get("path") else None
            as_ = u.get("as") if isinstance(u.get("as"), str) and u.get("as") else None
            if not path or not as_ or as_ == owner.id:
                continue
            if as_ in self.sources:
                raise RouterEngineUnsupported(f"unnest 'as' {as_!r} names an existing source")
            child = _Src(as_, False, [])
            self.sources[as_] = child
            child.joins = self._joins(u.get("join"), as_)
            child.unnests = self._unnests(u.get("unnest"), child)
            out.append({"path": path, "child": child, "keep": u.get("keep") is True,
                        "parent_key": u["parentKey"] if isinstance(u.get("parentKey"), str) and u["parentKey"]
                        else f"{owner.id}Id",
                        "key": u["key"] if isinstance(u.get("key"), str) and u["key"] else None})
        return out

    # --- load: the sources' live rows, then every source's shaped rows ----------------------
    def load(self) -> None:
        for sid, entry in self.raw.items():
            src = self.sources[str(sid)]
            col = entry.get("columnar") or {}
            keys = col.get(ROW_ID) or []
            fields = [f for f in col if f != ROW_ID]
            rows: dict[str, dict] = {}
            for i in range(len(keys)):
                row: dict = {ROW_ID: keys[i], SOURCE: src.id}
                for f in fields:
                    row[f] = col[f][i]
                _coerce_fields(src.fields, row)
                if src.ns:
                    row[ROW_ID] = f"{src.id}:{row.get(ROW_ID)}"
                rows[js_str(row[ROW_ID])] = row
            src.live = rows
            self._unnest_live(src)
        for src in self.sources.values():
            self._shape(src)

    def _unnest_live(self, src: _Src) -> None:
        for u in src.unnests:
            child: _Src = u["child"]
            for pk, prow in src.live.items():
                lst = _get_path(prow, u["path"])
                for g, a in enumerate(lst if isinstance(lst, list) else []):
                    k = None
                    if u["key"] and isinstance(a, dict):
                        k = a.get(u["key"])
                    elif u["key"]:
                        k = UNDEF
                    key = f"{pk}:{g}" if _nullish(k) else js_str(k)
                    row = dict(a) if isinstance(a, dict) else {}
                    row[u["parent_key"]] = prow[ROW_ID]
                    row[ROW_ID] = key
                    row[SOURCE] = child.id
                    child.live[key] = row
            self._unnest_live(child)

    def _shape(self, src: _Src) -> None:
        if src.done:
            return
        src.done = True
        for j in src.joins:
            dep = self.sources.get(j.src)
            if dep is not None:
                self._shape(dep)
        for key, live in src.live.items():
            ok, row = self._row(src, live)
            if ok:
                src.out[key] = row

    def _index(self, j: _Join) -> dict[str, list[str]]:
        if j.index is None:
            idx: dict[str, list[str]] = {}
            frm = self.sources.get(j.src)
            for k, row in (frm.live.items() if frm else ()):
                v = row.get(j.fk, UNDEF)
                if not _nullish(v):
                    idx.setdefault(js_str(v), []).append(k)
            j.index = idx
        return j.index

    def _children(self, j: _Join, s: str | None) -> list[dict]:
        frm = self.sources.get(j.src)
        if s is None or frm is None:
            return []
        return [frm.out[k] for k in self._index(j).get(s, ()) if k in frm.out]

    def _row(self, src: _Src, live: dict) -> tuple[bool, dict]:
        r = dict(live)
        for u in src.unnests:
            if not u["keep"]:
                _delete_path(r, u["path"])
        for j in src.joins:
            lv = r.get(j.local_key, UNDEF)
            s = None if _nullish(lv) else js_str(lv)
            if j.kind == "spread":
                for name, value in self._spread(j, s).items():
                    if name in live:
                        continue  # a spread field never overwrites a base parent field
                    r[name] = value
            elif j.kind == "rollup":
                kids = self._children(j, s)
                if j.missing == "hold" and not kids:
                    return False, r
                for name, a in j.aggregates:
                    r[name] = _aggregate(a, kids)
            elif j.kind == "collect":
                kids = self._children(j, s)
                if j.missing == "hold" and not kids:
                    return False, r
                if j.missing == "null" and not kids:
                    r[j.as_] = None
                else:
                    r[j.as_] = self._collect(j, kids)
            else:
                frm = self.sources.get(j.src)
                match = None
                if s is not None and frm is not None:
                    ids = self._index(j).get(s)
                    if ids:
                        match = frm.out.get(ids[-1])
                if match is not None:
                    for f, to in j.pairs:
                        r[to] = match.get(f, UNDEF)
                else:
                    if j.missing == "hold":
                        return False, r
                    if j.missing == "null":
                        for _f, to in j.pairs:
                            r[to] = None
        return True, r

    def _collect(self, j: _Join, kids: list[dict]) -> list:
        sel = j.select
        if isinstance(sel, str):
            vals = [k.get(sel, UNDEF) for k in kids]
        elif isinstance(sel, list):
            vals = [{str(f): k.get(str(f), UNDEF) for f in sel} for k in kids]
        elif isinstance(sel, dict):
            vals = [{str(to): k.get(f, UNDEF) for f, to in sel.items()} for k in kids]
        else:
            vals = [dict(k) for k in kids]
        if j.distinct:
            seen, uniq = set(), []
            for v in vals:
                tag = _json_text(v) if isinstance(v, (dict, list)) else f"{_js_typeof(v)}:{js_str(v)}"
                if tag not in seen:
                    seen.add(tag)
                    uniq.append(v)
            vals = uniq
        if j.sort:
            vals = sorted(vals, key=cmp_to_key(_js_compare))
        return vals

    def _spread(self, j: _Join, s: str | None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for row in self._children(j, s):
            name = row.get(j.name, UNDEF)
            if _nullish(name):
                continue
            nm = js_str(name)
            if j.include is not None and nm not in j.include:
                continue
            out[j.prefix + nm] = _spread_value(row.get(j.value, UNDEF), j.types.get(nm))
        return out

    # --- the route's rows, as the browser reports them ---------------------------------------
    def report(self) -> dict:
        src = self.sources.get(self.route)
        rows = [r for r in (src.out.values() if src else ()) if r.get(SOURCE) == self.route]
        fields: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for f in r:
                if f not in (ROW_ID, SOURCE) and f not in seen:
                    seen.add(f)
                    fields.append(f)
        out: dict[str, list] = {ROW_ID: [js_str(r[ROW_ID]) for r in rows]}
        for f in fields:
            out[f] = [_clean(r.get(f, UNDEF)) for r in rows]
        return out


def _js_typeof(v: Any) -> str:
    if v is UNDEF:
        return "undefined"
    if isinstance(v, bool):
        return "boolean"
    if _is_num(v):
        return "number"
    if isinstance(v, str):
        return "string"
    return "object"


def _get_path(row: dict, path: str) -> Any:
    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _delete_path(row: dict, path: str) -> None:
    parts = path.split(".")
    cur = row
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            return
        nxt = dict(nxt)  # the row is a shallow copy: do not edit the live row's nested object
        cur[part] = nxt
        cur = nxt
    cur.pop(parts[-1], None)


def _spread_value(v: Any, spec: dict | None) -> Any:
    if _nullish(v) or spec is None:
        return v
    t = spec["type"]
    if t in ("boolean", "date"):
        if isinstance(v, str) and v.strip(_JS_WS) == "":
            return None
        out = _coerce(spec, v)
        return None if out is _FAIL else out
    if t == "number":
        if _is_num(v):
            return v
        x = js_number(v)
        return x if _finite(x) else v
    if t == "text":
        return js_str(v)
    return v


def _aggregate(a: dict, rows: list[dict]) -> Any:
    fn, field, order = a["fn"], a["field"], a["orderBy"]
    if fn == "count":
        return len(rows)
    if fn in ("first", "last"):
        if not rows:
            return None
        best = rows[0]
        for r in rows[1:]:
            c = _js_compare(r.get(order, UNDEF), best.get(order, UNDEF))
            if (fn == "first" and c < 0) or (fn == "last" and c > 0):
                best = r
        return best.get(field, UNDEF) if field else dict(best)
    if fn in ("min", "max"):
        nums: list[float] = []
        other: list[Any] = []
        for r in rows:
            g = r.get(field, UNDEF)
            if _nullish(g):
                continue
            if _is_num(g):
                if g == g:
                    nums.append(g)
                continue
            if isinstance(g, str) and g.strip(_JS_WS) != "" and _finite(js_number(g)):
                nums.append(js_number(g))
                continue
            other.append(g)
        if nums:
            return min(nums) if fn == "min" else max(nums)
        best, has = None, False
        for g in other:
            if not has or (_js_compare(g, best) < 0 if fn == "min" else _js_compare(g, best) > 0):
                best, has = g, True
        return best if has else None
    if fn == "distinctCount":
        if not rows:
            return None
        return len({_same_value_zero(r.get(field, UNDEF)) for r in rows})
    vals = [x for x in (js_number(r.get(field, UNDEF)) for r in rows) if _finite(x)]
    if fn == "sum":
        return _js_sum(vals)
    if fn == "avg" and vals:
        return _js_sum(vals) / len(vals)
    return None


def _js_sum(vals: list[float]) -> float:
    """Left-to-right float addition, as ``reduce((s, a) => s + a, 0)`` does it."""
    total = 0.0
    for x in vals:
        total += x
    return total


def _same_value_zero(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return ("object", id(v))
    return (_js_typeof(v), v if not _is_num(v) else float(v))


def shape_route(payload: dict) -> dict:
    """The shaped rows of ``payload["route"]``, columnar -- what the browser's router would report.

    ``payload`` is ``LatticeRouter._payload(route)``. Raises :class:`RouterEngineUnsupported`
    where only the browser can say what the rows are.
    """
    eng = _Engine(payload)
    eng.load()
    return eng.report()
