"""Grid / column options passed from Python: plain dicts, snake_case accepted.

Options are DATA. Functions (``compute``, ``cell``, ``rowClass``, hooks, ...) cannot
cross the comm and are refused with a clear message instead of being dropped
silently.

snake_case -> camelCase: a key is rewritten only when its camelCase form is a
property name the grid's own declarations contain (``row_height`` -> ``rowHeight``).
Anything else is left exactly as written, so ids you chose (a column called
``sales_total``, keys of a ``context`` dict) are never mangled; a typo then reaches
the grid unchanged and the grid reports it (``config.unknown:*``).
"""

from __future__ import annotations

import re
import warnings
from typing import Any

from ._option_names import CAMEL_NAMES

# Python owns these: they are built from the DataFrame.
RESERVED = ("columns", "rows", "rowKey")
# Values under these keys are the caller's own data: do not rewrite inside them.
OPAQUE = frozenset({"context", "lookup", "dataTypes", "typeOptions", "state", "views"})

_SNAKE = re.compile(r"_([a-z0-9])")


class LatticeGridWarning(UserWarning):
    """A warning raised by the grid itself (``[lattice] <id>``) or by option handling."""


def to_camel(key: str) -> str:
    """``row_height`` -> ``rowHeight`` when that is a grid property name, else ``key``."""
    if not isinstance(key, str) or key.startswith("_") or "_" not in key:
        return key
    camel = _SNAKE.sub(lambda m: m.group(1).upper(), key)
    return camel if camel in CAMEL_NAMES else key


def map_keys(value: Any, _path: str = "") -> Any:
    """Recursively map snake_case option keys; reject anything that is not data."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            ck = to_camel(k)
            out[ck] = v if ck in OPAQUE else map_keys(v, f"{_path}.{ck}" if _path else ck)
        return out
    if isinstance(value, (list, tuple)):
        return [map_keys(v, f"{_path}[]") for v in value]
    if callable(value):
        raise TypeError(
            f"option {_path or '<root>'!r} is a function ({value!r}). Options are data: "
            "functions cannot be sent to the browser. Use the declarative form "
            "(a column type/format/formula string, a named preset) instead."
        )
    return value


def grid_options(user: dict | None, flags: dict | None = None) -> dict:
    """The effective top-level options: convenience ``flags`` under the user's dict."""
    mapped = map_keys(dict(user or {}))
    for k in RESERVED:
        if k in mapped:
            warnings.warn(
                f"[lattice] option {k!r} is built from the DataFrame and was ignored"
                + (" (pass per-column options as columns=[...])" if k == "columns" else ""),
                LatticeGridWarning, stacklevel=4,
            )
            del mapped[k]
    return {**(flags or {}), **mapped}


def merge_columns(built: list[dict], overrides: list[dict] | None) -> list[dict]:
    """Layer per-column option dicts (matched on ``field``/``id``) over the built columns."""
    if not overrides:
        return built
    by_field = {c["field"]: c for c in built}
    merged = [dict(c) for c in built]
    index = {c["field"]: i for i, c in enumerate(merged)}
    for o in overrides:
        if not isinstance(o, dict):
            raise TypeError("columns=[...] takes one dict per column")
        o = map_keys(dict(o), "columns[]")
        field = o.get("field", o.get("id"))
        if field is None or str(field) not in by_field:
            warnings.warn(
                f"[lattice] columns=[...]: no DataFrame column named {field!r}; "
                "the entry was ignored", LatticeGridWarning, stacklevel=4)
            continue
        i = index[str(field)]
        merged[i] = {**merged[i], **{k: v for k, v in o.items() if k != "field"}}
    return merged


def pivot_options(pivot: Any) -> dict:
    """Translate the ``pivot=`` convenience flag into grid *options*.

    ``pivot={...}`` is passed through as the grid's ``pivot`` option;
    ``pivot=True`` switches the pivot config on. A field name or list of names
    (``pivot="region"``) is not an option at all: it is the grid's pivot *state*,
    which the widget seeds through ``state`` (see ``_flag_state_for``).
    """
    if isinstance(pivot, dict):
        return {"pivot": map_keys(dict(pivot), "pivot")}
    if pivot is True:
        return {"pivot": {"enabled": True}}
    return {}
