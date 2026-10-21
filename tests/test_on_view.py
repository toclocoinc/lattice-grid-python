"""BACKLOG-0005030: ``w.view`` never waits for the browser; ``w.on_view(callback)``
delivers a browser-only view when the browser's report arrives. No blocking, no
awaiting, no kernel pump -- the widget's own comm handling (a trait change) calls
the callback directly, so this works on every ipykernel version."""

import time

import pandas as pd
import pytest

from lattice_grid_jupyter import LatticeGridWidget, LatticeRouter, LatticeViewNeedsBrowser
from lattice_grid_jupyter import _serialize as S

CUST = pd.DataFrame({"cid": ["a", "b", "c"], "name": ["A", "B", "C"]})


def _router_widget():
    return LatticeGridWidget(LatticeRouter(sources={"customer": {"data": CUST, "key": "cid"}}))


def test_sync_view_raises_a_named_error_at_once_for_a_browser_only_view():
    w = _router_widget()
    t0 = time.monotonic()
    with pytest.raises(LatticeViewNeedsBrowser, match="on_view"):
        w.view
    assert time.monotonic() - t0 < 0.5


def test_sync_view_returns_for_a_kernel_computable_view():
    df = pd.DataFrame({"name": ["Ada", "Grace", "Linus"], "score": [91, 88, 77]})
    w = LatticeGridWidget(df, state={"version": 2, "filters": {"col": "score", "op": "gte", "value": 80}})
    t0 = time.monotonic()
    rows = w.view
    assert time.monotonic() - t0 < 0.5
    assert sorted(rows["name"]) == ["Ada", "Grace"]


def test_on_view_fires_immediately_when_a_view_is_already_known():
    df = pd.DataFrame({"name": ["Ada", "Grace"], "score": [91, 88]})
    w = LatticeGridWidget(df)
    seen = []
    handle = w.on_view(lambda view: seen.append(view))
    assert len(seen) == 1
    assert sorted(seen[0]["name"]) == ["Ada", "Grace"]
    handle.remove()


def test_on_view_does_not_fire_until_the_browsers_report_arrives_for_a_router():
    w = _router_widget()
    seen = []
    handle = w.on_view(lambda view: seen.append(view))
    assert seen == []  # not yet known: no browser report, and no error either

    # the browser's report, delivered as a separate message (a trait change),
    # not from inside a running cell -- exactly what the real comm handler does
    w._shaped = {S.ROW_KEY: ["a", "b"], "cid": ["a", "b"]}

    assert len(seen) == 1
    assert list(seen[0]["cid"]) == ["a", "b"]
    handle.remove()


def test_on_view_handle_remove_stops_further_callbacks():
    w = _router_widget()
    seen = []
    handle = w.on_view(lambda view: seen.append(view))
    handle.remove()
    w._shaped = {S.ROW_KEY: ["a"], "cid": ["a"]}
    assert seen == []
