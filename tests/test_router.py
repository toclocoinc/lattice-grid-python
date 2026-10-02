"""Card 1620: the data router in the notebook.

The router RUNS IN THE BROWSER on the grid's own ``data-router`` module. Every
semantics test therefore renders the real widget front end in Chromium, lets the
router shape the sources there, and reads the shaped rows back through the real
``_shaped`` comm into Python (``widget.view`` / ``widget.df``), asserting the
numbers against pandas. Nothing here re-implements a join in Python.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # noqa: E402

from _fixture import customers_txns_labels  # noqa: E402
from _hosts import launch_browser  # noqa: E402
from _page import page_html, sets  # noqa: E402
from lattice_grid_jupyter import LatticeGridWarning, LatticeGridWidget, LatticeRouter  # noqa: E402

CUST = pd.DataFrame({"cid": ["a", "b", "c"], "name": ["A", "B", "C"]})
TXN = pd.DataFrame({"tid": ["t1", "t2", "t3", "t4"], "cid": ["a", "a", "b", "a"],
                    "amt": [5.0, 3.0, 9.0, 5.0], "kind": ["x", "y", "x", "x"]})


# --- declaring specs as plain data (no browser) ----------------------------------------
def test_unknown_spec_keys_are_warned_by_name_at_every_level():
    with pytest.warns(LatticeGridWarning) as rec:
        LatticeRouter(sources={
            "customer": {"data": CUST, "key": "cid", "frobnicate": 1,
                         "join": {"from": "txn", "many": True, "foreing_key": "cid", "as": "t"}},
            "txn": {"data": TXN, "key": "tid"},
        })
    text = " ".join(str(r.message) for r in rec)
    assert "router.unknown:sources.customer.frobnicate" in text and "frobnicate" in text
    assert "router.unknown:sources.customer.join.foreing_key" in text and "foreing_key" in text
    assert "[lattice]" in text


def test_snake_case_is_mapped_and_functions_are_refused():
    r = LatticeRouter(sources={
        "customer": {"data": CUST, "key": "cid", "join": {
            "from": "txn", "many": True, "foreign_key": "cid", "on_parent_delete": "cascade",
            "aggregate": {"last": {"fn": "last", "field": "amt", "order_by": "tid"}}}},
        "txn": {"data": TXN, "key": "tid"},
    })
    j = r._sources["customer"].options["join"]
    assert j["foreignKey"] == "cid" and j["onParentDelete"] == "cascade"
    assert j["aggregate"]["last"]["orderBy"] == "tid"
    assert j["localKey"] == "cid"          # a rollup matches on THIS source's key column
    with pytest.raises(TypeError, match="function"):
        LatticeRouter(sources={"customer": {"data": CUST, "key": "cid",
                                            "join": {"from": "txn", "many": True, "as": "t", "where": lambda r: True}},
                               "txn": {"data": TXN, "key": "tid"}})
    with pytest.raises(TypeError, match="function"):
        LatticeRouter(sources={"customer": {"data": CUST, "key": "cid", "map": lambda r: r}})


def test_sources_keys_joins_and_routes_are_validated_by_name():
    with pytest.raises(ValueError, match="name its key column"):
        LatticeRouter(sources={"customer": CUST})
    with pytest.raises(ValueError, match="not a column"):
        LatticeRouter(sources={"customer": {"data": CUST, "key": "nope"}})
    with pytest.raises(ValueError, match="not unique"):
        LatticeRouter(sources={"txn": {"data": pd.concat([TXN, TXN]), "key": "tid"}})
    with pytest.raises(ValueError, match="join from 'ghost'"):
        LatticeRouter(sources={"customer": {"data": CUST, "key": "cid", "join": {"from": "ghost", "local_key": "cid"}}})
    with pytest.raises(ValueError, match="route 'x' reads 'ghost'"):
        LatticeRouter(sources={"customer": {"data": CUST, "key": "cid"}}, routes={"x": "ghost"})
    r = LatticeRouter(sources={"customer": {"data": CUST, "key": "cid"}})
    with pytest.raises(ValueError, match="needs one of"):
        LatticeGridWidget(r, route="nope")


def test_a_router_backed_grid_refuses_the_dataframe_mutators():
    r = LatticeRouter(sources={"customer": {"data": CUST, "key": "cid"}})
    w = LatticeGridWidget(r)           # one route: no route= needed
    assert w.route == "customer" and w._mode == "router"
    for call in (lambda: w.set_data(CUST), lambda: w.append_rows([{}]), lambda: w.delete_rows(["0"])):
        with pytest.raises(TypeError, match="router.update"):
            call()


def test_update_sends_a_keyed_diff_and_keeps_late_views_current():
    c, t, l = customers_txns_labels()
    r = LatticeRouter(sources={"customer": {"data": c, "key": "customer_id"},
                               "txn": {"data": t, "key": "txn_id"}})
    w = LatticeGridWidget(r, route="txn", offline=True)
    sent = []
    w.send = lambda msg, buffers=None: sent.append(msg)
    t2 = t.copy()
    t2.loc[t2.txn_id == t.txn_id.iloc[0], "amount"] = 999.0          # one changed
    t2 = t2[t2.txn_id != t.txn_id.iloc[1]]                            # one removed
    t2 = pd.concat([t2, pd.DataFrame([{"txn_id": "T-new", "customer_id": "C001", "amount": 1.0, "ts": "2024-01-01"}])])
    diff = r.update("txn", t2)
    assert diff == {"added": 1, "updated": 1, "removed": 1}
    assert len(sent) == 1 and sent[0]["type"] == "lg:router" and sent[0]["source"] == "txn"
    assert sorted(sent[0]["upsert"]["__row_id__"]) == sorted([t.txn_id.iloc[0], "T-new"])
    assert sent[0]["remove"] == [t.txn_id.iloc[1]]
    # a view that renders later reads the synced state: it must carry the NEW rows
    cols = w.get_state()["_router"]["sources"]["txn"]["columnar"]
    assert "T-new" in cols["txn_id"] and t.txn_id.iloc[1] not in cols["txn_id"]
    assert r.update("txn", t2) == {"added": 0, "updated": 0, "removed": 0} and len(sent) == 1  # no change, no message
    with pytest.raises(ValueError, match="same columns"):
        r.update("txn", t2.drop(columns=["ts"]))
    with pytest.raises(KeyError, match="unknown source 'nope'"):
        r.update("nope", t2)


# --- the real browser ----------------------------------------------------------------
def render(p, w, size=(1100, 600)):
    browser = launch_browser(p)
    if browser is None:
        pytest.skip("no real browser available to launch")
    page = browser.new_page(viewport={"width": size[0], "height": size[1]})
    page.set_content(page_html(w))
    page.wait_for_function("window.__rendered === true || window.__error", timeout=30000)
    assert page.evaluate("window.__error") is None
    page.wait_for_function("window.__sets.some(s => s[0] === '_shaped')", timeout=10000)
    page.wait_for_selector(".lat-cell[role=gridcell]", timeout=10000)
    return browser, page


def shaped(sources, route):
    """Declare, render in a real browser, and bring the shaped rows back to Python."""
    r = LatticeRouter(sources=sources, routes={route: route})
    w = LatticeGridWidget(r, route=route, offline=True)
    with sync_playwright() as p:
        browser, page = render(p, w)
        try:
            w._shaped = sets(page, "_shaped")[-1]
            w._router_stats = sets(page, "_router_stats")[-1]
        finally:
            browser.close()
    return w


def test_real_browser_lookup_join_with_rename_and_missing_null():
    w = shaped({
        "txn": {"data": TXN, "key": "tid", "join": {"from": "customer", "local_key": "cid", "foreign_key": "cid",
                                                    "fields": {"name": "owner"}, "missing": "null"}},
        "customer": {"data": CUST, "key": "cid"}}, "txn")
    assert list(w.df.owner) == ["A", "A", "B", "A"] and list(w.df.tid) == ["t1", "t2", "t3", "t4"]


def test_real_browser_collect_join_gathers_children_with_select_distinct_and_sort():
    w = shaped({
        "customer": {"data": CUST, "key": "cid", "join": {"from": "txn", "many": True, "foreign_key": "cid",
                                                           "as": "amounts", "select": "amt", "distinct": True, "sort": True}},
        "txn": {"data": TXN, "key": "tid"}}, "customer")
    assert [list(x) for x in w.df.amounts] == [[3, 5], [9], []]


def test_real_browser_rollup_onto_parent_matches_pandas_groupby():
    c, t, l = customers_txns_labels()
    w = shaped({
        "customer": {"data": c, "key": "customer_id", "join": {
            "from": "txn", "many": True, "foreign_key": "customer_id",
            "aggregate": {"n": {"fn": "count"}, "spend": {"fn": "sum", "field": "amount"},
                          "mean": {"fn": "avg", "field": "amount"}, "top": {"fn": "max", "field": "amount"},
                          "low": {"fn": "min", "field": "amount"}, "days": {"fn": "distinctCount", "field": "ts"}}}},
        "txn": {"data": t, "key": "txn_id"}}, "customer")
    got = w.df.set_index("customer_id")
    g = t.groupby("customer_id").amount
    want = pd.DataFrame({"n": g.count(), "spend": g.sum(), "mean": g.mean(), "top": g.max(), "low": g.min(),
                         "days": t.groupby("customer_id").ts.nunique()}).reindex(c.customer_id)
    assert len(got) == len(c)
    assert got.n.to_numpy() == pytest.approx(want.n.fillna(0).to_numpy())
    assert got.spend.to_numpy() == pytest.approx(want.spend.fillna(0).to_numpy())
    has = want.n.notna().to_numpy()
    for col in ("mean", "top", "low"):
        assert got[col].to_numpy()[has] == pytest.approx(want[col].to_numpy()[has])
    assert got.days.to_numpy()[has] == pytest.approx(want.days.to_numpy()[has])


def test_real_browser_spread_turns_attribute_rows_into_columns():
    attr = pd.DataFrame({"aid": ["1", "2", "3"], "cid": ["a", "a", "b"], "k": ["tier", "age", "tier"],
                         "v": ["gold", "41", "silver"]})
    w = shaped({
        "customer": {"data": CUST, "key": "cid", "join": {"from": "attr", "foreign_key": "cid",
                                                           "spread": {"name": "k", "value": "v", "prefix": "cf_",
                                                                      "type": {"age": "number"}}}},
        "attr": {"data": attr, "key": "aid"}}, "customer")
    assert list(w.df.cf_tier[:2]) == ["gold", "silver"] and pd.isna(w.df.cf_tier.iloc[2])
    assert w.df.cf_age.iloc[0] == 41 and pd.isna(w.df.cf_age.iloc[1])


def test_real_browser_unnest_expands_nested_arrays_into_their_own_route():
    acc = pd.DataFrame({"cid": ["a", "b"], "name": ["A", "B"], "addresses": [
        [{"city": "Leeds", "zip": "LS1"}, {"city": "York", "zip": "YO1"}], [{"city": "Bath", "zip": "BA1"}]]})
    w = shaped({"acc": {"data": acc, "key": "cid",
                        "unnest": {"path": "addresses", "as": "address", "parent_key": "acc_id"}}}, "address")
    assert list(w.df.city) == ["Leeds", "York", "Bath"] and list(w.df.acc_id) == ["acc:a", "acc:a", "acc:b"]


def test_real_browser_fields_coerce_raw_values_to_typed_ones():
    vals = pd.DataFrame({"vid": ["1", "2"], "num": ["1.234,5", "2.000,0"], "flag": ["yes", "no"],
                         "when": ["1700000000", "1700086400"], "blank": ["n/a", "x"], "d": ["03/02/2024", "04/02/2024"]})
    w = shaped({"vals": {"data": vals, "key": "vid", "fields": {
        "num": {"type": "number", "decimal": ","}, "flag": {"type": "boolean", "true": ["yes"], "false": ["no"]},
        "when": {"type": "date", "format": "epoch-s"}, "blank": {"nulls": ["n/a"]},
        "d": {"type": "date", "format": "dd/MM/yyyy"}}}}, "vals")
    assert list(w.df.num) == [1234.5, 2000.0] and list(w.df.flag) == [True, False]
    assert list(w.df.when.dt.strftime("%Y-%m-%d")) == ["2023-11-14", "2023-11-15"]
    assert w.df.blank.isna().tolist() == [True, False]
    assert list(w.df.d.dt.strftime("%Y-%m-%d")) == ["2024-02-03", "2024-02-04"]


# --- a source update re-emits ONLY the affected parents --------------------------------------
def recipe_router():
    c, t, l = customers_txns_labels()
    r = LatticeRouter(sources={
        "customer": {"data": c, "key": "customer_id", "fields": {"signup": {"type": "date", "format": "dd/MM/yyyy"}},
                     "join": [{"from": "txn", "many": True, "foreign_key": "customer_id",
                               "aggregate": {"n_txn": {"fn": "count"}, "spend": {"fn": "sum", "field": "amount"}}},
                              {"from": "label", "local_key": "customer_id", "foreign_key": "customer_id",
                               "fields": ["churned"], "missing": "null"}]},
        "txn": {"data": t, "key": "txn_id"},
        "label": {"data": l, "key": "customer_id"},
    }, routes={"customer": "customer"})
    return c, t, l, r


def test_real_browser_a_source_update_re_emits_only_the_affected_parents_and_view_is_the_shaped_frame():
    c, t, l, r = recipe_router()
    w = LatticeGridWidget(r, route="customer", offline=True)
    with sync_playwright() as p:
        browser, page = render(p, w)
        try:
            page.wait_for_function("window.__sets.filter(s => s[0] === '_router_stats').length >= 1")
            base = sets(page, "_router_stats")[-1]
            assert base["added"] == len(c) and base["updated"] == 0     # the initial shaping, counted
            # play the kernel: whatever Python sends the widget reaches the front end
            w.send = lambda msg, buffers=None: page.evaluate("(m) => window.__pymsg(m)", msg)
            # change one txn of C005, add one for C007, delete one of C009
            t2 = t.copy()
            i5 = t2.index[t2.customer_id == "C005"][0]
            t2.loc[i5, "amount"] += 100.0
            i9 = t2.index[t2.customer_id == "C009"][0]
            t2 = t2.drop(index=i9)
            t2 = pd.concat([t2, pd.DataFrame([{"txn_id": "T-new", "customer_id": "C007", "amount": 50.0, "ts": "2024-05-05"}])])
            before = len(sets(page, "_shaped"))
            diff = r.update("txn", t2)
            assert diff == {"added": 1, "updated": 1, "removed": 1}
            page.wait_for_function(f"window.__sets.filter(s => s[0] === '_shaped').length > {before}", timeout=8000)
            stats = sets(page, "_router_stats")[-1]
            re_emitted = {k: stats[k] - base[k] for k in ("added", "updated", "removed")}
            # 40 customers, 3 touched: ONLY those three parents re-emit (as updates)
            assert re_emitted == {"added": 0, "updated": 3, "removed": 0}, re_emitted
            assert stats["batches"] - base["batches"] <= 3
            w._shaped = sets(page, "_shaped")[-1]
        finally:
            browser.close()
    want = t2.groupby("customer_id").amount.agg(["count", "sum"]).reindex(c.customer_id)
    got = w.view.set_index("customer_id")
    assert got.n_txn.to_numpy() == pytest.approx(want["count"].fillna(0).to_numpy())
    assert got.spend.to_numpy() == pytest.approx(want["sum"].fillna(0).to_numpy())


def test_real_browser_end_to_end_recipe_feature_table_into_a_model():
    c, t, l, r = recipe_router()
    w = LatticeGridWidget(r, route="customer", offline=True)
    with sync_playwright() as p:
        browser, page = render(p, w, size=(1300, 700))
        try:
            page.wait_for_function("window.__sets.some(s => s[0] === '_view_keys')", timeout=8000)
            # filter in the grid, as a person would: the view is the filtered shaped frame
            page.evaluate("document.querySelector('div[style*=height]').__latticeGrid.filters.quick('pro')")
            page.wait_for_function(
                "window.__sets.filter(s => s[0]==='_view_keys').slice(-1)[0][1].length === %d" % int((c.plan == "pro").sum()),
                timeout=8000)
            w._shaped = sets(page, "_shaped")[-1]
            w._view_keys = sets(page, "_view_keys")[-1]
            headers = page.evaluate("[...document.querySelectorAll('.lat-header-cell, [role=columnheader]')].map(e => e.textContent.trim())")
        finally:
            browser.close()
    feats = w.view
    assert isinstance(feats, pd.DataFrame) and len(feats) == int((c.plan == "pro").sum())
    assert set(feats.plan) == {"pro"} and feats.churned.dtype == bool
    assert {"customer_id", "n_txn", "spend", "churned", "signup"} <= set(feats.columns)
    assert str(feats.signup.dtype).startswith("datetime64")
    assert {"n_txn", "spend", "churned"} <= set(headers)          # the grid itself showed the shaped columns
    # it goes straight into a model: a least-squares fit on the feature table
    X = np.c_[np.ones(len(feats)), feats.n_txn.to_numpy(float), feats.spend.to_numpy(float)]
    beta, *_ = np.linalg.lstsq(X, feats.churned.to_numpy(float), rcond=None)
    assert beta.shape == (3,) and np.isfinite(beta).all()


# --- the docs run as written ------------------------------------------------------------------
def test_every_python_block_in_docs_router_runs_as_written():
    import pathlib
    import re
    text = (pathlib.Path(__file__).parent.parent / "docs" / "router.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    ns: dict = {}
    ran = 0
    for block in blocks:
        if "sources={\"customer\": {...}" in block or ("new_txns" in block and "new_txns =" not in block):
            continue  # a shape sketch, not a runnable block
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            if "features = w.view" in block:
                # the browser has not shaped anything yet in a bare Python run: give the widget the
                # shaped rows the way the browser reports them (the real round trip is the tests above)
                ns["w"]._shaped = {"__row_id__": ["customer:C000"], "n_txn": [2], "spend": [3.0],
                                   "biggest": [2.0], "churned": [True]}
            if "router.update" in block:
                ns["w"].send = lambda msg, buffers=None: None
            exec(compile(block, "docs/router.md", "exec"), ns)  # noqa: S102
        ran += 1
    assert ran == 4 and isinstance(ns["router"], LatticeRouter) and ns["X"].shape == (1, 3)
    assert ns["router"].update("txn", ns["txns"]) == {"added": 0, "updated": 1, "removed": 0}
