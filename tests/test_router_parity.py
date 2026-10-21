"""BACKLOG-0005044: the kernel's shaped rows are the browser engine's, row for row.

Each case declares a router once and runs it through two engines: the vendored
``data-router`` bundle under node (driven exactly as the widget's front end drives it,
``tests/_router_node.js``) and ``lattice_grid_jupyter._router_engine`` in this process. The
two shaped frames must be identical -- same columns in the same order, same rows in the same
order, same values and dtypes -- for collect, every rollup aggregate, lookup, spread, field
coercion and unnest, on the retail feature-table data and on awkward values (nulls, numeric
text, mixed types, duplicates, unmatched keys).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from _fixture import customers_txns_labels
from _router_parity import NODE, both
from lattice_grid_jupyter import LatticeGridWarning, LatticeRouter
from lattice_grid_jupyter._router import shaped_frame
from lattice_grid_jupyter._router_engine import RouterEngineUnsupported, shape_route

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def router(sources, routes=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LatticeGridWarning)
        return LatticeRouter(sources=sources, routes=routes)


def assert_parity(r, route):
    browser, kernel = both(r, route)
    assert list(kernel) == list(browser), "columns / order"
    assert kernel["__row_id__"] == browser["__row_id__"], "row ids / order"
    for col in browser:
        assert kernel[col] == browser[col], f"column {col!r}"
    dates = r._date_fields()
    a, b = shaped_frame(browser, dates), shaped_frame(kernel, dates)
    pd.testing.assert_frame_equal(a, b)
    return kernel


# --- the retail feature table -----------------------------------------------------------
def retail(aggregate, **extra):
    c, t, l = customers_txns_labels()
    return router({
        "customer": {"data": c, "key": "customer_id", "fields": {"signup": {"type": "date", "format": "dd/MM/yyyy"}},
                     "join": [{"from": "txn", "many": True, "foreign_key": "customer_id", "aggregate": aggregate, **extra},
                              {"from": "label", "local_key": "customer_id", "foreign_key": "customer_id",
                               "fields": ["churned"], "missing": "null"}]},
        "txn": {"data": t, "key": "txn_id"},
        "label": {"data": l, "key": "customer_id"},
    }, routes={"customer": "customer", "txn": "txn"})


@pytest.mark.parametrize("fn", ["count", "sum", "avg", "min", "max", "distinctCount", "first", "last"])
def test_each_rollup_aggregate_matches_the_browser_engine_on_the_retail_data(fn):
    spec = {"fn": fn} if fn == "count" else {"fn": fn, "field": "amount"}
    out = assert_parity(retail({"x": spec}), "customer")
    assert len(out["x"]) == 40


def test_all_aggregates_over_text_and_date_fields_too():
    agg = {"n": {"fn": "count"}, "spend": {"fn": "sum", "field": "amount"}, "mean": {"fn": "avg", "field": "amount"},
           "lo": {"fn": "min", "field": "ts"}, "hi": {"fn": "max", "field": "ts"}, "days": {"fn": "distinctCount", "field": "ts"},
           "first_amt": {"fn": "first", "field": "amount", "order_by": "ts"},
           "last_ts": {"fn": "last", "field": "ts"}, "last_amt": {"fn": "last", "field": "amount", "order_by": "ts"}}
    assert_parity(retail(agg), "customer")


def test_the_retail_routes_other_than_the_parent_match_too():
    assert_parity(retail({"n": {"fn": "count"}}), "txn")


@pytest.mark.parametrize("missing", ["hold", "null", "passthrough"])
def test_rollup_missing_policies(missing):
    out = assert_parity(retail({"n": {"fn": "count"}, "s": {"fn": "sum", "field": "amount"}}, missing=missing), "customer")
    if missing == "hold":
        assert len(out["n"]) < 40            # customers with no transaction are withheld


MESSY = pd.DataFrame({
    "k": ["a", "b", "c", "d", "e", "f"],
    "grp": ["g1", "g1", "g1", "g2", "g2", None],
    "num": [1, None, "7", "x", 2.5, 3],
    "txt": ["pear", "apple", None, "fig", "", "apple"],
    "flag": [True, False, None, True, True, False],
    "when": ["2024-01-02", "2024-03-04", None, "2023-12-31", "2024-03-04", "2024-01-01"],
    "blank": [" ", "", "5", " 6 ", "1e3", "0x10"],
})
PARENTS = pd.DataFrame({"gid": ["g1", "g2", "g3", None, "5"], "label": list("vwxyz")})


@pytest.mark.parametrize("fn,field", [(f, c) for f in ("sum", "avg", "min", "max", "distinctCount", "first", "last")
                                      for c in ("num", "txt", "flag", "when", "blank", "nope")])
def test_aggregates_over_messy_values(fn, field):
    r = router({"p": {"data": PARENTS, "key": "gid", "join": {
        "from": "m", "many": True, "foreign_key": "grp", "aggregate": {"v": {"fn": fn, "field": field}}}},
        "m": {"data": MESSY, "key": "k"}}, routes={"p": "p"})
    assert_parity(r, "p")


def test_first_and_last_with_order_by_and_with_no_field_return_the_row():
    r = router({"p": {"data": PARENTS, "key": "gid", "join": {
        "from": "m", "many": True, "foreign_key": "grp", "aggregate": {
            "a": {"fn": "first", "field": "txt", "order_by": "when"}, "b": {"fn": "last", "field": "num", "order_by": "txt"}}}},
        "m": {"data": MESSY, "key": "k"}}, routes={"p": "p"})
    assert_parity(r, "p")


def test_bad_aggregates_are_dropped_like_the_module_drops_them():
    r = router({"p": {"data": PARENTS, "key": "gid", "join": {
        "from": "m", "many": True, "foreign_key": "grp", "aggregate": {
            "ok": {"fn": "count"}, "nofield": {"fn": "sum"}, "median": {"fn": "median", "field": "num"}}}},
        "m": {"data": MESSY, "key": "k"}}, routes={"p": "p"})
    assert "median" not in assert_parity(r, "p")
    r = router({"p": {"data": PARENTS, "key": "gid", "join": {"from": "m", "many": True, "foreign_key": "grp",
                                                               "aggregate": {"nofield": {"fn": "sum"}}}},
                "m": {"data": MESSY, "key": "k"}}, routes={"p": "p"})
    assert_parity(r, "p")          # no usable reducer: the join is ignored


# --- collect ----------------------------------------------------------------------------------
@pytest.mark.parametrize("join", [
    {"as": "all"},
    {"as": "nums", "select": "num"},
    {"as": "nums", "select": "num", "distinct": True},
    {"as": "nums", "select": "txt", "distinct": True, "sort": True},
    {"as": "nums", "select": "num", "sort": True},
    {"as": "rows", "select": ["k", "txt", "nope"]},
    {"as": "rows", "select": {"k": "key", "num": "n"}, "distinct": True},
    {"as": "rows", "fields": ["k", "flag"]},
    {"as": "rows", "fields": {"k": "id"}},
    {"as": "whens", "select": "when", "sort": True, "distinct": True, "missing": "hold"},
    {"as": "whens", "select": "flag", "sort": True, "missing": "null"},
    {"as": "whens", "select": "flag", "distinct": True, "missing": "passthrough"},
    {"as": "none", "select": "nope"},
])
def test_collect_joins(join):
    r = router({"p": {"data": PARENTS, "key": "gid", "join": {"from": "m", "many": True, "foreign_key": "grp", **join}},
                "m": {"data": MESSY, "key": "k"}}, routes={"p": "p"})
    assert_parity(r, "p")


def test_collect_on_the_retail_data():
    c, t, l = customers_txns_labels()
    r = router({"customer": {"data": c, "key": "customer_id", "join": {
        "from": "txn", "many": True, "foreign_key": "customer_id", "as": "amounts", "select": "amount", "distinct": True, "sort": True}},
        "txn": {"data": t, "key": "txn_id"}}, routes={"customer": "customer"})
    assert_parity(r, "customer")


# --- lookup ---------------------------------------------------------------------------------
@pytest.mark.parametrize("join", [
    {"fields": ["label"]}, {"fields": {"label": "lab", "gid": "g"}}, {"select": ["label"]},
    {"fields": ["label"], "missing": "null"}, {"fields": ["label"], "missing": "hold"},
    {"fields": ["label"], "missing": "passthrough"}, {"fields": ["label"], "missing": "bogus"}, {},
])
def test_lookup_joins_including_duplicate_foreign_keys(join):
    r = router({"m": {"data": MESSY, "key": "k", "join": {"from": "p", "local_key": "grp", "foreign_key": "gid", **join}},
                "p": {"data": PARENTS, "key": "gid"}}, routes={"m": "m"})
    assert_parity(r, "m")


def test_lookup_matches_on_string_of_the_value_so_one_and_text_one_meet():
    left = pd.DataFrame({"id": ["a", "b", "c", "d"], "ref": [1, 2.0, "3", 1.5]})
    right = pd.DataFrame({"rid": ["1", "2", "3", "1.5", "7"], "name": list("vwxyz")})
    r = router({"l": {"data": left, "key": "id", "join": {"from": "r", "local_key": "ref", "foreign_key": "rid", "fields": ["name"]}},
                "r": {"data": right, "key": "rid"}}, routes={"l": "l"})
    assert_parity(r, "l")


def test_a_later_join_can_key_on_a_field_an_earlier_join_added():
    a = pd.DataFrame({"aid": ["a1", "a2", "a3"], "bref": ["b1", "b2", "zz"]})
    b = pd.DataFrame({"bid": ["b1", "b2"], "cref": ["c1", "c9"]})
    c = pd.DataFrame({"cid": ["c1"], "val": [10]})
    r = router({"a": {"data": a, "key": "aid", "join": [
        {"from": "b", "local_key": "bref", "foreign_key": "bid", "fields": ["cref"], "missing": "null"},
        {"from": "c", "local_key": "cref", "foreign_key": "cid", "fields": ["val"], "missing": "hold"}]},
        "b": {"data": b, "key": "bid"}, "c": {"data": c, "key": "cid"}}, routes={"a": "a"})
    assert_parity(r, "a")


def test_a_chain_of_sources_rolls_up_through_a_derived_child():
    cust = pd.DataFrame({"cid": ["c1", "c2", "c3"]})
    orders = pd.DataFrame({"oid": ["o1", "o2", "o3", "o4"], "cid": ["c1", "c1", "c2", "c9"]})
    lines = pd.DataFrame({"lid": ["l1", "l2", "l3", "l4", "l5"], "oid": ["o1", "o1", "o2", "o3", "o4"],
                          "qty": [1, 2, 3, 4, 5], "sku": ["x", "y", "x", "z", "x"]})
    r = router({
        "cust": {"data": cust, "key": "cid", "join": {"from": "orders", "many": True, "foreign_key": "cid", "aggregate": {
            "items": {"fn": "sum", "field": "n_items"}, "n": {"fn": "count"}}}},
        "orders": {"data": orders, "key": "oid", "join": {"from": "lines", "many": True, "foreign_key": "oid", "aggregate": {
            "n_items": {"fn": "sum", "field": "qty"}}}},
        "lines": {"data": lines, "key": "lid"}}, routes={"cust": "cust", "orders": "orders"})
    assert_parity(r, "cust")
    assert_parity(r, "orders")


def test_a_join_cycle_is_refused_and_a_join_with_no_key_is_ignored():
    a = pd.DataFrame({"aid": ["1", "2"], "bid": ["1", "2"]})
    b = pd.DataFrame({"bid": ["1", "2"], "aid": ["2", "1"], "v": [5, 6]})
    r = router({"a": {"data": a, "key": "aid", "join": {"from": "b", "local_key": "bid", "foreign_key": "bid", "fields": ["v"]}},
                "b": {"data": b, "key": "bid", "join": {"from": "a", "local_key": "aid", "foreign_key": "aid", "fields": ["bid"]}}},
               routes={"a": "a", "b": "b"})
    assert_parity(r, "a")
    assert_parity(r, "b")
    r = router({"a": {"data": a, "key": "aid", "join": {"from": "b", "fields": ["v"]}}, "b": {"data": b, "key": "bid"}},
               routes={"a": "a"})
    assert_parity(r, "a")


# --- spread -------------------------------------------------------------------------------------
ATTR = pd.DataFrame({"aid": ["1", "2", "3", "4", "5", "6", "7"],
                     "cid": ["a", "a", "b", "a", "c", "b", "zz"],
                     "k": ["tier", "age", "tier", "vip", None, "name", "age"],
                     "v": ["gold", "41", "silver", "yes", "ghost", None, "9"]})
CUST = pd.DataFrame({"cid": ["a", "b", "c", "d"], "name": ["A", "B", "C", "D"]})


@pytest.mark.parametrize("spread", [
    {"name": "k", "value": "v"},
    {"name": "k", "value": "v", "prefix": "cf_"},
    {"name": "k", "value": "v", "prefix": "cf_", "type": {"age": "number"}},
    {"name": "k", "value": "v", "type": {"age": "number", "vip": "boolean", "tier": "text", "name": "date"}},
    {"name": "k", "value": "v", "type": {"age": {"type": "number"}, "vip": {"type": "boolean", "true": ["yes"]}}},
    {"name": "k", "value": "v", "include": ["tier", "age"]},
    {"name": "k", "value": "v", "include": ["nothing"], "prefix": "x."},
    {"name": "nope", "value": "v"},
    {"name": "k"},
])
def test_spread_turns_attribute_rows_into_columns(spread):
    r = router({"customer": {"data": CUST, "key": "cid", "join": {"from": "attr", "foreign_key": "cid", "spread": spread}},
                "attr": {"data": ATTR, "key": "aid"}}, routes={"customer": "customer"})
    assert_parity(r, "customer")


def test_spread_never_overwrites_a_base_field_and_the_last_duplicate_wins():
    attr = pd.DataFrame({"aid": ["1", "2", "3"], "cid": ["a", "a", "a"], "k": ["name", "x", "x"], "v": ["clash", "first", "last"]})
    r = router({"customer": {"data": CUST, "key": "cid", "join": {"from": "attr", "foreign_key": "cid",
                                                                    "spread": {"name": "k", "value": "v"}}},
                "attr": {"data": attr, "key": "aid"}}, routes={"customer": "customer"})
    assert_parity(r, "customer")


def test_spread_after_a_rollup_on_the_same_parent():
    r = router({"customer": {"data": CUST, "key": "cid", "join": [
        {"from": "attr", "foreign_key": "cid", "spread": {"name": "k", "value": "v", "prefix": "cf_"}},
        {"from": "attr", "many": True, "foreign_key": "cid", "aggregate": {"n": {"fn": "count"}}}]},
        "attr": {"data": ATTR, "key": "aid"}}, routes={"customer": "customer"})
    assert_parity(r, "customer")


# --- coercion -----------------------------------------------------------------------------------
RAW = pd.DataFrame({
    "vid": ["1", "2", "3", "4", "5", "6"],
    "num": ["1.234,5", "2.000,0", " 12 ", "1 000", "2*3", "abc"],
    "plain": ["1,234.5", "-7", "+.5", "1e3", "", None],
    "flag": ["yes", "no", "TRUE", "0", "maybe", None],
    "when": ["1700000000", 1700086400, "x", None, 0, 86400],
    "d": ["03/02/2024", "04/02/2024", "31/12/1999", "bad", "", None],
    "iso": ["2024-02-03", "2024-02-03T10:00:00Z", "2024-02-03 10:00", None, "", "2024-13-45"],
    "stamp": [1700000000000, 1700086400000, 0, -86400000, None, 1.5],
    "stamp_text": [1700000000000, "1700086400000", "0", "z", None, " 1.5 "],
    "blank": ["n/a", "x", "N/A", None, "", " "],
    "js": ['{"a": 1}', "[1, 2]", "nope", None, "7", '"s"'],
    "txt": [1, 2.5, True, None, "s", 1e21],
    "ints": ["3", "3.5", 4, 4.0, "x", None],
})


@pytest.mark.parametrize("fields", [
    {"num": {"type": "number", "decimal": ","}},
    {"num": "number", "plain": "number", "ints": "integer"},
    {"flag": "boolean", "iso": {"type": "boolean"}},
    {"flag": {"type": "boolean", "true": ["yes", "TRUE"], "false": ["no", "0"]}},
    {"when": {"type": "date", "format": "epoch-s"}, "stamp": {"type": "date", "format": "epoch-ms"},
     "stamp_text": {"type": "date", "format": "epoch-ms"}},
    {"d": {"type": "date", "format": "dd/MM/yyyy"}, "when": {"type": "date", "format": "yyyy"}},
    {"d": {"type": "date", "format": "d/M/yy"}, "plain": {"type": "date", "format": "yyyy-MM-dd'T'HH:mm"}},
    {"iso": "date"},
    {"iso": {"type": "date", "format": "iso"}, "stamp": "date"},
    {"blank": {"nulls": ["n/a", "N/A"]}, "num": {"type": "text", "nulls": ["abc"]}},
    {"js": "json", "txt": "text"},
    {"missing_col": "number", "ghost": {"type": "text"}},
    {"num": "weird", "plain": {"type": "weird"}},
])
def test_field_coercion(fields):
    r = router({"vals": {"data": RAW, "key": "vid", "fields": fields}}, routes={"vals": "vals"})
    assert_parity(r, "vals")


def test_coercion_of_a_source_that_is_then_rolled_up_and_looked_up():
    t = pd.DataFrame({"tid": ["1", "2", "3", "4"], "cid": ["a", "a", "b", "b"], "amt": ["1.5", "2,5", "x", "4"],
                      "day": ["01/02/2024", "02/02/2024", "03/03/2024", "04/03/2024"]})
    r = router({"customer": {"data": CUST, "key": "cid", "join": {"from": "t", "many": True, "foreign_key": "cid", "aggregate": {
        "total": {"fn": "sum", "field": "amt"}, "first_day": {"fn": "min", "field": "day"}, "n": {"fn": "count"}}}},
        "t": {"data": t, "key": "tid", "fields": {"amt": "number", "day": {"type": "date", "format": "dd/MM/yyyy"}}}},
        routes={"customer": "customer"})
    assert_parity(r, "customer")


def test_a_non_iso_date_text_without_a_format_is_left_to_the_browser():
    r = router({"v": {"data": pd.DataFrame({"i": ["1"], "d": ["Jan 5, 2024"]}), "key": "i", "fields": {"d": "date"}}},
               routes={"v": "v"})
    with pytest.raises(RouterEngineUnsupported, match="format"):
        shape_route(r._payload("v"))


# --- unnest ---------------------------------------------------------------------------------------
ACC = pd.DataFrame({"cid": ["a", "b", "c"], "name": ["A", "B", "C"], "addresses": [
    [{"city": "Leeds", "zip": "LS1", "id": "x1"}, {"city": "York", "zip": "YO1", "id": "x2"}],
    [{"city": "Bath", "zip": "BA1", "id": "x1", "tags": [{"t": 1}, {"t": 2}]}], None]})


@pytest.mark.parametrize("unnest", [
    {"path": "addresses", "as": "address", "parent_key": "acc_id"},
    {"path": "addresses", "as": "address"},
    {"path": "addresses", "as": "address", "keep": True},
    {"path": "addresses", "as": "address", "key": "id"},
    {"path": "addresses", "as": "address", "unnest": {"path": "tags", "as": "tag"}},
    {"path": "addresses", "as": "address", "join": {"from": "zips", "local_key": "zip", "foreign_key": "zip", "fields": ["area"], "missing": "hold"}},
])
@pytest.mark.parametrize("route", ["acc", "address", "tag"])
def test_unnest_expands_nested_arrays_into_their_own_route(unnest, route):
    zips = pd.DataFrame({"zip": ["LS1", "BA1"], "area": ["north", "west"]})
    routes = {"acc": "acc", "address": "address"}
    if "unnest" in unnest:
        routes["tag"] = "tag"
    elif route == "tag":
        pytest.skip("this declaration has no tag route")
    r = router({"acc": {"data": ACC, "key": "cid", "unnest": unnest}, "zips": {"data": zips, "key": "zip"}}, routes=routes)
    assert_parity(r, route)


# --- the whole thing, edge shapes ---------------------------------------------------------------------
def test_empty_sources_and_a_route_with_no_rows():
    r = router({"p": {"data": PARENTS.iloc[0:0], "key": "gid", "join": {"from": "m", "many": True, "foreign_key": "grp",
                                                                      "aggregate": {"n": {"fn": "count"}}}},
                "m": {"data": MESSY.iloc[0:0], "key": "k"}}, routes={"p": "p"})
    assert_parity(r, "p")
    r = router({"p": {"data": PARENTS, "key": "gid", "join": {"from": "m", "many": True, "foreign_key": "grp",
                                                               "aggregate": {"n": {"fn": "count"}}, "missing": "hold"}},
                "m": {"data": MESSY.iloc[0:0], "key": "k"}}, routes={"p": "p"})
    assert_parity(r, "p")


def test_numpy_and_pandas_dtypes_travel_the_same_way():
    n = 30
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"i": np.arange(n), "g": rng.integers(0, 5, n), "f": rng.normal(size=n),
                       "b": rng.random(n) < 0.5, "t": pd.date_range("2024-01-01", periods=n, freq="9h"),
                       "s": pd.array(rng.choice(["a", "b", None], n), dtype="string"),
                       "ni": pd.array(rng.choice([1, None, 3], n), dtype="Int64")})
    par = pd.DataFrame({"g": np.arange(7)})
    r = router({"par": {"data": par, "key": "g", "join": {"from": "df", "many": True, "foreign_key": "g", "aggregate": {
        "n": {"fn": "count"}, "mean": {"fn": "avg", "field": "f"}, "first_t": {"fn": "min", "field": "t"},
        "any_b": {"fn": "max", "field": "b"}, "s": {"fn": "last", "field": "s"}, "ni": {"fn": "sum", "field": "ni"}}}},
        "df": {"data": df, "key": "i", "fields": {"t": "date"}}}, routes={"par": "par"})
    assert_parity(r, "par")
