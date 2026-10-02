"""Card 1618: large frames -- the threshold, the comm protocol and every pushed
operation, answered in Python over the whole frame.

These are the Python-side unit tests. ``test_large_browser.py`` holds the same
operations against the grid's own client-side path in a real browser.
"""

import os

import numpy as np
import pandas as pd
import pytest

from lattice_grid_jupyter import LatticeGridWidget
from lattice_grid_jupyter._engine import PandasEngine
from lattice_grid_jupyter.widget import DEFAULT_LARGE_THRESHOLD


def frame(n=60, seed=3):
    r = np.random.default_rng(seed)
    names = np.array(["Ada", "grace", "Linus", "ken", "Barbara", "alan"])
    df = pd.DataFrame({
        "name": names[r.integers(0, len(names), n)].astype(object),
        "score": r.integers(0, 50, n).astype("int64"),
        "ratio": np.round(r.random(n) * 10, 3),
        "kind": pd.Categorical(np.array(["a", "b", "c"])[r.integers(0, 3, n)]),
        "ok": r.random(n) > 0.5,
    })
    df.loc[[i for i in (3, 17) if i < n], "ratio"] = np.nan
    if n > 5:
        df.loc[[5], "name"] = None
    return df


def req(w, method, rid=1, **payload):
    """Play the browser: one request through the widget's comm handler."""
    sent = []
    w.send = lambda content, buffers=None: sent.append(content)
    w._on_custom(w, {"type": "lg:req", "id": rid, "method": method, "payload": payload}, [])
    assert len(sent) == 1 and sent[0]["type"] == "lg:res" and sent[0]["id"] == rid, sent
    assert sent[0]["ok"], sent[0].get("error")
    return sent[0]["result"]


def keys_of(result):
    return [int(k) for k in result["columnar"]["__row_id__"]]


# --- the threshold ------------------------------------------------------------
def test_default_threshold_is_documented_100k():
    assert DEFAULT_LARGE_THRESHOLD == 100_000


def test_threshold_crossing_switches_to_the_windowed_source():
    df = frame(100)
    below = LatticeGridWidget(df, large_threshold=101)
    at = LatticeGridWidget(df, large_threshold=100)
    assert below._mode == "client" and below._columnar["__row_id__"][:2] == ["0", "1"]
    assert at._mode == "windowed" and at._columnar == {}      # no rows sent up front
    assert at._columns == below._columns                       # same schema either way
    assert set(at._engine_caps["operators"]) >= {"eq", "contains", "in", "between"}


def test_windowed_override_both_ways():
    assert LatticeGridWidget(frame(10), windowed=True)._mode == "windowed"
    assert LatticeGridWidget(frame(10), windowed=False, large_threshold=1)._mode == "client"


def test_set_data_crossing_the_threshold_switches_source_live():
    w = LatticeGridWidget(frame(20), large_threshold=50)
    v0 = w._data_version
    w.set_data(frame(80))
    assert w._mode == "windowed" and w._columnar == {} and w._data_version == v0 + 1
    w.set_data(frame(10))
    assert w._mode == "client" and len(w._columnar["__row_id__"]) == 10


# --- the comm protocol --------------------------------------------------------
def test_request_answered_with_its_correlation_id():
    w = LatticeGridWidget(frame(), windowed=True)
    res = req(w, "execute", rid=42, query={"range": {"start": 0, "end": 5}})
    assert keys_of(res) == [0, 1, 2, 3, 4] and res["total"] == 60


def test_engine_error_is_answered_not_raised():
    w = LatticeGridWidget(frame(), windowed=True)
    sent = []
    w.send = lambda content, buffers=None: sent.append(content)
    w._on_custom(w, {"type": "lg:req", "id": 7, "method": "execute",
                     "payload": {"query": {"filters": {"col": "nope", "op": "eq", "value": 1}}}}, [])
    assert sent[0]["id"] == 7 and sent[0]["ok"] is False and "nope" in sent[0]["error"]


# --- window request -----------------------------------------------------------
def test_window_request_returns_only_that_window():
    df = frame(60)
    w = LatticeGridWidget(df, windowed=True)
    res = req(w, "execute", query={"range": {"start": 20, "end": 30}})
    assert keys_of(res) == list(range(20, 30))
    assert res["columnar"]["score"] == df["score"].iloc[20:30].tolist()
    assert res["total"] == 60
    assert len(res["columnar"]["name"]) == 10


# --- sort pushed --------------------------------------------------------------
def test_sort_pushed_desc_with_nulls_last_and_stable_ties():
    df = frame(60)
    w = LatticeGridWidget(df, windowed=True)
    res = req(w, "execute", query={"sort": [{"col": "ratio", "dir": "desc"}], "range": {"start": 0, "end": 60}})
    got = keys_of(res)
    expect = df["ratio"].sort_values(ascending=False, kind="stable", na_position="last").index.tolist()
    assert got == expect
    assert got[-2:] == [3, 17]  # the NaNs, last, in frame order


def test_multi_sort_pushed():
    df = frame(60)
    w = LatticeGridWidget(df, windowed=True)
    res = req(w, "execute", query={"sort": [{"col": "kind", "dir": "asc"}, {"col": "score", "dir": "desc"}],
                                   "range": {"start": 0, "end": 60}})
    expect = df.assign(k=df["kind"].astype(str)).sort_values(
        ["k", "score"], ascending=[True, False], kind="stable").index.tolist()
    assert keys_of(res) == expect


def test_text_sort_follows_collation_not_codepoints():
    df = pd.DataFrame({"name": ["banana", "Apple", "apple", "Cherry", None]})
    w = LatticeGridWidget(df, windowed=True)
    res = req(w, "execute", query={"sort": [{"col": "name", "dir": "asc"}], "range": {"start": 0, "end": 9}})
    assert res["columnar"]["name"] == ["apple", "Apple", "banana", "Cherry", None]


# --- filter pushed ------------------------------------------------------------
@pytest.mark.parametrize("cond,expect", [
    ({"col": "score", "op": "gt", "value": 40}, lambda d: d["score"] > 40),
    ({"col": "score", "op": "between", "value": [10, 20]}, lambda d: d["score"].between(10, 20)),
    ({"col": "name", "op": "eq", "value": "ada"}, lambda d: d["name"].str.lower() == "ada"),
    ({"col": "name", "op": "ne", "value": "ada"}, lambda d: ~(d["name"].str.lower() == "ada").fillna(False)),
    ({"col": "name", "op": "contains", "value": "AR"}, lambda d: d["name"].str.lower().str.contains("ar").fillna(False)),
    ({"col": "name", "op": "in", "value": ["Ken", "alan"]}, lambda d: d["name"].str.lower().isin(["ken", "alan"])),
    ({"col": "ratio", "op": "blank"}, lambda d: d["ratio"].isna()),
    ({"col": "kind", "op": "eq", "value": "B"}, lambda d: d["kind"] == "b"),
    ({"col": "ok", "op": "eq", "value": True}, lambda d: d["ok"]),
])
def test_filter_pushed_matches_the_grid_semantics(cond, expect):
    df = frame(60)
    w = LatticeGridWidget(df, windowed=True)
    res = req(w, "execute", query={"filters": cond, "range": {"start": 0, "end": 100}})
    want = df.index[expect(df).fillna(False).astype(bool)].tolist()
    assert keys_of(res) == want and res["total"] == len(want)


def test_filter_tree_and_or_not():
    df = frame(60)
    w = LatticeGridWidget(df, windowed=True)
    tree = {"op": "and", "conditions": [
        {"op": "or", "conditions": [{"col": "kind", "op": "eq", "value": "a"},
                                    {"col": "score", "op": "lt", "value": 5}]},
        {"op": "not", "conditions": [{"col": "ok", "op": "eq", "value": True}]},
    ]}
    res = req(w, "execute", query={"filters": tree, "range": {"start": 0, "end": 100}})
    m = ((df["kind"] == "a") | (df["score"] < 5)) & ~df["ok"]
    assert keys_of(res) == df.index[m].tolist()


def test_quick_search_pushed():
    df = frame(60)
    w = LatticeGridWidget(df, windowed=True)
    res = req(w, "execute", query={"quick": "LIN", "range": {"start": 0, "end": 100}})
    assert keys_of(res) == df.index[df["name"].fillna("").str.lower().str.contains("lin")].tolist()


# --- group-by aggregates / distinct values / statistics ------------------------
def test_grouped_aggregates_pushed():
    df = frame(60)
    w = LatticeGridWidget(df, windowed=True)
    out = req(w, "grouped", query={"filters": {"col": "score", "op": "gte", "value": 10}},
              groupBy=["kind"], aggregates=[{"id": "score", "col": "score", "fn": "sum"},
                                            {"id": "n", "col": "kind", "fn": "count"}])
    sub = df[df["score"] >= 10]
    level1 = {e["keys"][0]: e["values"] for e in out if e["level"] == 1}
    assert set(level1) == set(sub["kind"].astype(str))
    for k, v in level1.items():
        assert v["score"] == sub.loc[sub["kind"] == k, "score"].sum()
        assert v["n"] == int((sub["kind"] == k).sum())
    grand = [e for e in out if e["level"] == 0][0]["values"]
    assert grand["n"] == len(sub) and grand["score"] == sub["score"].sum()


def test_distinct_values_for_the_filter_menu():
    df = frame(60)
    w = LatticeGridWidget(df, windowed=True)
    # exactly what the grid's columnValues() asks: a group-by with a count,
    # narrowed by the menu's search
    out = req(w, "grouped", query={"filters": {"col": "name", "type": "text", "op": "contains", "value": "a"}},
              groupBy=["name"], aggregates=[{"id": "__lattice_value_count", "col": "name", "fn": "count"}])
    got = {e["keys"][0]: e["values"]["__lattice_value_count"] for e in out if e["level"] == 1}
    s = df["name"].dropna()
    want = s[s.str.lower().str.contains("a")].value_counts().to_dict()
    assert got == want


def test_whole_set_statistics():
    df = frame(60)
    w = LatticeGridWidget(df, windowed=True)
    out = req(w, "aggregates", query={}, aggregates=[
        {"id": "a", "col": "ratio", "fn": "avg"}, {"id": "m", "col": "ratio", "fn": "median"},
        {"id": "d", "col": "name", "fn": "distinct"}, {"id": "c", "col": "ratio", "fn": "count"}])
    assert out["a"] == pytest.approx(df["ratio"].mean())
    assert out["m"] == pytest.approx(df["ratio"].median())
    assert out["d"] == df["name"].nunique() and out["c"] == 60


# --- histograms / profile -----------------------------------------------------
def test_header_histogram_counts_over_the_whole_frame():
    df = frame(60)
    w = LatticeGridWidget(df, windowed=True)
    out = req(w, "facet", colId="score", filters=None, quick="", bounds=None, buckets=5)
    assert out["bounds"]["kind"] == "numeric" and len(out["counts"]) == 5
    assert sum(out["counts"]) == 60 and out["unfiltered"] == out["counts"]
    # cross-filtered: the same bounds, counted under another column's filter
    again = req(w, "facet", colId="score", filters={"col": "kind", "op": "eq", "value": "a"},
                quick="", bounds=out["bounds"], buckets=5)
    assert sum(again["counts"]) == int((df["kind"] == "a").sum()) and "unfiltered" not in again


def test_profile_over_the_whole_frame():
    df = frame(60)
    w = LatticeGridWidget(df, windowed=True)
    p = req(w, "profile", colId="ratio", filters=None, quick="")
    assert p["rows"] == 60 and p["missing"] == 2 and p["numeric"] == 58
    assert p["mean"] == pytest.approx(df["ratio"].mean())
    assert p["median"] == pytest.approx(df["ratio"].median())
    assert sum(b["count"] for b in p["histogram"]) == 58
    t = req(w, "profile", colId="name", filters=None, quick="")
    assert "topValues" in t and t["topValues"][0]["count"] == df["name"].value_counts().iloc[0]


# --- edits, selected, view ----------------------------------------------------
def test_edit_over_the_windowed_source_reaches_the_frame():
    w = LatticeGridWidget(frame(60), windowed=True)
    assert w._engine_caps["editable"] is True
    assert req(w, "mutate", op={"kind": "update", "key": "4", "patch": {"score": "77"}}) == {"ok": True}
    assert w.df["score"].iloc[4] == 77 and w.df["score"].dtype == np.int64
    res = req(w, "execute", query={"filters": {"col": "score", "op": "eq", "value": 77},
                                   "range": {"start": 0, "end": 9}})
    assert 4 in keys_of(res)  # caches dropped: the edit is visible to the next query


def test_selected_and_lazy_view_when_windowed():
    df = frame(60)
    df.index = pd.Index(range(1000, 1060), name="id")
    w = LatticeGridWidget(df, windowed=True)
    w._selected_keys = ["7", "2"]
    assert list(w.selected.index) == [1002, 1007] and (w.selected.dtypes == df.dtypes).all()
    w.state = {"version": 2, "filters": {"col": "score", "op": "lt", "value": 10},
               "sort": [{"col": "score", "dir": "asc"}]}
    v = w.view
    assert type(v).__name__ == "LazyView" and v._pos is None   # nothing computed yet
    sub = df[df["score"] < 10].sort_values("score", kind="stable")
    assert len(v) == len(sub)
    assert list(v.head(3).index) == list(sub.index[:3])
    assert (v.to_pandas()["score"].values == sub["score"].values).all()


def test_observe_view_fires_on_state_change_when_windowed():
    w = LatticeGridWidget(frame(60), windowed=True)
    seen = []
    w.observe(lambda c: seen.append(len(c["new"])), names="view")
    w.state = {"version": 2, "quick": "ada"}
    w.state = {"version": 2, "quick": "ada", "columns": {"x": 1}}   # not a view change
    w.state = {"version": 2, "quick": "linus"}
    assert len(seen) == 2


# --- Parquet / Polars ---------------------------------------------------------
QUERIES = [
    ("execute", dict(query={"sort": [{"col": "name", "dir": "asc"}, {"col": "score", "dir": "desc"}],
                            "filters": {"col": "ratio", "op": "notBlank"}, "range": {"start": 3, "end": 40}})),
    ("execute", dict(query={"quick": "a", "range": {"start": 0, "end": 60}})),
    ("grouped", dict(query={}, groupBy=["name"], aggregates=[{"id": "s", "col": "score", "fn": "avg"}])),
    ("facet", dict(colId="ratio", filters=None, quick="", bounds=None, buckets=8)),
    ("facet", dict(colId="kind", filters=None, quick="", bounds=None, buckets=8)),
    ("profile", dict(colId="score", filters=None, quick="")),
]


@pytest.mark.parametrize("source", ["parquet", "polars"])
def test_parquet_and_polars_answer_exactly_as_pandas(source, tmp_path):
    pytest.importorskip("pyarrow")
    df = frame(60)
    if source == "parquet":
        pytest.importorskip("duckdb")
        data = os.path.join(tmp_path, "f.parquet")
        df.to_parquet(data)
    else:
        pl = pytest.importorskip("polars")
        data = pl.from_pandas(df)
    big = LatticeGridWidget(data, large_threshold=10)
    ref = LatticeGridWidget(df, windowed=True)
    assert big._mode == "windowed" and big._engine.source == source
    assert [c["field"] for c in big._columns] == [c["field"] for c in ref._columns]
    for method, payload in QUERIES:
        a, b = req(big, method, **payload), req(ref, method, **payload)
        if method == "grouped":
            a = sorted(a, key=lambda e: (e["level"], str(e["keys"])))
            b = sorted(b, key=lambda e: (e["level"], str(e["keys"])))
        assert a == b, method
    with pytest.raises(AttributeError):
        big.df  # noqa: B018 -- no pandas frame behind it
    assert len(big.view) == 60


def test_parquet_below_the_threshold_is_the_client_path(tmp_path):
    pytest.importorskip("duckdb")
    df = frame(30)
    path = os.path.join(tmp_path, "f.parquet")
    df.to_parquet(path)
    w = LatticeGridWidget(path)
    assert w._mode == "client" and len(w._columnar["__row_id__"]) == 30
    pd.testing.assert_frame_equal(w.df.reset_index(drop=True), df.reset_index(drop=True),
                                  check_categorical=False, check_dtype=False)


def test_engine_caches_one_order_per_query():
    e = PandasEngine(frame(60))
    q = {"sort": [{"col": "score", "dir": "asc"}]}
    a = e.positions(None, "", q["sort"])
    assert e.positions(None, "", q["sort"]) is a     # scrolling re-slices, never re-sorts
