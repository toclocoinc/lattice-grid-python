"""BACKLOG-0005165: the kernel's answers for a grouped large frame (``grouplevel``).

One level of the grouped view at a time: the groups under a group (ordered by the
sort, an aggregate included, windowed) or the leaf rows of the innermost one.
``test_large_group_browser.py`` holds the same answers against the grid in a browser.
"""

import numpy as np
import pandas as pd
import pytest

from lattice_grid_jupyter import LatticeGridWidget
from test_large_engine import req


def frame(n=2000, seed=11):
    r = np.random.default_rng(seed)
    cat = np.array(["a", "b", "c", "d"], dtype=object)[r.choice(4, n, p=[.5, .3, .15, .05])]
    cat[[7, 40]] = None
    return pd.DataFrame({
        "category": cat,
        "sub": np.array(["x", "y", "z"], dtype=object)[r.integers(0, 3, n)],
        "v": r.integers(0, 100, n).astype("int64"),
    })


TOTALS = [{"id": "v", "col": "v", "fn": "sum"}]


def level(w, values=(), group_by=("category", "sub"), sort=None, rng=(0, 100), **extra):
    query = {"groupBy": list(group_by), "groupValues": list(values), "sort": sort or [],
             "range": {"start": rng[0], "end": rng[1]}, "filters": None, "quick": ""}
    query.update(extra)
    return req(w, "grouplevel", query=query, aggregates=TOTALS)


@pytest.fixture(scope="module")
def df():
    return frame()


@pytest.fixture(scope="module")
def w(df):
    return LatticeGridWidget(df, windowed=True)


def test_top_level_groups_sorted_by_key_with_counts_and_totals(w, df):
    out = level(w, sort=[{"col": "category", "dir": "asc"}])
    sizes = df.groupby("category", dropna=False).size()
    assert [r["category"] for r in out["rows"]] == ["a", "b", "c", "d", None]   # a missing key sorts last
    assert [r["leafCount"] for r in out["rows"]] == [int(sizes.get(k, 0)) if k is not None else 2 for k in ["a", "b", "c", "d", None]]
    assert out["rows"][0]["totals"]["v"] == float(df.loc[df.category == "a", "v"].sum())
    assert out["total"] == 5 and out["leaves"] is False
    assert out["matchCount"] == len(df) and out["grand"]["v"] == float(df.v.sum())


def test_sort_descending_by_key(w):
    out = level(w, sort=[{"col": "category", "dir": "desc"}])
    assert [r["category"] for r in out["rows"]][:4] == ["d", "c", "b", "a"]


def test_sort_by_an_aggregate_orders_the_groups_by_it(w, df):
    sums = df.dropna(subset=["category"]).groupby("category").v.sum()
    for direction in ("asc", "desc"):
        got = [r["category"] for r in level(w, sort=[{"col": "v", "dir": direction}])["rows"] if r["category"] is not None]
        assert got == list(sums.sort_values(ascending=direction == "asc").index)


def test_sort_by_the_count(w, df):
    counts = req(w, "grouplevel", query={"groupBy": ["category"], "groupValues": [], "range": {"start": 0, "end": 10},
                                        "sort": [{"col": "n", "dir": "desc"}]},
                 aggregates=[{"id": "n", "col": "v", "fn": "count"}])
    sizes = df.dropna(subset=["category"]).groupby("category").size().sort_values(ascending=False)
    assert [r["category"] for r in counts["rows"] if r["category"] is not None] == list(sizes.index)


def test_groups_are_windowed_and_the_total_is_the_group_count(w):
    out = level(w, sort=[{"col": "category", "dir": "asc"}], rng=(1, 3))
    assert [r["category"] for r in out["rows"]] == ["b", "c"] and out["total"] == 5


def test_second_level_is_the_groups_under_one_group(w, df):
    out = level(w, values=["a"], sort=[{"col": "sub", "dir": "asc"}])
    under = df[df.category == "a"]
    assert [r["sub"] for r in out["rows"]] == ["x", "y", "z"]
    assert [r["leafCount"] for r in out["rows"]] == [int((under["sub"] == s).sum()) for s in "xyz"]
    assert "matchCount" not in out                       # only the top level carries the grand row


def test_a_missing_key_is_its_own_group_and_can_be_opened(w):
    out = level(w, values=[None])
    assert sum(r["leafCount"] for r in out["rows"]) == 2


def test_leaf_rows_of_the_innermost_group_are_windowed_and_sorted(w, df):
    sort = [{"col": "v", "dir": "desc"}]
    out = level(w, values=["a", "x"], sort=sort, rng=(0, 5))
    want = df[(df.category == "a") & (df["sub"] == "x")].sort_values("v", ascending=False, kind="stable")
    assert out["leaves"] is True and out["total"] == len(want)
    assert [int(k) for k in out["columnar"]["__row_id__"]] == list(want.index[:5])
    assert out["columnar"]["v"] == list(want.v[:5])


def test_filters_narrow_the_groups(w, df):
    filt = {"col": "sub", "op": "eq", "value": "x"}
    out = level(w, sort=[{"col": "category", "dir": "asc"}], filters=filt)
    assert sum(r["leafCount"] for r in out["rows"]) == int((df["sub"] == "x").sum()) and out["matchCount"] == int((df["sub"] == "x").sum())


def test_an_unknown_group_value_is_an_empty_level(w):
    out = level(w, values=["nope"])
    assert out["rows"] == [] and out["total"] == 0
