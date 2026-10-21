"""BACKLOG-0005165: grouping a large frame (windowed source) behaves as in memory.

The widget's real front end in Chromium over a frame above the 100,000-row
threshold (so the grid holds no rows and every group level is asked of the Python
kernel): ``group_default_expanded`` decides which groups start open, the group
order follows the sort (an aggregate included), expanding a group paints its child
rows (no skeleton left), a group collapses and re-opens, and two levels nest.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
from playwright.sync_api import sync_playwright  # noqa: E402

from lattice_grid_jupyter import LatticeGridWidget  # noqa: E402
from test_large_browser import GRID, Session  # noqa: E402

N = 120_000
CATS = ["a", "b", "c", "d"]
COLUMNS = [{"field": "v", "total": "sum"}]


@pytest.fixture(scope="module")
def df():
    r = np.random.default_rng(5)
    return pd.DataFrame({
        "category": np.array(CATS, dtype=object)[r.choice(4, N, p=[.15, .5, .05, .3])],
        "sub": np.array(["x", "y"], dtype=object)[r.integers(0, 2, N)],
        "v": r.integers(0, 100, N).astype("int64"),
    })


@pytest.fixture(scope="module")
def sess():
    with sync_playwright() as p:
        s = Session(p)
        try:
            yield s
        finally:
            s.close()


def open_grouped(sess, df, group, expanded, sort=None):
    w = LatticeGridWidget(df, offline=True, columns=COLUMNS, options={"group_default_expanded": expanded},
                          state={"version": 2, "group": group, **({"sort": sort} if sort else {})})
    assert w.windowed
    page = sess.open(w)
    settle_top(page)
    return page


def settle_top(page, rows=40):
    """Wait until the rows on screen (the first ``rows`` display rows) are loaded: no window in flight."""
    page.wait_for_function(
        f"""(() => {{ const g = {GRID}; const n = Math.min(g.rows.count(), {rows});
          for (let i = 0; i < n; i++) {{ const r = g.rows.get(i); if (!r || (!r.group && !r.data)) return false; }}
          return true; }})()""", timeout=15000)
    page.wait_for_timeout(300)


def settle_top_at(page, at, rows=40):
    """Wait until the ``rows`` display rows from ``at`` are loaded (the grid fetches the window it scrolled to)."""
    page.wait_for_function(
        f"""(() => {{ const g = {GRID};
          for (let i = {at}; i < {at} + {rows}; i++) {{ const r = g.rows.get(i); if (r === undefined) return true;
            if (!r || (!r.group && !r.data)) return false; }}
          return true; }})()""", timeout=15000)
    page.wait_for_timeout(300)


def view(page, limit=None):
    """The display rows: groups (key, level, expanded, leafCount, label) and leaf counts between them."""
    return page.evaluate(f"""(() => {{ const g = {GRID}; const out = [];
      for (let i = 0; i < Math.min(g.rows.count(), {limit or 1e9}); i++) {{ const r = g.rows.get(i);
        if (r && r.group) out.push({{key: r.key, level: r.level || 0, expanded: !!r.expanded,
          leafCount: r.leafCount, label: r.data ? Object.values(r.data)[0] : null}});
        else if (r) out.push({{leaf: true, loaded: !!r.data}}); }}
      return out; }})()""")


def view_head(page, rows=40):
    return view(page, rows)


def groups(page, level=0):
    return [r for r in view(page) if not r.get("leaf") and r["level"] == level]


def no_skeleton(page):
    page.wait_for_function("!document.querySelector('.lat-skeleton-bar, .lat-cell--loading-block')", timeout=15000)


def toggle(page, key, expand):
    page.evaluate(f"{GRID}.rows.{'expand' if expand else 'collapse'}({key!r})")
    settle_top(page)


def test_default_collapsed_groups_start_closed_in_the_sorted_order(sess, df):
    page = open_grouped(sess, df, ["category"], False, sort=[{"col": "category", "dir": "desc"}])
    rows = view(page)
    assert [r["label"] for r in rows] == ["d", "c", "b", "a"]
    assert not any(r["expanded"] for r in rows) and len(rows) == 4          # nothing opened, no child rows
    assert [r["leafCount"] for r in rows] == [int((df.category == c).sum()) for c in "dcba"]


def test_default_expanded_true_opens_every_group_and_paints_its_rows(sess, df):
    page = open_grouped(sess, df, ["category"], True, sort=[{"col": "category", "dir": "asc"}])
    no_skeleton(page)
    top = view_head(page)                                           # the rows on screen
    assert top[0]["expanded"] and top[0]["label"] == "a" and all(r.get("loaded", True) for r in top)
    at = 0                                                          # scroll group by group: each opens onto painted rows
    for c in CATS:
        page.evaluate(f"{GRID}.scroll.toRow({at})")
        settle_top_at(page, at)
        row = page.evaluate(f"(() => {{ const r = {GRID}.rows.get({at}); return [r.group, r.expanded, r.data && Object.values(r.data)[0]]; }})()")
        assert row == [True, True, c]
        at += 1 + int((df.category == c).sum())


def test_group_order_follows_a_sort_on_an_aggregate(sess, df):
    sums = df.groupby("category").v.sum()
    for direction in ("desc", "asc"):
        page = open_grouped(sess, df, ["category"], False, sort=[{"col": "v", "dir": direction}])
        assert [r["label"] for r in groups(page)] == list(sums.sort_values(ascending=direction == "asc").index)


def test_group_order_follows_a_sort_on_the_count(sess, df):
    cols = [{"field": "v", "total": "count"}]
    w = LatticeGridWidget(df, offline=True, columns=cols, options={"group_default_expanded": False},
                          state={"version": 2, "group": ["category"], "sort": [{"col": "v", "dir": "desc"}]})
    page = sess.open(w)
    settle_top(page)
    sizes = df.groupby("category").size().sort_values(ascending=False)
    assert [r["label"] for r in groups(page)] == list(sizes.index)
    assert [r["leafCount"] for r in groups(page)] == list(sizes.values)


def test_expanding_a_group_loads_and_paints_its_child_rows_and_it_collapses_and_reopens(sess, df):
    page = open_grouped(sess, df, ["category"], False, sort=[{"col": "category", "dir": "asc"}])
    key = groups(page)[2]["key"]                                     # "c"
    want = int((df.category == "c").sum())
    for _ in range(2):                                              # open, close, open again
        toggle(page, key, True)
        no_skeleton(page)
        rows = view(page)
        assert len(rows) == 4 + want and all(r.get("loaded", True) for r in rows[:40])       # the rows on screen are painted
        page.wait_for_selector(".lat-cell[role=gridcell]")
        assert "Loading rows" not in page.evaluate("document.body.innerText")
        toggle(page, key, False)
        assert len(view(page)) == 4 and not any(r["expanded"] for r in groups(page))
    toggle(page, key, True)
    assert len(view(page)) == 4 + want


def test_two_level_grouping_with_a_level_number_opens_only_that_many_levels(sess, df):
    page = open_grouped(sess, df, ["category", "sub"], 1, sort=[{"col": "category", "dir": "asc"}])
    top, inner = groups(page, 0), groups(page, 1)
    assert [r["label"] for r in top] == CATS and all(r["expanded"] for r in top)
    assert len(inner) == 8 and not any(r["expanded"] for r in inner)
    assert [r["label"] for r in inner[:2]] == ["x", "y"]
    assert inner[0]["leafCount"] == int(((df.category == "a") & (df["sub"] == "x")).sum())


def test_two_level_grouping_children_load_when_the_inner_group_opens(sess, df):
    page = open_grouped(sess, df, ["category", "sub"], False, sort=[{"col": "category", "dir": "asc"}])
    assert len(view(page)) == 4
    toggle(page, groups(page)[0]["key"], True)
    inner = groups(page, 1)
    assert [r["label"] for r in inner] == ["x", "y"] and not any(r["expanded"] for r in inner)
    toggle(page, inner[1]["key"], True)
    no_skeleton(page)
    want = int(((df.category == "a") & (df["sub"] == "y")).sum())
    rows = view(page)
    assert len(rows) == 4 + 2 + want and all(r.get("loaded", True) for r in rows[:40])
