"""BACKLOG-0005164: a DuckDB relation / connection as the grid source -- every request is SQL.

* a generated 5,000,000-row DuckDB table: each request kind is answered by SQL (read off the
  engine's captured SQL log), no statement hands Python more than a window / a few groups, and
  Python's allocations stay tiny (nothing is materialised);
* a seeded sample: the answers are equal to the existing Arrow path's, for a connection + table,
  a connection + sql and a relation;
* user values are bound parameters (a quote in a filter value works and drops nothing);
* a Parquet path with ``pushdown=True`` takes the same route;
* a notebook using a relation runs under ``Run All`` (nbclient).
"""

import json
import re
import textwrap
import tracemalloc

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from lattice_grid_jupyter import LatticeGridWidget
from lattice_grid_jupyter._arrow_engine import ArrowEngine
from lattice_grid_jupyter._duckdb_engine import DuckDBEngine

BIG = 5_000_000


def L(col, op, value=None, **extra):
    return {"col": col, "op": op, "value": value, **extra}


def req(w, method, rid=1, **payload):
    sent = []
    w.send = lambda content, buffers=None: sent.append(content)
    w._on_custom(w, {"type": "lg:req", "id": rid, "method": method, "payload": payload}, [])
    assert len(sent) == 1 and sent[0]["ok"], sent
    return sent[0]["result"]


# --- the 5M-row table -----------------------------------------------------------
@pytest.fixture(scope="module")
def big_con():
    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE big AS SELECT
            i,
            'name' || (i % 50000) || CASE WHEN i % 1000 = 0 THEN ' O''Brien "Q"' ELSE '' END AS name,
            (hash(i) % 100000) / 1000.0 AS val,
            i % 7 AS g,
            CASE WHEN i % 11 = 0 THEN NULL ELSE (i * 7 % 1000)::INTEGER END AS opt,
            (i % 3 = 0) AS flag,
            TIMESTAMP '2020-01-01' + to_seconds(i * 60) AS ts
        FROM range({BIG}) t(i)""")
    return con


def _sql_text(e):
    return [s for s, _, _ in e.sql_log]


def _max_rows(e):
    return max((n for _, _, n in e.sql_log), default=0)


def test_every_request_kind_is_sql_and_nothing_is_materialised(big_con):
    w = LatticeGridWidget(con=big_con, table="big")
    e = w._engine
    assert isinstance(e, DuckDBEngine) and w.windowed and w._mode == "windowed" and w._columnar == {}
    assert len(e) == BIG
    tracemalloc.start()
    try:
        # row window (unsorted, unfiltered)
        e.sql_log.clear()
        r = req(w, "execute", query={"range": {"start": 2_500_000, "end": 2_500_050}})
        assert r["total"] == BIG and r["columnar"]["__row_id__"][0] == "2500000"
        assert len(r["columnar"]["i"]) == 50 and r["columnar"]["i"][0] == 2_500_000
        assert any("LIMIT" in s and "OFFSET" in s for s in _sql_text(e))

        # sort + filter + quick search, windowed
        e.sql_log.clear()
        q = {"filters": {"op": "and", "conditions": [L("val", "gt", 50), L("name", "startsWith", "name1")]},
             "quick": "name12", "sort": [{"col": "val", "dir": "desc"}], "range": {"start": 10, "end": 40}}
        r = req(w, "execute", query=q)
        vals = r["columnar"]["val"]
        assert len(vals) == 30 and vals == sorted(vals, reverse=True) and min(vals) > 50
        assert all(v.startswith("name12") for v in r["columnar"]["name"])
        text = " ".join(_sql_text(e))
        assert "WHERE" in text and "ORDER BY" in text and "count(*)" in text and "LIMIT" in text
        assert all(isinstance(p, (int, float, str, list)) for _, ps, _ in e.sql_log for p in ps.values())
        assert any("name12" == v for _, ps, _ in e.sql_log for v in ps.values())   # bound, not spliced
        assert not re.search(r"name12|name1\b", text)

        # grouping with aggregates (and the child windows of one group)
        e.sql_log.clear()
        groups = req(w, "grouped", query={"filters": L("flag", "eq", True)}, groupBy=["g", "flag"],
                     aggregates=[{"id": "n", "col": "i", "fn": "count"}, {"id": "s", "col": "val", "fn": "sum"},
                                 {"id": "m", "col": "val", "fn": "median"}])
        assert "ROLLUP" in " ".join(_sql_text(e))
        leaf = [g for g in groups if g["level"] == 2]
        assert len(leaf) == 7 and sum(g["values"]["n"] for g in leaf) == (BIG + 2) // 3
        assert [g for g in groups if g["level"] == 0][0]["values"]["n"] == (BIG + 2) // 3
        e.sql_log.clear()
        child = req(w, "execute", query={"filters": {"op": "and", "conditions": [L("g", "eq", 3), L("flag", "eq", True)]},
                                         "range": {"start": 0, "end": 25}})
        assert child["total"] == [g for g in leaf if g["keys"][0] == 3][0]["values"]["n"]

        # distinct values a filter menu lists: a group-by with a count
        d = req(w, "grouped", query={}, groupBy=["g"], aggregates=[{"id": "n", "col": "g", "fn": "count"}])
        assert sorted(x["keys"][0] for x in d if x["level"] == 1) == list(range(7))

        # whole-set statistics
        e.sql_log.clear()
        agg = req(w, "aggregates", query={"filters": L("opt", "gte", 500)},
                  aggregates=[{"id": "c", "col": "i", "fn": "count"}, {"id": "med", "col": "val", "fn": "median"},
                              {"id": "sd", "col": "val", "fn": "stddev"}, {"id": "dv", "col": "opt", "fn": "distinct"}])
        assert agg["c"] > 0 and 0 < agg["med"] < 100 and agg["dv"] == 500
        assert "quantile_cont" in " ".join(_sql_text(e)) and "stddev_samp" in " ".join(_sql_text(e))

        # header histograms (numeric, category) and the Statistics tab
        e.sql_log.clear()
        hv = req(w, "facet", colId="val", filters=L("g", "eq", 2))
        assert len(hv["counts"]) == 20
        assert sum(hv["unfiltered"]) == BIG and 0 < sum(hv["counts"]) < BIG
        assert "FILTER" in " ".join(_sql_text(e))
        hn = req(w, "facet", colId="name")
        assert hn["bounds"].get("suppressed") == "cardinality"
        e.sql_log.clear()
        pr = req(w, "profile", colId="val", filters=None)
        assert pr["rows"] == BIG and pr["numeric"] == BIG and pr["histogram"] and pr["median"] and pr["q1"] < pr["q3"]
        assert sum(b["count"] for b in pr["histogram"]) == BIG
        pt = req(w, "profile", colId="name")
        assert len(pt["topValues"]) == 5 and "topValues" not in pr
        assert "quantile_cont" in " ".join(_sql_text(e))

        # w.view: a lazy handle
        w.state = {"version": 2, "filters": L("g", "eq", 4), "sort": [{"col": "i", "dir": "desc"}]}
        v = w.view
        assert len(v) == len([i for i in range(4, BIG, 7)])
        head = v.head(3)
        last = 4_999_999 - ((4_999_999 - 4) % 7)
        assert head["i"].tolist() == [last, last - 7, last - 14]
        assert "<LazyView" in repr(v)
        # nothing wider than a window / a handful of groups ever reached Python
        assert _max_rows(e) <= 25_000   # the 20,000-value sample that decides a text column's type is the widest
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 60_000_000, f"Python allocated {peak / 1e6:.1f} MB over a {BIG:,}-row table"


def test_rss_stays_bounded_serving_windows_from_5m_rows(big_con, tmp_path):
    """Windows, filter, quick search, grouping and a histogram on 5M rows grow the process by far
    less than holding the table in Python would (about 600 MB as Arrow, more as pandas)."""
    import subprocess
    import sys
    script = tmp_path / "rss.py"
    script.write_text(textwrap.dedent(f"""
        import subprocess, os, duckdb
        from lattice_grid_jupyter import LatticeGridWidget
        def rss():
            return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).split()[0]) / 1024
        con = duckdb.connect()
        con.execute("SET threads TO 4")
        con.execute("SET memory_limit TO '300MB'")
        con.execute("CREATE TABLE big AS SELECT i, 'n' || (i % 50000) AS name, (hash(i) % 100000) / 1000.0 AS val, "
                    "i % 7 AS g FROM range({BIG}) t(i)")
        w = LatticeGridWidget(con=con, table="big")
        e = w._engine
        before = rss()
        e.execute({{"range": {{"start": 10, "end": 110}}}})
        e.execute({{"sort": [{{"col": "val"}}], "range": {{"start": 0, "end": 100}}}})
        e.execute({{"filters": {{"col": "name", "op": "contains", "value": "99"}}, "quick": "n12", "range": {{"start": 0, "end": 100}}}})
        e.grouped_aggregates({{}}, ["g"], [{{"id": "s", "col": "val", "fn": "sum"}}])
        e.facet({{"colId": "val"}})
        e.profile("val")
        print("DELTA", rss() - before)
    """))
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    delta = float(out.stdout.split("DELTA")[1])
    assert delta < 450, f"RSS grew {delta:.0f} MB"


# --- bound parameters -------------------------------------------------------------
def test_user_values_are_bound_never_spliced(big_con):
    w = LatticeGridWidget(con=big_con, table="big")
    e = w._engine
    for value in ["name7 O'Brien \"Q\"", "x'; DROP TABLE big; --", "100%_\\"]:
        e.sql_log.clear()
        r = req(w, "execute", query={"filters": L("name", "contains", value), "range": {"start": 0, "end": 5}})
        assert isinstance(r["total"], int)
        for sql, params, _ in e.sql_log:
            assert value.lower() not in sql.lower() and value.lower() in params.values()
    e.sql_log.clear()
    r = req(w, "execute", query={"filters": L("name", "eq", "name1000 O'Brien \"Q\"")})
    assert r["total"] == BIG // 50000 and r["columnar"]["i"][:2] == [1000, 51000]
    r = req(w, "execute", query={"filters": L("name", "contains", "O'Brien"), "range": {"start": 0, "end": 3}})
    assert r["total"] == BIG // 1000 and all("O'Brien" in n for n in r["columnar"]["name"])
    r = req(w, "execute", query={"quick": "O'Brien \"Q\"", "range": {"start": 0, "end": 3}})
    assert r["total"] == BIG // 1000
    assert big_con.execute("SELECT count(*) FROM big").fetchone()[0] == BIG   # nothing dropped
    for sql, params, _ in e.sql_log:
        assert "Brien" not in sql
    # an identifier that is not a column is refused, never quoted into a statement
    with pytest.raises(KeyError):
        e.execute({"filters": L('i" ; DROP TABLE big; --', "eq", 1)})
    assert big_con.execute("SELECT count(*) FROM big").fetchone()[0] == BIG


# --- equality with the Arrow path ---------------------------------------------------
def _sample(n=3000, seed=5):
    r = np.random.default_rng(seed)
    names = np.array(["Ada", "grace", "Linus", "ken's", "Barbara", "alan", "ALAN", "", "x%y", 'O"Q'], dtype=object)
    df = pd.DataFrame({
        "name": names[r.integers(0, len(names), n)],
        "score": r.integers(0, 50, n),
        "ratio": np.round(r.random(n) * 10, 3),
        "opt": pd.array(r.integers(0, 9, n), dtype="Int64"),
        "ok": r.random(n) > .5,
        "ts": pd.Timestamp("2020-01-01") + pd.to_timedelta(r.integers(0, 400 * 86400, n), unit="s"),
        "kind": np.array(["a", "b", "c"])[r.integers(0, 3, n)],
    })
    df.loc[::17, "ratio"] = np.nan
    df.loc[::23, "name"] = None
    df.loc[::29, "ts"] = pd.NaT
    df.loc[::31, "opt"] = pd.NA
    return df


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    path = tmp_path_factory.mktemp("duck") / "sample.parquet"
    pq.write_table(pa.Table.from_pandas(_sample(), preserve_index=False), str(path))
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM read_parquet(?)", [str(path)])
    arrow = ArrowEngine(str(path))
    engines = {
        "con+table": DuckDBEngine(con=con, table="t"),
        "con+sql": DuckDBEngine(con=con, sql="SELECT * FROM t;"),
        "relation": DuckDBEngine(relation=con.table("t")),
    }
    return arrow, engines, path


def _norm(x):
    return json.loads(json.dumps(x, default=str))


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return a is b if a is None or b is None else abs(a - b) <= 1e-6 * max(1, abs(a))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _agree(pair, what, fa, fb):
    arrow, engines, _ = pair
    want = _norm(fa(arrow))
    for name, e in engines.items():
        got = _norm(fb(e))
        assert _same(want, got), f"{what} [{name}]\n arrow: {str(want)[:400]}\n duck:  {str(got)[:400]}"


FILTERS = [
    L("name", "eq", "ada"), L("name", "eq", "ada", caseSensitive=True), L("name", "ne", "ken's"),
    L("name", "contains", "a"), L("name", "notContains", "a"), L("name", "startsWith", "A"),
    L("name", "endsWith", "N"), L("name", "matches", "^a.*n$"), L("name", "in", ["ADA", "ken's", None]),
    L("name", "notIn", ["ada"]), L("name", "blank"), L("name", "notBlank"), L("name", "lt", "b"),
    L("name", "gte", "grace"), L("name", "between", ["b", "l"]), L("name", "between", ["b", "l"], bounds="()"),
    L("name", "notBetween", ["b", "l"]), L("name", "contains", "x%y"), L("name", "eq", None),
    L("name", "eq", 'O"Q'), L("name", "eq", "ken's"),
    L("score", "eq", 7), L("score", "ne", 7), L("score", "gt", 30), L("score", "lte", "12"),
    L("score", "between", [10, 20]), L("score", "notBetween", [10, 20]), L("score", "in", [1, 2, "3"]),
    L("score", "notIn", [1, 2]), L("score", "contains", "1"), L("score", "eq", "abc"),
    L("ratio", "gt", 5.5), L("ratio", "eq", None), L("ratio", "blank"), L("ratio", "ne", 5),
    L("ratio", "startsWith", "3"), L("opt", "gte", 4), L("opt", "in", [1, None]),
    L("ok", "eq", True), L("ok", "eq", "false"),
    L("ts", "gt", "2020-06-01"), L("ts", "between", ["2020-03-01", "2020-04-01"]), L("ts", "eq", None),
    L("ts", "contains", "2020-02"), L("ts", "lt", 1600000000000),
    L("kind", "eq", "B"), L("kind", "in", ["a", "c"]),
    {"op": "or", "conditions": [L("score", "lt", 3), L("name", "eq", "ada")]},
    {"op": "not", "conditions": [L("score", "lt", 30), L("ok", "eq", True)]},
    {"op": "and", "conditions": [L("score", "lt", 30), {"op": "or", "conditions": [L("kind", "eq", "a"), L("ok", "eq", True)]}]},
]
SORTS = [None, [{"col": "score", "dir": "asc"}], [{"col": "name", "dir": "desc"}, {"col": "score"}],
         [{"col": "ratio", "dir": "desc", "nullsFirst": True}], [{"col": "ts"}, {"col": "name"}],
         [{"col": "ok", "dir": "desc"}, {"col": "kind"}], [{"col": "name"}]]


@pytest.mark.parametrize("f", FILTERS, ids=lambda f: json.dumps(f, default=str)[:60])
def test_filter_operators_equal_the_arrow_path(pair, f):
    q = {"filters": f, "range": {"start": 0, "end": 60}}
    _agree(pair, "filter", lambda e: e.execute(q), lambda e: e.execute(q))


@pytest.mark.parametrize("sort", SORTS, ids=lambda s: json.dumps(s)[:50])
@pytest.mark.parametrize("quick", ["", "an", "2020-0", "tru", "7"])
def test_sort_and_quick_search_equal_the_arrow_path(pair, sort, quick):
    q = {"sort": sort, "quick": quick, "filters": L("score", "lt", 40), "range": {"start": 5, "end": 50}}
    _agree(pair, "sort", lambda e: e.execute(q), lambda e: e.execute(q))


def test_aggregates_groups_facets_and_profiles_equal_the_arrow_path(pair):
    fns = ("count", "countValues", "sum", "avg", "min", "max", "range", "distinct", "median", "p25", "p75", "p90",
           "p95", "p99", "iqr", "variance", "varianceP", "stddev", "stddevP", "sumSquares")
    aggs = [{"id": f"{c}_{fn}", "col": c, "fn": fn} for c in ("score", "ratio", "opt", "ok", "ts", "name", "kind") for fn in fns]
    for f in (None, L("score", "gt", 25), L("score", "gt", 9999)):
        q = {"filters": f, "quick": ""}
        _agree(pair, "aggregates", lambda e: e.aggregates(q, aggs), lambda e: e.aggregates(q, aggs))

    def ordered(groups):
        return sorted(groups, key=lambda g: (-g["level"], json.dumps(g["keys"], default=str)))
    ag = [{"id": "c", "col": "score", "fn": "count"}, {"id": "s", "col": "ratio", "fn": "sum"},
          {"id": "a", "col": "score", "fn": "avg"}]
    for gb in (["kind"], ["kind", "ok"], ["name", "kind"], ["ok"], ["opt"], ["ts"]):
        for f in (None, L("score", "gt", 25)):
            q = {"filters": f}
            _agree(pair, f"group {gb}", lambda e: ordered(e.grouped_aggregates(q, gb, ag)),
                   lambda e: ordered(e.grouped_aggregates(q, gb, ag)))
    for col in ("name", "score", "ratio", "opt", "ok", "ts", "kind"):
        for extra in ({}, {"buckets": 7}, {"strategy": "quantile"}, {"strategy": "log"}, {"granularity": "month"}):
            for f in (None, L("score", "gt", 25)):
                r = {"colId": col, "filters": f, **extra}
                _agree(pair, f"facet {col} {extra}", lambda e: e.facet(r), lambda e: e.facet(r))
        for f in (None, L("score", "gt", 25)):
            _agree(pair, f"profile {col}", lambda e: e.profile(col, f, ""), lambda e: e.profile(col, f, ""))


def test_the_lazy_view_equals_the_arrow_path(pair):
    arrow, engines, _ = pair
    st = {"filters": L("score", "gt", 20), "quick": "a", "sort": [{"col": "name"}, {"col": "score", "dir": "desc"}]}
    from lattice_grid_jupyter._view import LazyView
    want = LazyView(arrow, st)
    for e in engines.values():
        got = LazyView(e, st)
        assert len(got) == len(want)
        assert got.head(12)["score"].tolist() == want.head(12)["score"].tolist()
        assert got.slice(40, 52)["name"].tolist() == want.slice(40, 52)["name"].tolist()
        assert got.to_pandas().shape == want.to_pandas().shape
        assert got.positions().tolist() == want.positions().tolist()


# --- the widget: every input route ----------------------------------------------------
def test_widget_inputs_and_errors(pair):
    arrow, engines, path = pair
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM read_parquet(?)", [str(path)])
    kw = dict(windowed=True)
    for w in (LatticeGridWidget(con.table("t"), **kw), LatticeGridWidget(con.sql("SELECT * FROM t WHERE score > 5"), **kw),
              LatticeGridWidget(con=con, table="t", **kw), LatticeGridWidget(con=con, sql="SELECT * FROM t", **kw)):
        assert w.windowed and w._engine.name == "duckdb" and w._mode == "windowed"
        assert {c["field"] for c in w._columns} >= {"name", "score", "ts"}
        assert req(w, "execute", query={"range": {"start": 0, "end": 5}})["total"] > 0
        with pytest.raises(AttributeError):
            w.df
    # small relations stay in the browser unless windowed
    assert LatticeGridWidget(con.table("t"))._mode == "client"
    with pytest.raises(ValueError):
        LatticeGridWidget(con=con)
    with pytest.raises(ValueError):
        LatticeGridWidget(con=con, table="t", sql="SELECT 1")
    with pytest.raises(TypeError):
        LatticeGridWidget()
    w = LatticeGridWidget(con=con, table="t", windowed=True)
    w.set_data(con=con, sql="SELECT * FROM t WHERE score < 10")
    assert req(w, "execute", query={})["total"] == con.execute("SELECT count(*) FROM t WHERE score < 10").fetchone()[0]


def test_parquet_pushdown_is_the_same_route(pair):
    arrow, engines, path = pair
    plain = LatticeGridWidget(str(path), windowed=True)
    fast = LatticeGridWidget(str(path), windowed=True, pushdown=True)
    assert plain._engine.name == "arrow" and fast._engine.name == "duckdb"
    q = {"filters": L("name", "contains", "a"), "sort": [{"col": "score", "dir": "desc"}], "range": {"start": 0, "end": 20}}
    assert _norm(plain._engine.execute(q)) == _norm(fast._engine.execute(q))
    assert any("WHERE" in s for s, _, _ in fast._engine.sql_log)
    assert _same(_norm(plain._engine.profile("ratio")), _norm(fast._engine.profile("ratio")))
    with pytest.raises(FileNotFoundError):
        LatticeGridWidget("/no/such/file.parquet", windowed=True, pushdown=True)
    with pytest.raises(TypeError):
        LatticeGridWidget(12345, pushdown=True, windowed=True)


# --- a notebook, Run All --------------------------------------------------------------
def test_notebook_using_a_relation_runs_all_headless():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    cells = [
        """
        import duckdb
        from lattice_grid_jupyter import LatticeGridWidget
        con = duckdb.connect()
        rel = con.sql("SELECT i, 'row ' || i AS label, i % 5 AS g, (i * 37 % 1000) / 10.0 AS val FROM range(200000) t(i)")
        w = LatticeGridWidget(rel)
        assert w.windowed and w._engine.name == "duckdb"
        w.state = {"version": 2, "filters": {"col": "g", "op": "eq", "value": 2},
                   "sort": [{"col": "val", "dir": "desc"}]}
        """,
        """
        v = w.view
        assert len(v) == 40000, len(v)
        top = v.head(3)
        assert (top["g"] == 2).all() and top["val"].is_monotonic_decreasing
        assert w._engine.sql_log, "answered by SQL"
        """,
        """
        w2 = LatticeGridWidget(con=con, sql="SELECT * FROM range(10) t(n)", windowed=True)
        assert w2.view.to_pandas()["n"].tolist() == list(range(10))
        """,
    ]
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(textwrap.dedent(c)) for c in cells]
    nbclient.NotebookClient(nb, kernel_name="python3", timeout=120).execute()
