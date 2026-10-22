"""BACKLOG-0005304: grouping over a DuckDB relation answers as SQL; ``router_stats`` needs no browser.

* ``grouplevel`` over a DuckDB source (levels, child windows, aggregates incl. median/percentile,
  sort by key or aggregate, filters) equals the Arrow path and runs as bound SQL;
* a generated 1,000,000-row table: grouped counts/aggregates equal a plain SQL GROUP BY;
* a notebook grouping a relation, and reading ``router_stats``, runs under ``Run All`` (nbclient).
"""

import json
import textwrap

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from lattice_grid_jupyter import LatticeGridWidget
from lattice_grid_jupyter._arrow_engine import ArrowEngine
from lattice_grid_jupyter._duckdb_engine import DuckDBEngine
from test_duckdb_pushdown import _norm, _same, _sample, req

AGGS = [{"id": "n", "col": "score", "fn": "count"}, {"id": "s", "col": "ratio", "fn": "sum"},
        {"id": "m", "col": "score", "fn": "median"}, {"id": "p", "col": "ratio", "fn": "p90"},
        {"id": "d", "col": "name", "fn": "distinct"}]


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    path = tmp_path_factory.mktemp("duckg") / "sample.parquet"
    pq.write_table(pa.Table.from_pandas(_sample(), preserve_index=False), str(path))
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM read_parquet(?)", [str(path)])
    return ArrowEngine(str(path)), {"table": DuckDBEngine(con=con, table="t"),
                                    "relation": DuckDBEngine(relation=con.table("t"))}


SORTS = [[], [{"col": "kind", "dir": "desc"}], [{"col": "s", "dir": "desc"}], [{"col": "m"}, {"col": "kind"}],
         [{"col": "p", "dir": "desc", "nullsFirst": True}], [{"col": "name"}], [{"col": "score", "dir": "desc"}]]


def _level(e, **kw):
    q = {"groupBy": ["kind", "ok"], "groupValues": [], "sort": [], "filters": None, "quick": "",
         "range": {"start": 0, "end": 100}}
    q.update(kw)
    return _norm(e.group_level(q, AGGS))


@pytest.mark.parametrize("sort", SORTS, ids=lambda s: json.dumps(s)[:40])
@pytest.mark.parametrize("flt", [None, {"col": "score", "op": "gt", "value": 25}])
def test_group_levels_equal_the_arrow_path(pair, sort, flt):
    arrow, engines = pair
    for gb, values in ((["kind", "ok"], []), (["kind", "ok"], ["b"]), (["kind", "ok"], ["b", True]),
                       (["name"], []), (["opt", "kind"], [None]), (["ts"], [])):
        if len(values) >= len(gb) and any(x["col"] in {a["id"] for a in AGGS} for x in sort):
            continue   # leaf rows sort by real columns only
        for rng in ({"start": 0, "end": 100}, {"start": 1, "end": 3}):
            want = _level(arrow, groupBy=gb, groupValues=values, sort=sort, filters=flt, range=rng)
            for name, e in engines.items():
                got = _level(e, groupBy=gb, groupValues=values, sort=sort, filters=flt, range=rng)
                assert _same(got, want), (name, gb, values, sort, got, want)


def test_a_datetime_group_opens_its_children(pair):
    arrow, engines = pair
    top = _level(arrow, groupBy=["ts", "kind"])
    key = top["rows"][1]["ts"]
    want = _level(arrow, groupBy=["ts", "kind"], groupValues=[key])
    assert want["rows"]
    for e in engines.values():
        assert _same(_level(e, groupBy=["ts", "kind"], groupValues=[key]), want)


def test_widget_request_is_answered_with_bound_sql(pair):
    _, engines = pair
    con = engines["table"]._con
    w = LatticeGridWidget(con.table("t"), state={"group": ["kind"]}, windowed=True)
    e = w._engine
    assert isinstance(e, DuckDBEngine)
    out = req(w, "grouplevel", query={"groupBy": ["kind"], "groupValues": [], "sort": [],
                                       "range": {"start": 0, "end": 10}}, aggregates=AGGS)
    assert [r["kind"] for r in out["rows"]] == ["a", "b", "c"]
    leaf = req(w, "grouplevel", query={"groupBy": ["kind"], "groupValues": ["b"], "sort": [{"col": "score"}],
                                        "range": {"start": 0, "end": 5}}, aggregates=AGGS)
    assert leaf["leaves"] and len(leaf["columnar"]["kind"]) == 5 and set(leaf["columnar"]["kind"]) == {"b"}
    assert e.sql_log and all("'b'" not in s for s, _, _ in e.sql_log)


def test_one_million_rows_group_like_sql():
    con = duckdb.connect()
    con.execute("""CREATE TABLE m AS SELECT i, 'c' || (i % 13) AS category, (i % 4) AS sub,
                   (hash(i) % 10000) / 10.0 AS val FROM range(1000000) t(i)""")
    w = LatticeGridWidget(con.table("m"), state={"group": ["category"]})
    aggs = [{"id": "n", "col": "val", "fn": "count"}, {"id": "s", "col": "val", "fn": "sum"},
            {"id": "md", "col": "val", "fn": "median"}, {"id": "p95", "col": "val", "fn": "p95"}]
    q = {"groupBy": ["category", "sub"], "groupValues": [], "sort": [{"col": "s", "dir": "desc"}],
         "range": {"start": 0, "end": 50}}
    out = req(w, "grouplevel", query=q, aggregates=aggs)
    want = con.execute("""SELECT category, count(*), count(val), sum(val), quantile_cont(val, .5),
                          quantile_cont(val, .95) FROM m GROUP BY 1 ORDER BY 4 DESC""").fetchall()
    assert out["total"] == 13 and out["matchCount"] == 1_000_000
    for r, (cat, cnt, n, s, md, p95) in zip(out["rows"], want):
        assert r["category"] == cat and r["leafCount"] == cnt
        assert r["totals"]["n"] == n and abs(r["totals"]["s"] - s) < 1e-6 * s
        assert abs(r["totals"]["md"] - md) < 1e-9 and abs(r["totals"]["p95"] - p95) < 1e-9
    top = out["rows"][0]["category"]
    # a child window loads, sorted and sized
    ch = req(w, "grouplevel", query={**q, "groupValues": [top], "sort": [{"col": "sub", "dir": "desc"}]},
             aggregates=aggs)
    assert [r["sub"] for r in ch["rows"]] == [3, 2, 1, 0] and ch["total"] == 4
    leaves = req(w, "grouplevel", query={**q, "groupValues": [top, 2], "sort": [{"col": "val", "dir": "desc"}],
                                         "range": {"start": 10, "end": 30}}, aggregates=aggs)
    exp = con.execute("SELECT val FROM m WHERE category = ? AND sub = 2 ORDER BY val DESC, i LIMIT 20 OFFSET 10",
                      [top]).fetchall()
    assert leaves["leaves"] and leaves["columnar"]["val"] == [x[0] for x in exp]
    assert leaves["total"] == con.execute("SELECT count(*) FROM m WHERE category = ? AND sub = 2", [top]).fetchone()[0]
    assert max(n for _, _, n in w._engine.sql_log) <= 1_000_000   # logged, and every statement is SQL
    assert any("GROUP BY" in s for s, _, _ in w._engine.sql_log)


def test_default_expanded_state_loads_in_the_widget():
    con = duckdb.connect()
    con.execute("CREATE TABLE g AS SELECT i, 'k' || (i % 3) AS k, i AS v FROM range(300) t(i)")
    w = LatticeGridWidget(con.table("g"), state={"group": ["k"], "groupDefaultExpanded": 1,
                                                  "sort": [{"col": "k", "dir": "desc"}]}, windowed=True)
    out = req(w, "grouplevel", query={"groupBy": ["k"], "groupValues": [], "sort": [{"col": "k", "dir": "desc"}],
                                       "range": {"start": 0, "end": 10}}, aggregates=[])
    assert [r["k"] for r in out["rows"]] == ["k2", "k1", "k0"] and [r["leafCount"] for r in out["rows"]] == [100] * 3


def test_router_stats_returns_its_keys_without_a_browser():
    w = LatticeGridWidget(pd.DataFrame({"a": [1, 2]}))
    assert w.router_stats == {"batches": 0, "added": 0, "updated": 0, "removed": 0}
    w._router_stats = {"batches": 2, "added": 5}
    assert w.router_stats == {"batches": 2, "added": 5, "updated": 0, "removed": 0}


def test_notebook_groups_a_relation_and_reads_router_stats_under_run_all():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    cells = [
        """
        import duckdb
        from lattice_grid_jupyter import LatticeGridWidget
        con = duckdb.connect()
        con.execute("CREATE TABLE products AS SELECT i, 'c' || (i % 5) AS category, i * 1.5 AS price FROM range(50000) t(i)")
        w = LatticeGridWidget(con.table('products'), state={'group': ['category']}, windowed=True)
        assert w._engine.name == "duckdb"
        """,
        """
        import json
        w._on_custom(w, {"type": "lg:req", "id": 1, "method": "grouplevel", "payload": {
            "query": {"groupBy": ["category"], "groupValues": [], "sort": [], "range": {"start": 0, "end": 10}},
            "aggregates": [{"id": "m", "col": "price", "fn": "median"}]}}, [])
        """,
        """
        sent = []
        w.send = lambda content, buffers=None: sent.append(content)
        w._on_custom(w, {"type": "lg:req", "id": 2, "method": "grouplevel", "payload": {
            "query": {"groupBy": ["category"], "groupValues": [], "sort": [], "range": {"start": 0, "end": 10}},
            "aggregates": [{"id": "m", "col": "price", "fn": "median"}]}}, [])
        res = sent[0]
        assert res["ok"], res
        assert [r["category"] for r in res["result"]["rows"]] == ["c0", "c1", "c2", "c3", "c4"]
        assert res["result"]["rows"][0]["leafCount"] == 10000
        """,
        """
        assert w.router_stats == {"batches": 0, "added": 0, "updated": 0, "removed": 0}
        """,
    ]
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(textwrap.dedent(c)) for c in cells]
    nbclient.NotebookClient(nb, kernel_name="python3", timeout=180).execute()
