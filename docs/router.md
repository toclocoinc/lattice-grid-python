# The data router in the notebook

Card 1620. `LatticeRouter` declares several DataFrames as **sources** of the grid's
own data router, with the router's own join, rollup, spread, unnest and
field-coercion specs written as plain dicts. `LatticeGridWidget(router, route=...)`
shows the **shaped rows** of one route, and `widget.view` hands them back to Python
as a DataFrame, so a feature table can go straight into a model.

The router **runs in the browser, on the grid's own `data-router` module**. Python
declares and ships the data; it does not reimplement a single join semantic. What a
`missing: "hold"` does, how `avg` of nothing reads, how a spread field collides: that is
the module's behaviour, documented in the grid's API reference.

```python
router = LatticeRouter(
    sources={"customer": {...}, "txn": txns, "label": {...}},   # name -> DataFrame, or a spec dict
    routes={"customer": "customer"},                            # route name -> source (default: one per source)
)
w = LatticeGridWidget(router, route="customer")                 # route= may be omitted with one route
```

## A source

A source is a DataFrame, or a dict:

| key | meaning |
|---|---|
| `data` | the DataFrame |
| `key` | the key column (or the index name). Default: a column `id`, else `<name>_id`. Must be unique |
| `join` | one join spec or a list, applied in order (lookup, collect, rollup onto the parent, spread) |
| `unnest` | one spec or a list: expand a nested list column into its own row type |
| `fields` | coerce raw values: `{"col": "number" \| "integer" \| "boolean" \| "date" \| "json" \| "text" \| {...}}` |

* **Plain data.** `snake_case` keys are mapped (`foreign_key` -> `foreignKey`). A key the
  router does not know is reported by name as a `LatticeGridWarning`
  (`[lattice] router.unknown:sources.customer.join.foreing_key: ...`) and left out. A
  function (`map`, `where`, a function `select`) raises `TypeError`: it cannot cross to
  the browser.
* **Join kinds** (all the module's):
  * lookup: `{"from": "label", "local_key": "customer_id", "fields": ["churned"], "missing": "null"}`
    (`fields` may rename: `{"name": "owner"}`);
  * collect: `{"from": "txn", "many": True, "foreign_key": "customer_id", "as": "amounts", "select": "amount", "distinct": True, "sort": True}`
    (`select` is a column name, or a list of names for an object per child);
  * rollup onto the parent: `{"from": "txn", "many": True, "foreign_key": "customer_id", "aggregate": {"spend": {"fn": "sum", "field": "amount"}}}`
    (`count`, `sum`, `avg`, `min`, `max`, `distinctCount`, `first`, `last`);
  * spread: `{"from": "attr", "foreign_key": "cid", "spread": {"name": "k", "value": "v", "prefix": "cf_", "type": {"age": "number"}}}`
    (each attribute row becomes a column; a new attribute adds a column live).
  A `many`/`spread` join matches on this source's own key column unless you give `local_key`.
* **Unnest:** `{"path": "addresses", "as": "address", "parent_key": "company_id"}` and
  route it: `routes={"address": "address"}`. The list column travels as data.

## Updating a source: a keyed diff

```python
router.update("txn", new_txns)      # {"added": 1, "updated": 1, "removed": 1}
```

Python compares the new frame with the old by key and sends the browser **only the
difference** (upserts and deleted keys). The router re-emits only the parents the diff
reaches, and the grid repaints those rows only. `widget.router_stats` counts what the
route has emitted (`batches`, `added`, `updated`, `removed`, cumulative): take the
difference around an `update`. A grid shown later reads the current data.

## The shaped rows in Python

* `widget.view` is the route's shaped rows that pass the grid's filters, in the grid's order,
  as a DataFrame; `widget.df` is all of them; `widget.selected` the selected ones. Numbers
  and booleans arrive typed; a `date` field arrives as `datetime64`.
* The shaped frame is reported by the browser (debounced), so read it after the grid has
  shown the rows, in a later cell, as with `selected`.
* Shaped rows are derived, so the grid is read-only and `set_data`, `append_rows` and
  `delete_rows` raise: change a source with `router.update`.
* Per-column options go in `columns=[...]` as for any grid (matched on field); the columns
  themselves are inferred from the shaped rows.

## Recipe: a feature table from customers, transactions and labels

The fixture is seeded and synthetic. Every block runs as written.

```python
import numpy as np, pandas as pd
from lattice_grid_jupyter import LatticeGridWidget, LatticeRouter

rng = np.random.default_rng(0)
n = 40
customers = pd.DataFrame({
    "customer_id": [f"C{i:03d}" for i in range(n)],
    "plan": rng.choice(["free", "pro", "team"], n),
    "signup": pd.date_range("2023-01-01", periods=n, freq="7D").strftime("%d/%m/%Y"),
})
txns = pd.DataFrame([
    {"txn_id": f"T{i:03d}-{j}", "customer_id": f"C{i:03d}", "amount": round(float(rng.gamma(2.0, 30.0)), 2)}
    for i in range(n) for j in range(int(rng.integers(0, 6)))
])
labels = pd.DataFrame({"customer_id": customers.customer_id, "churned": rng.random(n) < 0.3})
```

One router: the customer rows, each rolled up from its transactions and enriched with its label.

```python
router = LatticeRouter(
    sources={
        "customer": {
            "data": customers, "key": "customer_id",
            "fields": {"signup": {"type": "date", "format": "dd/MM/yyyy"}},
            "join": [
                {"from": "txn", "many": True, "foreign_key": "customer_id",
                 "aggregate": {"n_txn": {"fn": "count"},
                               "spend": {"fn": "sum", "field": "amount"},
                               "biggest": {"fn": "max", "field": "amount"}}},
                {"from": "label", "local_key": "customer_id", "foreign_key": "customer_id",
                 "fields": ["churned"], "missing": "null"},
            ],
        },
        "txn": {"data": txns, "key": "txn_id"},
        "label": {"data": labels, "key": "customer_id"},
    },
    routes={"customer": "customer"},
)
w = LatticeGridWidget(router, route="customer", offline=True)
w          # the feature table, one row per customer
```

Later cells: the shaped frame is a DataFrame, ready for a model.

```python
features = w.view                    # honours the grid's filters and sort
X = features[["n_txn", "spend", "biggest"]].fillna(0).to_numpy(float)
y = features["churned"].astype(float).to_numpy()
```

When transactions change, send the difference. Only the customers it reaches re-emit:

```python
new_txns = txns.copy()
new_txns.loc[new_txns.index[0], "amount"] += 50.0
router.update("txn", new_txns)       # {'added': 0, 'updated': 1, 'removed': 0}
```
