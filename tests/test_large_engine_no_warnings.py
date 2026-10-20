"""BACKLOG-0004700: large-mode filters must not raise pandas's object-dtype
``FutureWarning`` (``.fillna`` silently downcasting a boolean mask) -- and must
keep matching the same rows as the warning-free oracle below.

Run with ``pytest -W error::FutureWarning`` so the suite also fails if some
other call path starts downcasting again; the ``catch_warnings`` block below
makes each case self-contained regardless of the invocation flags.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from lattice_grid_jupyter import LatticeGridWidget
from lattice_grid_jupyter._engine import FACTORIZE_ABOVE

from test_large_engine import keys_of, req


def large_frame(n=2_000, seed=11):
    """Above ``FACTORIZE_ABOVE`` with nulls in every column kind a filter runs over."""
    r = np.random.default_rng(seed)
    names = np.array(["Ada", "grace", "Linus", "ken", "Barbara", "alan", None], dtype=object)
    df = pd.DataFrame({
        "name": names[r.integers(0, len(names), n)],
        "score": r.integers(0, 50, n).astype("float64"),
    })
    df.loc[r.random(n) < 0.1, "score"] = np.nan
    assert n > FACTORIZE_ABOVE
    assert df["name"].isna().any() and df["score"].isna().any()
    return df


@pytest.mark.parametrize("cond,expect", [
    ({"col": "name", "op": "contains", "value": "ar"},
     lambda d: d["name"].str.lower().str.contains("ar", na=False)),
    ({"col": "name", "op": "notContains", "value": "ar"},
     lambda d: ~d["name"].str.lower().str.contains("ar", na=False)),
    ({"col": "name", "op": "eq", "value": "ada"},
     lambda d: (d["name"].str.lower() == "ada") & d["name"].notna()),
    ({"col": "name", "op": "ne", "value": "ada"},
     lambda d: ~((d["name"].str.lower() == "ada") & d["name"].notna())),
    ({"col": "score", "op": "gt", "value": 25},
     lambda d: d["score"] > 25),
    ({"col": "score", "op": "between", "value": [10, 30]},
     lambda d: d["score"].between(10, 30)),
])
def test_filter_raises_no_future_warning_and_matches_oracle(cond, expect):
    df = large_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        w = LatticeGridWidget(df, windowed=True)
        res = req(w, "execute", query={"filters": cond, "range": {"start": 0, "end": len(df)}})

    want = df.index[expect(df)].tolist()
    assert keys_of(res) == want
    assert res["total"] == len(want)
