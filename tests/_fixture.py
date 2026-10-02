"""A seeded, synthetic customers / transactions / labels fixture (card 1620)."""

import numpy as np
import pandas as pd


def customers_txns_labels(n_customers: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    customers = pd.DataFrame({
        "customer_id": [f"C{i:03d}" for i in range(n_customers)],
        "name": [f"Customer {i}" for i in range(n_customers)],
        "signup": pd.date_range("2023-01-01", periods=n_customers, freq="7D").strftime("%d/%m/%Y"),
        "plan": rng.choice(["free", "pro", "team"], n_customers),
    })
    rows = []
    for i, cid in enumerate(customers.customer_id):
        for j in range(int(rng.integers(0, 6))):
            rows.append({"txn_id": f"T{i:03d}-{j}", "customer_id": cid,
                         "amount": round(float(rng.gamma(2.0, 30.0)), 2),
                         "ts": f"2024-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 28)):02d}"})
    txns = pd.DataFrame(rows)
    labels = pd.DataFrame({
        "customer_id": customers.customer_id,
        "churned": rng.random(n_customers) < 0.3,
    })
    return customers, txns, labels
