"""Generate the sample sales CSV. Seeded, so the file is reproducible.

    python data/make_data.py
"""
import numpy as np
import pandas as pd

SEED = 7
N_ROWS = 400

REGIONS = ["North", "South", "East", "West"]
CATEGORIES = {
    "Electronics": ["Laptop", "Headphones", "Monitor"],
    "Furniture": ["Desk", "Chair", "Bookshelf"],
    "Apparel": ["Jacket", "Sneakers", "T-Shirt"],
    "Grocery": ["Coffee", "Olive Oil", "Granola"],
}
REPS = ["Alvarez", "Brennan", "Chen", "Dubois", "Ibrahim", "Novak"]
BASE_PRICE = {
    "Laptop": 1200.0, "Headphones": 180.0, "Monitor": 320.0,
    "Desk": 450.0, "Chair": 260.0, "Bookshelf": 140.0,
    "Jacket": 130.0, "Sneakers": 95.0, "T-Shirt": 25.0,
    "Coffee": 18.0, "Olive Oil": 22.0, "Granola": 9.0,
}


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    days = pd.date_range("2024-01-01", "2024-06-30", freq="D")

    categories = rng.choice(list(CATEGORIES), size=N_ROWS, p=[0.3, 0.25, 0.25, 0.2])
    products = [rng.choice(CATEGORIES[c]) for c in categories]

    df = pd.DataFrame({
        "order_id": [f"ORD-{i:05d}" for i in range(1, N_ROWS + 1)],
        "date": rng.choice(days, size=N_ROWS),
        "region": rng.choice(REGIONS, size=N_ROWS, p=[0.3, 0.25, 0.25, 0.2]),
        "category": categories,
        "product": products,
        "sales_rep": rng.choice(REPS, size=N_ROWS),
        "units": rng.integers(1, 12, size=N_ROWS),
    })

    # Price jitters +/-10% around the product's base price.
    jitter = rng.normal(1.0, 0.06, size=N_ROWS).clip(0.9, 1.1)
    df["unit_price"] = [round(BASE_PRICE[p] * j, 2) for p, j in zip(df["product"], jitter)]
    df["revenue"] = (df["units"] * df["unit_price"]).round(2)

    # A few returns, so the data isn't uniformly clean.
    df["returned"] = rng.random(N_ROWS) < 0.05

    return df.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    out = build()
    out.to_csv("data/sales.csv", index=False, date_format="%Y-%m-%d")
    print(f"wrote data/sales.csv  ({len(out)} rows, {out['date'].min():%Y-%m-%d} to {out['date'].max():%Y-%m-%d})")
