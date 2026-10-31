"""One-time helper: builds car_options.json (brand -> car models) for the app.
Run from the project root:  python model/build_lookup.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "used_cars.csv")
df["price"] = df["price"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)

# same cleaning as the notebook: drop the top 1% price outliers
df = df[df["price"] <= df["price"].quantile(0.99)].copy()

# same rare-brand grouping as the notebook (<1% of rows -> "Other")
share = df["brand"].value_counts(normalize=True)
df["brand"] = df["brand"].replace(share[share < 0.01].index, "Other")

models_by_brand = {b: sorted(g["model"].unique().tolist()) for b, g in df.groupby("brand")}

out = Path(__file__).resolve().parent / "car_options.json"
out.write_text(json.dumps({"models_by_brand": models_by_brand}, indent=1))
print(f"Saved {out}  ({len(models_by_brand)} brands)")
