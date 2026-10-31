import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Used Car Valuation", page_icon="🚗")

BASE = Path(__file__).parent

# With drop_first=True, the alphabetically-first category of each column has no
# dummy column. It is the "baseline": all dummies = 0 means this value.
BASELINES = {
    "brand_": "Acura",
    "fuel_type_": "Diesel",
    "transmission_": "1-Speed A/T",
    "ext_col_": "Black",
    "int_col_": "Beige",
    "accident_": "At least 1 accident or damage reported",
}
# categories that exist in the training data but are junk values, not real choices
JUNK = {"not supported", "–"}

BRAND_LABELS = {
    "Land": "Land Rover",
    "Other": "Other (Ferrari, Bentley, Rolls-Royce, Maserati, ...)",
}


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE / "car_price_model.pkl")
    columns = joblib.load(BASE / "model_columns.pkl")
    model_prices = joblib.load(BASE / "model_mean_prices.pkl")
    with open(BASE / "car_options.json") as f:
        models_by_brand = json.load(f)["models_by_brand"]
    return model, columns, model_prices, models_by_brand


rf_model, columns, model_prices, models_by_brand = load_artifacts()


def choices(prefix):
    """Every value the model really knows for one categorical feature."""
    found = [c[len(prefix):] for c in columns if c.startswith(prefix)]
    return sorted(v for v in found + [BASELINES[prefix]] if v not in JUNK)


def build_features(year, miles, brand, model_name, fuel, trans, ext, int_, accident_free):
    """One row with exactly the columns the model was trained on."""
    row = pd.Series(0.0, index=columns)
    row["model_year"] = year
    row["milage"] = miles
    row["model_encoded"] = model_prices[model_name]
    accident = "None reported" if accident_free else "At least 1 accident or damage reported"
    for prefix, value in [
        ("brand_", brand), ("fuel_type_", fuel), ("transmission_", trans),
        ("ext_col_", ext), ("int_col_", int_), ("accident_", accident),
    ]:
        if prefix + value in columns:  # baseline value has no column -> stays 0
            row[prefix + value] = 1.0
    return pd.DataFrame([row])


# ---------------- UI ----------------
st.title("🚗 Used Car Price Predictor")
st.write("Enter the car's details to get an estimated market price from a Random Forest model.")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", choices("brand_"), format_func=lambda b: BRAND_LABELS.get(b, b))
    available = [m for m in models_by_brand.get(brand, []) if m in model_prices] or sorted(model_prices)
    model_name = st.selectbox("Model", available)
    year = st.slider("Model year", 1990, 2024, 2018)
    miles = st.number_input("Mileage (miles)", 0, 300_000, 50_000, step=1_000)

with col2:
    fuel = st.selectbox("Fuel type", choices("fuel_type_"), index=choices("fuel_type_").index("Gasoline"))
    trans = st.selectbox("Transmission", choices("transmission_"), index=choices("transmission_").index("A/T"))
    ext = st.selectbox("Exterior color", choices("ext_col_"))
    int_ = st.selectbox("Interior color", choices("int_col_"))
    accident_free = st.radio("Accident history", ["No accidents reported", "Accident or damage reported"]) \
        == "No accidents reported"

if st.button("Estimate price", type="primary"):
    X = build_features(year, miles, brand, model_name, fuel, trans, ext, int_, accident_free)
    price = rf_model.predict(X)[0]
    st.success(f"### Estimated price: ${price:,.0f}")
    st.caption("Trained on about 4,000 US used-car listings. Real prices vary with trim, condition and location.")