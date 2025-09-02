# Plot IQR outliers over time for NO2, TRANSITS, and TEMPERATURE

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths relative to the project root (works from notebooks/)
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_FILE = os.path.join(BASE, "data", "processed", "train.csv")

# Load and sort chronologically
df = pd.read_csv(TRAIN_FILE, parse_dates=["DATE"]).sort_values("DATE")
date = df["DATE"]

def iqr_plot(series, name):
    s = pd.to_numeric(series, errors="coerce")
    s_valid = s.dropna()
    if s_valid.empty:
        print(f"[{name}] no numeric data → skipping.")
        return
    q1, q3 = s_valid.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    is_out = (s < lo) | (s > hi)
    print(f"{name}: bounds [{lo:.2f}, {hi:.2f}] | outliers {int(is_out.sum())}/{int(s.notna().sum())}")

    plt.figure(figsize=(10, 4))
    plt.scatter(date[s.notna() & ~is_out], s[s.notna() & ~is_out], s=10)
    plt.scatter(date[s.notna() &  is_out], s[s.notna() &  is_out], s=30, marker="x")
    plt.axhline(lo, linestyle="--"); plt.axhline(hi, linestyle="--")
    plt.title(f"{name} — IQR outliers")
    plt.xlabel("Date"); plt.ylabel(name)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

# Run plots
iqr_plot(df["NO2"],            "NO2 (µg/m³)")
iqr_plot(df["TRANSITS"],       "Transits (cars/day)")
iqr_plot(df["TEMPERATURE C"],  "Temperature (°C)")
iqr_plot(df["WIND km/h"],      "Wind (km/h)")
