# Read raw traffic CSV, standardize date, coerce to integers, keep DATE/TRANSITS, save and preview.

import os
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IN_FILE  = os.path.join(BASE, "data", "raw", "traffic_csv", "traffic_milan_2019_2023.csv")
OUT_PATH = os.path.join(BASE, "data", "preprocessed", "preprocessed_traffic.csv")

# 1) Read raw (semicolon-separated)
t = pd.read_csv(IN_FILE, sep=";")

# 2) Keep only the YYYY-MM-DD part of the timestamp
t["date"] = t["data_giorno"].astype(str).str.slice(0, 10)

# 3) Coerce transits to numeric
t["transits"] = pd.to_numeric(t["numero_transiti_giornalieri"], errors="coerce")

# 4) One row per day (keep the last if duplicates), filter from 2020-01-01
t = t[["date", "transits"]].sort_values("date").drop_duplicates(subset="date", keep="last")
t = t[t["date"] >= "2020-01-01"].copy()

# 5) Force integers (nullable Int64 for safety)
t["transits"] = t["transits"].round().astype("Int64")

# 6) Keep only DATE/TRANSITS
out = t.rename(columns={"date": "DATE", "transits": "TRANSITS"})

# 7) Save
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
out.to_csv(OUT_PATH, index=False)

# Stats like NO2 utility
count_pos = int((out["TRANSITS"] > 0).sum())
print(f"Total observations: {len(out)}")
print(f"Observations > 0: {count_pos}")

# Preview
print("\nFirst 5 rows:")
print(out.head(5).to_string(index=False))
