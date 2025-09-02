# Concatenate air CSVs, filter to station 7 & NO2, sort/drop duplicates by date, and save only DATE/NO2.

import os
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FILES = [
    os.path.join(BASE, "data", "raw", "air_csv", "air_milan_2020.csv"),
    os.path.join(BASE, "data", "raw", "air_csv", "air_milan_2021.csv"),
    os.path.join(BASE, "data", "raw", "air_csv", "air_milan_2022.csv"),
    os.path.join(BASE, "data", "raw", "air_csv", "air_milan_2023.csv"),
]

OUT_PATH = os.path.join(BASE, "data", "preprocessed", "preprocessed_air.csv")

STATION_ID = 7
POLLUTANT  = "NO2"

# 1) Read each CSV (semicolon-separated)
dfs = [pd.read_csv(f, sep=";") for f in FILES]

# 2) Concatenate into a single DataFrame
big = pd.concat(dfs, ignore_index=True)

# 3) Ensure station_id is integer-like
big["stazione_id"] = pd.to_numeric(big["stazione_id"], errors="coerce").astype("Int64")

# 4) Filter by station and pollutant
mask = (big["stazione_id"] == STATION_ID) & (big["inquinante"] == POLLUTANT)
big = big.loc[mask].copy()

# 5) Keep only DATE and NO2, then sort and drop duplicate dates (keep last)
out = big[["data", "valore"]].rename(columns={"data": "DATE", "valore": "NO2"})
out = out.sort_values("DATE").drop_duplicates(subset="DATE", keep="last")

# 6) For safety, coerce NO2 to numeric before computing the count
out["NO2"] = pd.to_numeric(out["NO2"], errors="coerce")

# 7) Stats
count_pos = int((out["NO2"] > 0).sum())

# 8) Save
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
out.to_csv(OUT_PATH, index=False)

print(f"Total observations: {len(out)}")
print(f"Observations > 0: {count_pos}")

# 9) Preview
print("\nFirst 5 rows:")
print(out.head(5).to_string(index=False))

