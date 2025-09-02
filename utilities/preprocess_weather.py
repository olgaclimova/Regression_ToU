# Concatenate monthly weather CSVs (2020–2023), keep DATE / TEMPERATURE C / WIND km/h / RAIN,
# coerce numbers, sort & deduplicate, save, and print numeric counts.

import os
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_PATH = os.path.join(BASE, "data", "preprocessed", "preprocessed_weather.csv")

# 1) Read and concatenate Milano-YYYY-MM.csv (semicolon-separated)
frames = []
for year in range(2020, 2023 + 1):
    for month in range(1, 12 + 1):
        fname = os.path.join(BASE, "data", "raw", "meteo_csv", f"Milano-{year}-{month:02d}.csv")
        try:
            frames.append(pd.read_csv(fname, sep=";"))
        except FileNotFoundError:
            continue  # skip missing months

meteo = pd.concat(frames, ignore_index=True)

# 2) Keep only needed columns
meteo = meteo[["DATA", "TMEDIA °C", "VENTOMEDIA km/h", "FENOMENI"]]

# 3) Standardize date -> YYYY-MM-DD (rename to DATE)
meteo["DATE"] = pd.to_datetime(meteo["DATA"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")

# 4) Coerce numeric fields (comma decimals -> dot) for temperature and wind
meteo["TEMPERATURE C"] = pd.to_numeric(
    meteo["TMEDIA °C"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)
meteo["WIND km/h"] = pd.to_numeric(
    meteo["VENTOMEDIA km/h"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)

# 5) RAIN boolean: True if the text starts with "pioggia", else False
meteo["RAIN"] = meteo["FENOMENI"].astype(str).str.match(r"^\s*pioggia", case=False, na=False)

# 6) Final columns, sort & deduplicate by DATE (keep last)
out = meteo[["DATE", "TEMPERATURE C", "WIND km/h", "RAIN"]].sort_values("DATE").drop_duplicates(subset="DATE", keep="last")

# 7) Save
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
out.to_csv(OUT_PATH, index=False)

# 8) Stats (numeric counts, not NaNs)
total_rows = len(out)
temp_numeric = int(out["TEMPERATURE C"].notna().sum())
wind_numeric = int(out["WIND km/h"].notna().sum())

print(f"Total rows: {total_rows}")
print(f"Rows with numeric TEMPERATURE C: {temp_numeric}")
print(f"Rows with numeric WIND km/h:     {wind_numeric}")
print(f"Rows with non-boolean RAIN: {(~out['RAIN'].isin([True, False])).sum()}")

print("\nFirst 5 rows:")
print(out.head(5).to_string(index=False))


