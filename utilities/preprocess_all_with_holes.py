# Merge preprocessed air/traffic/weather on DATE (outer join), save, preview, and plot missingness.

import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

AIR_FILE     = os.path.join(BASE, "data", "preprocessed", "preprocessed_air.csv")       # DATE, NO2
TRAFFIC_FILE = os.path.join(BASE, "data", "preprocessed", "preprocessed_traffic.csv")   # DATE, TRANSITS
WEATHER_FILE = os.path.join(BASE, "data", "preprocessed", "preprocessed_weather.csv")   # DATE, TEMPERATURE C, WIND km/h, RAIN

OUT_FILE     = os.path.join(BASE, "data", "preprocessed", "preprocessed_all_with_holes.csv")

# 1) Load preprocessed inputs
air     = pd.read_csv(AIR_FILE)      # expects: DATE, NO2
traffic = pd.read_csv(TRAFFIC_FILE)  # expects: DATE, TRANSITS
weather = pd.read_csv(WEATHER_FILE)  # expects: DATE, TEMPERATURE C, WIND km/h, RAIN

# 2) Outer-join on DATE and sort
out = (
    air.merge(traffic, on="DATE", how="outer")
       .merge(weather, on="DATE", how="outer")
       .sort_values("DATE")
)

# 3) Save merged table
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
out.to_csv(OUT_FILE, index=False)

print("\nFirst 5 rows:")
print(out.head(5).to_string(index=False))
print("\nLast 5 rows:")
print(out.tail(5).to_string(index=False))

# 4) Missingness matrix (1 = missing, 0 = present)
df = out.copy()
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.sort_values("DATE").set_index("DATE")

cols = ["NO2", "TRANSITS", "TEMPERATURE C", "WIND km/h", "RAIN"]
mask = df[cols].isna().astype(int)

plt.figure(figsize=(12, 4))
plt.imshow(mask.T, aspect="auto", interpolation="nearest", vmin=0, vmax=1)  # <— scala 0–1
plt.yticks(range(len(cols)), cols)

step = max(len(mask) // 10, 1)
xticks = range(0, len(mask), step)
xlabels = df.index.strftime("%Y-%m-%d").to_list()[::step]
plt.xticks(xticks, xlabels, rotation=45, ha="right")

plt.title("Missingness matrix (1 = missing, 0 = present)")
plt.xlabel("Date"); plt.ylabel("Feature")

cbar = plt.colorbar()
cbar.set_ticks([0, 1])                              # <— solo 0 e 1
cbar.set_ticklabels(["present (0)", "missing (1)"])

plt.tight_layout()
plt.show()

