# NO2 ~ RAIN on test.csv. Prints counts, means, and correlations.

import os
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILE = os.path.join(BASE, "data", "processed", "train.csv")  # change to train.csv if you want

df = pd.read_csv(FILE, parse_dates=["DATE"]).sort_values("DATE")

# RAIN is already boolean; NO2 is numeric in your pipeline
no2  = df["NO2"]
rain = df["RAIN"]

# Counts and share
counts = rain.value_counts().reindex([False, True]).fillna(0).astype(int)
total  = int(counts.sum())
share_rain = counts.get(True, 0) / total if total else float("nan")

print("EDA RAIN:")
print(f"Days without rain: {counts.get(False, 0)}")
print(f"Days with rain:    {counts.get(True,  0)}")
print(f"Share of rainy days: {share_rain:.3f}")

# Mean NO2 by rain
means = df.groupby("RAIN")["NO2"].mean()
print("\nMean NO₂ by rain:")
for flag, label in [(False, "No rain"), (True, "Rain")]:
    if flag in means.index:
        print(f"  {label}: {means.loc[flag]:.2f}")

# Correlations (encode boolean as 0/1 just for the calculation)
pearson  = pd.DataFrame({"NO2": no2, "RAIN": rain.astype(int)}).corr().loc["NO2", "RAIN"]
spearman = pd.DataFrame({"NO2": no2, "RAIN": rain.astype(int)}).corr(method="spearman").loc["NO2", "RAIN"]

print("\nCorrelation NO₂ ~ RAIN (RAIN=0/1):")
print(f"  Pearson (point-biserial): {pearson:.3f}")
print(f"  Spearman:                 {spearman:.3f}")

