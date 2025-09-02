# EDA for WIND km/h: summary stats, correlation with NO2, seaborn plots (hist + boxplot).

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # solo per layout (subplots/show)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_FILE = os.path.join(BASE, "data", "processed", "train.csv")

sns.set_theme(style="whitegrid")

# Load & sort
df = pd.read_csv(TRAIN_FILE, parse_dates=["DATE"]).sort_values("DATE")

# Series
wind = pd.to_numeric(df["WIND km/h"], errors="coerce")
no2  = pd.to_numeric(df["NO2"],       errors="coerce")

# Stats
mean   = wind.mean();   median = wind.median();  std = wind.std()
q1     = wind.quantile(0.25);  q3 = wind.quantile(0.75);  iqr = q3 - q1
minv   = wind.min();    maxv   = wind.max()
min_day = df.loc[wind.idxmin(), "DATE"].date() if wind.notna().any() else "n/a"
max_day = df.loc[wind.idxmax(), "DATE"].date() if wind.notna().any() else "n/a"

print(" EDA WIND: ")
print(f"mean:   {mean:.2f} km/h")
print(f"median: {median:.2f} km/h")
print(f"std:    {std:.2f} km/h")
print(f"min:    {minv:.2f} km/h on {min_day}")
print(f"max:    {maxv:.2f} km/h on {max_day}")
print(f"Q1/Q3:  {q1:.2f} / {q3:.2f} km/h  (IQR={iqr:.2f} km/h)")

# Correlation with NO2
pearson  = pd.DataFrame({"NO2": no2, "WIND": wind}).corr().loc["NO2", "WIND"]
spearman = pd.DataFrame({"NO2": no2, "WIND": wind}).corr(method="spearman").loc["NO2", "WIND"]
print("\nCorrelation NO₂ ~ Wind:")
print(f"  Pearson:  {pearson:.3f}")
print(f"  Spearman: {spearman:.3f}")

# Plots: histogram + boxplot (side by side)
fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# Left: histogram + KDE + mean/median (25 bins)
sns.histplot(wind.dropna(), bins=25, kde=True, ax=ax[0])
ax[0].axvline(mean,   ls="--", lw=2, color="tab:orange", label="mean")
ax[0].axvline(median, ls="-.", lw=2, color="tab:green",  label="median")
ax[0].set_title("Wind (km/h) — histogram")
ax[0].set_xlabel("Wind (km/h)"); ax[0].set_ylabel("Count")
ax[0].legend()

# Right: boxplot
sns.boxplot(x=wind.dropna(), ax=ax[1], orient="h")
ax[1].set_title("Wind (km/h) — boxplot")
ax[1].set_xlabel("Wind (km/h)")
ax[1].set_ylabel("")

plt.show()
