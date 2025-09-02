# utilities/eda_air.py
# EDA for NO2 (train.csv): summary stats + seaborn plots (hist + quantiles).

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_FILE = os.path.join(BASE, "data", "processed", "train.csv")

sns.set_theme(style="whitegrid")

# Load
df = pd.read_csv(TRAIN_FILE, parse_dates=["DATE"]).sort_values("DATE")

# Series
s = pd.to_numeric(df["NO2"], errors="coerce")

# Stats
mean   = s.mean();  median = s.median();  std = s.std()
q1     = s.quantile(0.25);  q3 = s.quantile(0.75);  iqr = q3 - q1
minv   = s.min();   maxv   = s.max()
min_day = df.loc[s.idxmin(), "DATE"].date() if s.notna().any() else "n/a"
max_day = df.loc[s.idxmax(), "DATE"].date() if s.notna().any() else "n/a"

print("NO2 EDA:")
print(f"mean:   {mean:.2f}")
print(f"median: {median:.2f}")
print(f"std:    {std:.2f}")
print(f"min:    {minv:.2f} on {min_day}")
print(f"max:    {maxv:.2f} on {max_day}")
print(f"Q1/Q3:  {q1:.2f} / {q3:.2f}  (IQR={iqr:.2f})")

# Plots: histogram + boxplot
fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# Left: histogram + KDE + mean/median lines
sns.histplot(s.dropna(), bins="auto", kde=True, ax=ax[0])
ax[0].axvline(mean,   ls="--", lw=2, color="tab:orange", label="mean")
ax[0].axvline(median, ls="-.", lw=2, color="tab:green",  label="median")
ax[0].set_title("NO2 — histogram")
ax[0].set_xlabel("NO2 (µg/m³)"); ax[0].set_ylabel("Count")
ax[0].legend()

# Right: boxplot (replace the quantiles-bar block with this)
sns.boxplot(x=s.dropna(), ax=ax[1], orient="h")
ax[1].set_title("NO2 — boxplot")
ax[1].set_xlabel("NO2 (µg/m³)")
ax[1].set_ylabel("")

plt.show()
