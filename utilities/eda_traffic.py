# EDA for TRANSITS: summary stats, correlation with NO2, seaborn plots (hist + scatterplot).

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
s   = pd.to_numeric(df["TRANSITS"], errors="coerce")
no2 = pd.to_numeric(df["NO2"],      errors="coerce")

# Stats
mean   = s.mean();  median = s.median();  std = s.std()
q1     = s.quantile(0.25);  q3 = s.quantile(0.75);  iqr = q3 - q1
minv   = s.min();   maxv   = s.max()
min_day = df.loc[s.idxmin(), "DATE"].date() if s.notna().any() else "n/a"
max_day = df.loc[s.idxmax(), "DATE"].date() if s.notna().any() else "n/a"

print("TRANSITS EDA:")
print(f"mean:   {mean:.2f}")
print(f"median: {median:.2f}")
print(f"std:    {std:.2f}")
print(f"min:    {minv:.0f} on {min_day}")
print(f"max:    {maxv:.0f} on {max_day}")
print(f"Q1/Q3:  {q1:.0f} / {q3:.0f}  (IQR={iqr:.0f})")

# Correlation with NO2
pearson  = pd.DataFrame({"NO2": no2, "TRANSITS": s}).corr().loc["NO2", "TRANSITS"]
spearman = pd.DataFrame({"NO2": no2, "TRANSITS": s}).corr(method="spearman").loc["NO2", "TRANSITS"]
print("\nCorrelation with NO₂:")
print(f"  Pearson:  {pearson:.3f}")
print(f"  Spearman: {spearman:.3f}")

# Plots: histogram + scatterplot (side by side)
fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# Left: histogram + KDE + mean/median
sns.histplot(s.dropna(), bins="auto", kde=True, ax=ax[0])
ax[0].axvline(mean,   ls="--", lw=2, color="tab:orange", label="mean")
ax[0].axvline(median, ls="-.", lw=2, color="tab:green",  label="median")
ax[0].set_title("Transits — histogram")
ax[0].set_xlabel("Transits"); ax[0].set_ylabel("Count")
ax[0].legend()

# Right: scatterplot Transits vs NO2
sns.scatterplot(x=s, y=no2, ax=ax[1], alpha=0.6)
ax[1].set_title("NO2 vs Transits")
ax[1].set_xlabel("Transits")
ax[1].set_ylabel("NO2 (µg/m³)")

plt.show()

