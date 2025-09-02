# EDA for TEMPERATURE C: summary stats, correlation with NO2, seaborn plots (hist + boxplot).

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
temp = pd.to_numeric(df["TEMPERATURE C"], errors="coerce")
no2  = pd.to_numeric(df["NO2"],           errors="coerce")

# Stats
mean   = temp.mean();  median = temp.median();  std = temp.std()
q1     = temp.quantile(0.25);  q3 = temp.quantile(0.75);  iqr = q3 - q1
minv   = temp.min();   maxv   = temp.max()
min_day = df.loc[temp.idxmin(), "DATE"].date() if temp.notna().any() else "n/a"
max_day = df.loc[temp.idxmax(), "DATE"].date() if temp.notna().any() else "n/a"

print("EDA TEMPERATURE (°C): ")
print(f"mean:   {mean:.2f}°C")
print(f"median: {median:.2f}°C")
print(f"std:    {std:.2f}°C")
print(f"min:    {minv:.2f}°C on {min_day}")
print(f"max:    {maxv:.2f}°C on {max_day}")
print(f"Q1/Q3:  {q1:.2f}°C / {q3:.2f}°C  (IQR={iqr:.2f}°C)")

# Correlation with NO2
pearson  = pd.DataFrame({"NO2": no2, "TEMP": temp}).corr().loc["NO2", "TEMP"]
spearman = pd.DataFrame({"NO2": no2, "TEMP": temp}).corr(method="spearman").loc["NO2", "TEMP"]
print("\nCorrelation NO₂ ~ Temperature:")
print(f"  Pearson:  {pearson:.3f}")
print(f"  Spearman: {spearman:.3f}")

# Plots: histogram + boxplot (side by side)
fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# Left: histogram + KDE + mean/median
sns.histplot(temp.dropna(), bins="auto", kde=True, ax=ax[0])
ax[0].axvline(mean,   ls="--", lw=2, color="tab:orange", label="mean")
ax[0].axvline(median, ls="-.", lw=2, color="tab:green",  label="median")
ax[0].set_title("Temperature (°C) — histogram")
ax[0].set_xlabel("Temperature (°C)"); ax[0].set_ylabel("Count")
ax[0].legend()

# Right: boxplot
sns.boxplot(x=temp.dropna(), ax=ax[1], orient="h")
ax[1].set_title("Temperature (°C) — boxplot")
ax[1].set_xlabel("Temperature (°C)")
ax[1].set_ylabel("")

plt.show()
