# Draw a |r| correlation heatmap for the main features (train.csv or test.csv)

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Change here if you want to switch file
FILE = os.path.join(BASE, "data", "processed", "test.csv")
# FILE = os.path.join(BASE, "data", "processed", "train.csv")

sns.set_theme(style="whitegrid")

# Load
df = pd.read_csv(FILE, parse_dates=["DATE"]).sort_values("DATE")

# Pick relevant columns
cols = [c for c in ["NO2", "TRANSITS", "TEMPERATURE C", "WIND km/h", "RAIN"] if c in df.columns]
X = df[cols].copy()

# Ensure RAIN is numeric 0/1
if "RAIN" in X.columns and X["RAIN"].dtype == bool:
    X["RAIN"] = X["RAIN"].astype(int)

# Correlation matrix (Pearson, abs)
corr_abs = X.corr(method="pearson").abs()

# Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(corr_abs, annot=True, fmt=".2f", vmin=0, vmax=1, cbar=True)
plt.title(f"|r| correlation heatmap ({os.path.basename(FILE)})")
plt.tight_layout()
plt.show()

