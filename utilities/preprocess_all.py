# Keep only rows with NO2 and TRANSITS present; save a gap-free table.

import os
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IN_FILE  = os.path.join(BASE, "data", "preprocessed", "preprocessed_all_with_holes.csv")
OUT_FILE = os.path.join(BASE, "data", "preprocessed", "preprocessed_all_without_holes.csv")

# load
df = pd.read_csv(IN_FILE)

# filter: keep only days with both NO2 and TRANSITS available
out = df[df["NO2"].notna() & df["TRANSITS"].notna()].sort_values("DATE")

# save
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
out.to_csv(OUT_FILE, index=False)

# report 
print(f"Rows kept: {len(out)} | Rows removed: {len(df) - len(out)}")
