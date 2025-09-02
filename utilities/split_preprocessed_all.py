import os
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IN_FILE = os.path.join(BASE, "data", "preprocessed", "preprocessed_all_without_holes.csv")
OUT_DIR = os.path.join(BASE, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_FILE, parse_dates=["DATE"]).sort_values("DATE")

train = df.iloc[:688].copy()
test  = df.iloc[688:].copy()

train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

print(len(train), "rows in train.csv")
print(len(test),  "rows in test.csv")
print("Train range:", train["DATE"].min().date(), "→", train["DATE"].max().date())
print("Test  range:", test["DATE"].min().date(),  "→", test["DATE"].max().date())
