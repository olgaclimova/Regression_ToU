"""
Detect and normalize outliers in NO2 using IQR method.
Output: test_log_outliers_NO2.csv
"""

import os
import pandas as pd

# --- CONFIG ---
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_FILE = os.path.join(BASE, "data", "processed", "train.csv")
OUTPUT_FILE = os.path.join(BASE, "data", "processed", "train_log_outliers_NO2.csv")
TARGET = "NO2"

def cap_outliers_iqr(df, column):
    """Replace outliers with lower/upper IQR bounds"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    capped = df[column].clip(lower, upper)
    return capped

def run():
    df = pd.read_csv(INPUT_FILE)

    df[TARGET] = cap_outliers_iqr(df, TARGET)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved new dataset with capped NO2 outliers in train_log_outliers_NO2.csv")

if __name__ == "__main__":
    run()
