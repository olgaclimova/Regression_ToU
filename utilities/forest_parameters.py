import os, json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

BEST_PARAMS_JSON = "../results/best_params.json"
TRAIN_FILE = "../data/processed/train_log_outliers_NO2.csv"
FEATURES = ["TRANSITS","TEMPERATURE C","WIND km/h"]
TARGET = "NO2"

def load_best_rf_params(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rf = data.get("RandomForest", {})
    # fallback essenziali
    rf.setdefault("random_state", 42)
    rf.setdefault("n_jobs", -1)
    return rf

def main():
    # 1) data
    df = pd.read_csv(TRAIN_FILE)
    X, y = df[FEATURES], df[TARGET]

    # 2) params -> model
    params = load_best_rf_params(BEST_PARAMS_JSON)
    rf = RandomForestRegressor(**params)
    pipe = Pipeline([("model", rf)])
    pipe.fit(X, y)

    # 3) built-in importances
    imp = pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"Feature": FEATURES, "Importance": imp}).sort_values("Importance", ascending=False)
    print("\n=== Random Forest — Built-in Feature Importances ===")
    print(imp_df.to_string(index=False))
    plt.figure(figsize=(6,4)); sns.barplot(data=imp_df, x="Importance", y="Feature"); plt.title("RF Feature Importances")
    plt.tight_layout(); plt.show()

    # 4) permutation importance (ΔR²)
    perm = permutation_importance(pipe, X, y, scoring="r2", n_repeats=20, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({"Feature": FEATURES, "PermImportance": perm.importances_mean, "Std": perm.importances_std}) \
                .sort_values("PermImportance", ascending=False)
    print("\n=== Random Forest — Permutation Importance (ΔR²) ===")
    print(perm_df.to_string(index=False))
    plt.figure(figsize=(6,4)); sns.barplot(data=perm_df, x="PermImportance", y="Feature"); plt.title("RF Permutation Importance (ΔR²)")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()

