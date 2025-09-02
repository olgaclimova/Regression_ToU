import os, argparse
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, cross_validate, learning_curve, KFold
from xgboost import XGBRegressor

RANDOM_STATE = 42
SCORING = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]

def evaluate_model(name, model, X, y, cv):
    cv_results = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    preds = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    return {
        "Model": name,
        "CV_RMSE": -cv_results["test_neg_root_mean_squared_error"].mean(),
        "CV_MAE": -cv_results["test_neg_mean_absolute_error"].mean(),
        "CV_R2":  cv_results["test_r2"].mean(),
        "Preds": preds,
        "Pipeline": model
    }

def plot_diagnostics(name, y_true, y_pred, model, X, y, cv_splits):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(name, fontsize=14, fontweight="bold")

    # 1) Prediction error
    sns.scatterplot(x=y_true, y=y_pred, ax=axes[0], alpha=0.6)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    axes[0].set_title("Prediction Error"); axes[0].set_xlabel("True"); axes[0].set_ylabel("Pred")

    # 2) Learning curve (R²)
    tr_sizes, tr_scores, te_scores = learning_curve(model, X, y, cv=cv_splits, scoring="r2", n_jobs=-1, random_state=RANDOM_STATE)
    axes[1].plot(tr_sizes, tr_scores.mean(axis=1), "o-", label="Train")
    axes[1].plot(tr_sizes, te_scores.mean(axis=1), "o-", label="CV")
    axes[1].set_title("Learning Curve"); axes[1].set_xlabel("Training examples"); axes[1].set_ylabel("R²"); axes[1].legend()

    # 3) Residuals
    residuals = y_true - y_pred
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[2], alpha=0.6)
    axes[2].axhline(0, color="r", linestyle="--")
    axes[2].set_title("Residuals"); axes[2].set_xlabel("Predictions"); axes[2].set_ylabel("Residuals")

    plt.tight_layout(); plt.show()

def run(train_file, features, target, cv_splits=5, save_csv=None, plots=True):
    df = pd.read_csv(train_file).dropna(subset=features + [target])
    X, y = df[features], df[target]
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    results = []

    results.append(evaluate_model("Linear",
        Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]), X, y, cv))

    results.append(evaluate_model("Lasso",
        Pipeline([("scaler", StandardScaler()), ("model", Lasso(random_state=RANDOM_STATE))]), X, y, cv))

    results.append(evaluate_model("Polynomial(2)",
        Pipeline([("poly", PolynomialFeatures(2, include_bias=False)),
                  ("scaler", StandardScaler()),
                  ("model", LinearRegression())]), X, y, cv))

    results.append(evaluate_model("RandomForest",
        Pipeline([("model", RandomForestRegressor(random_state=RANDOM_STATE))]), X, y, cv))

    results.append(evaluate_model("XGBoost",
        Pipeline([("model", XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1))]), X, y, cv))

    table = pd.DataFrame([{k:v for k,v in r.items() if k not in ["Preds","Pipeline"]} for r in results])
    print("\n=== CV Results (Default Hyperparameters) ===")
    print(table.to_string(index=False))

    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        table.to_csv(save_csv, index=False)

    if plots:
        for r in results:
            plot_diagnostics(r["Model"], y, r["Preds"], r["Pipeline"], X, y, cv_splits)

    return table

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--features", required=True, help="Comma-separated feature names")
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--save-csv", default=None)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    feats = [f.strip() for f in args.features.split(",") if f.strip()]
    run(
        train_file=args.train_file,
        features=feats,
        target=args.target,
        cv_splits=args.cv_splits,
        save_csv=args.save_csv,
        plots=not args.no_plots
    )

