import os, argparse, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

RANDOM_STATE = 42
CV_SPLITS = 5
SCORING = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]

# ---------- helpers ----------
def cv_scores(est, X, y, cv):
    cvres = cross_validate(est, X, y, cv=cv, n_jobs=-1, scoring=SCORING)
    return {
        "CV_RMSE": -cvres["test_neg_root_mean_squared_error"].mean(),
        "CV_MAE": -cvres["test_neg_mean_absolute_error"].mean(),
        "CV_R2":  cvres["test_r2"].mean(),
    }

def prediction_error(y_true, y_pred, ax):
    sns.scatterplot(x=y_true, y=y_pred, ax=ax, alpha=0.6)
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_title("Prediction Error"); ax.set_xlabel("True"); ax.set_ylabel("Pred")

def learning_curve_plot(model, X, y, ax):
    ts, tr, te = learning_curve(model, X, y, cv=CV_SPLITS, scoring="r2", n_jobs=-1)
    ax.plot(ts, tr.mean(axis=1), "o-", label="Train")
    ax.plot(ts, te.mean(axis=1), "o-", label="CV")
    ax.set_title("Learning Curve (R²)"); ax.set_xlabel("Training examples"); ax.set_ylabel("R²"); ax.legend()

def residuals_plot(y_true, y_pred, ax):
    res = y_true - y_pred
    sns.scatterplot(x=y_pred, y=res, ax=ax, alpha=0.6)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_title("Residuals"); ax.set_xlabel("Predictions"); ax.set_ylabel("Residuals")

def tune_rf(X, y, cv, n_iter=40):
    pipe = Pipeline([("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    space = {
        "model__n_estimators": [300, 500, 800, 1000],
        "model__max_depth": [None, 8, 12, 16, 24],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.5, 1.0],
        "model__bootstrap": [True, False],
    }
    rs = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=cv, n_jobs=-1,
                            scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE, verbose=0)
    rs.fit(X, y)
    return rs.best_estimator_, rs.best_params_

def tune_xgb(X, y, cv, n_iter=60):
    pipe = Pipeline([("model", XGBRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", objective="reg:squarederror"
    ))])
    space = {
        "model__n_estimators": [600, 800, 1000, 1200],
        "model__max_depth": [4, 6, 8, 10],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 4.0],
        "model__min_child_weight": [1, 3, 5, 7],
        "model__gamma": [0, 0.1, 0.3],
    }
    rs = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=cv, n_jobs=-1,
                            scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE, verbose=0)
    rs.fit(X, y)
    return rs.best_estimator_, rs.best_params_

def rf_params_table(best_estimator, default_estimator):
    keys = ["n_estimators","max_depth","min_samples_split","min_samples_leaf","max_features","bootstrap"]
    bp = best_estimator.get_params(); dp = default_estimator.get_params()
    rows = []
    for k in keys:
        kk = "model__" + k
        rows.append({"param": k, "Tuned": bp.get(kk, None), "Default": dp.get(kk, None)})
    return pd.DataFrame(rows).set_index("param")

def xgb_params_table_hardcoded(best_estimator):
    bp = best_estimator.get_params()
    tuned = {
        "n_estimators":  bp.get("model__n_estimators"),
        "max_depth":     bp.get("model__max_depth"),
        "learning_rate": bp.get("model__learning_rate"),
    }
    defaults = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
    }
    df = pd.DataFrame({"Tuned": tuned, "Default": defaults})
    return df

# ---------- main ----------
def run(train_file, features, target, n_iter_rf=40, n_iter_xgb=60, save_params=None):
    assert os.path.basename(train_file).startswith("train_log_"), "Input must start with 'train_log_...'"
    df = pd.read_csv(train_file).dropna(subset=features + [target])
    X, y = df[features], df[target]
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Tuning
    best_rf,  rf_best_params = tune_rf(X, y, cv, n_iter=n_iter_rf)
    best_xgb, xgb_best_params = tune_xgb(X, y, cv, n_iter=n_iter_xgb)

    # CV (tuned)
    rf_scores  = cv_scores(best_rf, X, y, cv)
    xgb_scores = cv_scores(best_xgb, X, y, cv)
    tuned_df = pd.DataFrame([
        {"Model":"RandomForest", **rf_scores},
        {"Model":"XGBoost",      **xgb_scores},
    ])
    print("\n=== CV Results (Tuned) ===")
    print(tuned_df.to_string(index=False))

    # Parameter tables (stampa)
    rf_default = Pipeline([("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    rf_table = rf_params_table(best_estimator=best_rf, default_estimator=rf_default)
    xgb_table = xgb_params_table_hardcoded(best_estimator=best_xgb)

    print("\n=== Random Forest Parameters ===")
    print(rf_table.to_string())
    print("\n=== XGBoost Parameters ===")
    print(xgb_table.to_string())

    # Salvataggio best params (opzionale)
    if save_params:
        os.makedirs(os.path.dirname(save_params), exist_ok=True)
        payload = {
            "RandomForest": rf_best_params,
            "XGBoost": xgb_best_params
        }
        # normalizza: togliamo "model__" dalle chiavi per comodità downstream
        clean_payload = {}
        for model, params in payload.items():
            clean = {k.replace("model__", ""): v for k, v in params.items()}
            clean_payload[model] = clean
        with open(save_params, "w", encoding="utf-8") as f:
            json.dump(clean_payload, f, indent=2)
        print(f"\nSaved best params to: {save_params}")

    # --------- plots (tuned models) ----------
    rf_preds = cross_val_predict(best_rf, X, y, cv=cv, n_jobs=-1)
    fig, axes = plt.subplots(1, 3, figsize=(18,5)); fig.suptitle("RandomForest (Tuned)")
    prediction_error(y, rf_preds, axes[0]); learning_curve_plot(best_rf, X, y, axes[1]); residuals_plot(y, rf_preds, axes[2])
    plt.tight_layout(); plt.show()

    xgb_preds = cross_val_predict(best_xgb, X, y, cv=cv, n_jobs=-1)
    fig, axes = plt.subplots(1, 3, figsize=(18,5)); fig.suptitle("XGBoost (Tuned)")
    prediction_error(y, xgb_preds, axes[0]); learning_curve_plot(best_xgb, X, y, axes[1]); residuals_plot(y, xgb_preds, axes[2])
    plt.tight_layout(); plt.show()

    return tuned_df, rf_table, xgb_table, best_rf, best_xgb

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--features", required=True, help="Comma-separated feature names")
    ap.add_argument("--n-iter-rf", type=int, default=40)
    ap.add_argument("--n-iter-xgb", type=int, default=60)
    ap.add_argument("--save-params", default=None, help="Path to save best params JSON (e.g., ../results/best_params.json)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    feats = [f.strip() for f in args.features.split(",") if f.strip()]
    run(args.train_file, feats, args.target, n_iter_rf=args.n_iter_rf, n_iter_xgb=args.n_iter_xgb, save_params=args.save_params)


