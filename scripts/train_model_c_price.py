"""
Drishti - Task 4: Model C — Price Impact Model
================================================
Predicts Price_Return_1M (primary), also checks Price_Return_3M, Price_Volatility_3M.
CASCADE: Uses Trade_Return_1M_Pred from Model A and Production_Growth_Pred from Model B.

Run: python scripts/train_model_c_price.py
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MAIN_CSV = os.path.join(DATA_DIR, "Drishti_Cascade_Final_With_EMDAT.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Leakage-safe price-forecast features.  All trade/shock variables are lagged;
# price level is explicitly lagged by one period.
REMOVED_CONTEMPORANEOUS_FEATURES = [
    "Trade_Share",
    "Effective_Shock",
    "Trade_Shock_Exposure",
    "Conflict_Exposure",
    "Protest_Exposure",
    "Net_Hostility_Exposure",
    "Natural_Disaster_Trade_Exposure_USD",
]

LAGGED_FEATURES_C = [
    "Lagged_Effective_Shock_1",
    "Lagged_Effective_Shock_2",
    "Shock_Intensity_Lag1",
    "Shock_Intensity_Lag2",
    "Trade_Share_Lag1",
    "Trade_Share_Lag2",
    "Price_Lag1",
]

EXOGENOUS_FEATURES_C = [
    "GPR",
    "INR_USD_Rate",
]

# Natural_Disaster_Severity_Index is calculated from observed same-period
# disaster outcomes.  It has no lagged counterpart in the dataset, so its
# contemporaneous version is excluded from forecast-time predictors.
EXCLUDED_SAME_PERIOD_REALIZED_FEATURES = ["Natural_Disaster_Severity_Index"]

CASCADE_FEATURES_C = [
    "Trade_Return_1M_Pred",
    "Production_Growth_Pred_Lag1",
]

FEATURES_C = LAGGED_FEATURES_C + EXOGENOUS_FEATURES_C + CASCADE_FEATURES_C

PRIMARY_TARGET = "Price_Return_1M"
SECONDARY_TARGETS = ["Price_Return_3M", "Price_Volatility_3M"]

FORBIDDEN_PREDICTOR_FEATURES = {
    PRIMARY_TARGET,
    *SECONDARY_TARGETS,
    *REMOVED_CONTEMPORANEOUS_FEATURES,
    *EXCLUDED_SAME_PERIOD_REALIZED_FEATURES,
}
unsafe_features = set(FEATURES_C) & FORBIDDEN_PREDICTOR_FEATURES
if unsafe_features:
    raise ValueError(f"Leakage-unsafe Model C features configured: {sorted(unsafe_features)}")

TRAIN_END_YEAR = 2022
VAL_YEAR = 2023
TEST_START_YEAR = 2024


def evaluate_model(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2}


def compute_naive_baseline(df, target_col):
    """Naive persistence baseline for price."""
    df_sorted = df.sort_values(["Country", "Trade_Type", "HS4", "Year", "Month"]).copy()
    df_sorted["naive_pred"] = (
        df_sorted.groupby(["Country", "Trade_Type", "HS4"])[target_col]
        .shift(1)
        .fillna(0)
    )
    return df_sorted["naive_pred"]


def print_leakage_prevention_audit():
    """Report the forecast-time feature constraints used by Model C."""
    print("\n" + "=" * 70)
    print("LEAKAGE-PREVENTION AUDIT")
    print("=" * 70)
    print(f"  Removed contemporaneous features: {', '.join(REMOVED_CONTEMPORANEOUS_FEATURES)}")
    print(f"  Excluded same-period realized feature: {', '.join(EXCLUDED_SAME_PERIOD_REALIZED_FEATURES)}")
    print(f"  Retained lagged features: {', '.join(LAGGED_FEATURES_C)}")
    print("  Cascade feature Trade_Return_1M_Pred is retained, but may include in-sample Model A predictions (not OOF yet).")
    print("  Cascade feature Production_Growth_Pred_Lag1 is retained, but lagging does not make upstream in-sample predictions fully OOF.")
    print("  Dataset was not modified; all feature preparation was performed in memory.")


def main():
    print("=" * 70)
    print("Drishti - Task 4: Model C -- Price Impact")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load main dataset
    print("\nLoading main dataset...")
    df = pd.read_csv(MAIN_CSV)
    df = df.sort_values(["Country", "Trade_Type", "HS4", "Year", "Month"]).reset_index(drop=True)

    # Load Model A predictions
    pred_a_path = os.path.join(RESULTS_DIR, "model_a_predictions.csv")
    if os.path.exists(pred_a_path):
        pred_a = pd.read_csv(
            pred_a_path,
            usecols=["Year", "Month", "Country", "Trade_Type", "HS4", "Trade_Return_1M_Pred"],
        )
        df = df.merge(
            pred_a[["Year", "Month", "Country", "Trade_Type", "HS4", "Trade_Return_1M_Pred"]],
            on=["Year", "Month", "Country", "Trade_Type", "HS4"],
            how="left"
        )
        df["Trade_Return_1M_Pred"] = df["Trade_Return_1M_Pred"].fillna(0)
        print(f"  Model A predictions loaded: {pred_a_path}")
        print("  NOTE: This artifact may contain in-sample Model A predictions; it is not yet an OOF cascade feature.")
    else:
        print("  WARNING: Model A predictions not found. Using zeros.")
        df["Trade_Return_1M_Pred"] = 0

    # Load Model B predictions
    pred_b_path = os.path.join(RESULTS_DIR, "model_b_predictions.csv")
    if os.path.exists(pred_b_path):
        # usecols avoids loading Production_Risk, which is not a Model C
        # feature and can otherwise trigger an unnecessary dtype warning.
        pred_b = pd.read_csv(
            pred_b_path,
            usecols=["Year", "Month", "Country", "Trade_Type", "HS4", "Production_Growth_Pred"],
        )
        df = df.merge(
            pred_b[["Year", "Month", "Country", "Trade_Type", "HS4", "Production_Growth_Pred"]],
            on=["Year", "Month", "Country", "Trade_Type", "HS4"],
            how="left"
        )
        df["Production_Growth_Pred"] = df["Production_Growth_Pred"].fillna(0)

        # LAG by 1 period to prevent leakage (Task 4 spec: shift by one period)
        df = df.sort_values(["Country", "Trade_Type", "HS4", "Year", "Month"])
        df["Production_Growth_Pred_Lag1"] = (
            df.groupby(["Country", "Trade_Type", "HS4"])["Production_Growth_Pred"]
            .shift(1)
            .fillna(0)
        )
        print(f"  Model B predictions loaded and lagged: {pred_b_path}")
        print("  NOTE: Lagging reduces temporal leakage but does not make an upstream in-sample prediction fully out-of-sample.")
    else:
        print("  WARNING: Model B predictions not found. Using zeros.")
        df["Production_Growth_Pred"] = 0
        df["Production_Growth_Pred_Lag1"] = 0

    # Ensure all feature columns exist and have no nulls
    for col in FEATURES_C:
        if col not in df.columns:
            print(f"  WARNING: Feature {col} not found, filling with 0")
            df[col] = 0
        df[col] = df[col].fillna(0)

    # Chronological split
    train = df[df["Year"] <= TRAIN_END_YEAR].copy()
    val = df[df["Year"] == VAL_YEAR].copy()
    test = df[df["Year"] >= TEST_START_YEAR].copy()

    print(f"\n  Chronological split:")
    print(f"    Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    # ============================================================
    # TRAIN ON PRIMARY TARGET
    # ============================================================
    print("\n" + "=" * 70)
    print(f"TRAINING — PRIMARY TARGET: {PRIMARY_TARGET}")
    print("=" * 70)

    X_train = train[FEATURES_C]
    y_train = train[PRIMARY_TARGET]
    X_val = val[FEATURES_C]
    y_val = val[PRIMARY_TARGET]
    X_test = test[FEATURES_C]
    y_test = test[PRIMARY_TARGET]

    # Naive baseline
    df["naive_pred_price"] = compute_naive_baseline(df, PRIMARY_TARGET)
    bl_val = evaluate_model(
        val[PRIMARY_TARGET],
        df.loc[val.index, "naive_pred_price"] if "naive_pred_price" in df.columns else np.zeros(len(val)),
        "baseline_val"
    )
    bl_test = evaluate_model(
        test[PRIMARY_TARGET],
        df.loc[test.index, "naive_pred_price"] if "naive_pred_price" in df.columns else np.zeros(len(test)),
        "baseline_test"
    )
    print(f"\n  Baseline: Val MAE={bl_val['MAE']:.4f} R2={bl_val['R2']:.4f}")
    print(f"            Test MAE={bl_test['MAE']:.4f} R2={bl_test['R2']:.4f}")

    all_results = {}
    best_val_mae = float("inf")
    best_model = None
    best_name = None

    for ModelCls, name, params in [
        (RandomForestRegressor, "RandomForest", dict(
            n_estimators=300, max_depth=15, min_samples_split=10,
            min_samples_leaf=5, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1)),
        (xgb.XGBRegressor, "XGBoost", dict(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
        (lgb.LGBMRegressor, "LightGBM", dict(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
    ]:
        print(f"\n  Training {name}...")
        model = ModelCls(**params)
        model.fit(X_train, y_train)

        m_train = evaluate_model(y_train, model.predict(X_train), f"{name}_train")
        m_val = evaluate_model(y_val, model.predict(X_val), f"{name}_val")
        m_test = evaluate_model(y_test, model.predict(X_test), f"{name}_test")

        print(f"    Train: MAE={m_train['MAE']:.4f} R2={m_train['R2']:.4f}")
        print(f"    Val:   MAE={m_val['MAE']:.4f} R2={m_val['R2']:.4f}")
        print(f"    Test:  MAE={m_test['MAE']:.4f} R2={m_test['R2']:.4f}")
        beats = m_test['MAE'] < bl_test['MAE']
        print(f"    {'BEATS' if beats else 'DOES NOT BEAT'} baseline on test")

        # Suspiciously high R2 check
        for split_n, m in [("train", m_train), ("val", m_val), ("test", m_test)]:
            if m["R2"] > 0.95:
                print(f"    *** RED FLAG: R2={m['R2']:.4f} on {split_n} — re-audit features!")

        # Feature importances
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(FEATURES_C, model.feature_importances_))
            importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
            if name == "LightGBM":  # Print for best model
                print(f"    Feature importances ({name}):")
                for rank, (feat, imp) in enumerate(importances.items(), 1):
                    print(f"      {rank:>2}. {feat:<45} {imp:.6f}")
        else:
            importances = {}

        all_results[name] = {
            "metrics": {"train": m_train, "val": m_val, "test": m_test},
            "importances": importances,
            "beats_baseline_test": beats,
        }

        if m_val["MAE"] < best_val_mae:
            best_val_mae = m_val["MAE"]
            best_model = model
            best_name = name

    print(f"\n  Best model: {best_name} (Val MAE={best_val_mae:.4f})")

    # ============================================================
    # SECONDARY TARGETS (brief check)
    # ============================================================
    for sec_target in SECONDARY_TARGETS:
        print(f"\n  --- Secondary target: {sec_target} ---")
        model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        model.fit(X_train, train[sec_target])
        m_train = evaluate_model(train[sec_target], model.predict(X_train), f"LGB_train")
        m_val = evaluate_model(val[sec_target], model.predict(X_val), f"LGB_val")
        m_test = evaluate_model(test[sec_target], model.predict(X_test), f"LGB_test")
        print(f"    LightGBM Train: MAE={m_train['MAE']:.4f} R2={m_train['R2']:.4f}")
        print(f"             Val:   MAE={m_val['MAE']:.4f} R2={m_val['R2']:.4f}")
        print(f"             Test:  MAE={m_test['MAE']:.4f} R2={m_test['R2']:.4f}")
        if sec_target == "Price_Volatility_3M" and abs(m_test["R2"]) < 0.05:
            print("    HONEST ASSESSMENT: Test R2 is close to zero; this does not demonstrate strong predictive ability.")
        all_results[f"{sec_target}_LightGBM"] = {"train": m_train, "val": m_val, "test": m_test}

    # ============================================================
    # SAVE ARTIFACTS
    # ============================================================
    print("\n" + "=" * 70)
    print("SAVING ARTIFACTS")
    print("=" * 70)

    # Save best model
    model_path = os.path.join(MODELS_DIR, "model_c_price.joblib")
    joblib.dump(best_model, model_path)
    print(f"  Model saved: {model_path}")

    # Save predictions for Model D cascade
    pred_df = df[["Year", "Month", "Country", "Trade_Type", "HS4"]].copy()
    pred_df["Price_Return_1M_Pred"] = best_model.predict(df[FEATURES_C])
    pred_df["Price_Return_1M_Actual"] = df[PRIMARY_TARGET]
    pred_path = os.path.join(RESULTS_DIR, "model_c_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"  Predictions saved: {pred_path}")

    # Save results
    results = {
        "task": "Task 4 - Model C: Price Impact",
        "timestamp": datetime.now().isoformat(),
        "features": FEATURES_C,
        "primary_target": PRIMARY_TARGET,
        "best_model": best_name,
        "baseline": {"val": bl_val, "test": bl_test},
        "model_results": all_results,
        "split": {"train": len(train), "val": len(val), "test": len(test)},
    }
    results_path = os.path.join(RESULTS_DIR, "model_c_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {results_path}")

    print_leakage_prevention_audit()
    print("\n  TASK 4 COMPLETE")


if __name__ == "__main__":
    main()
