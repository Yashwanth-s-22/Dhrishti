"""
Drishti - Tasks 5+6: Model D (Agricultural Economic Impact) - Walk-Forward OOF Implementation
=============================================================================================
Trains Model D targets using temporally valid Model C out-of-fold (OOF) predictions:
- Target 1: Agri_GVA_Growth_Percent (annual macro growth rate)
- Target 2: Inflation_Change_3M (3-month food inflation delta)

Inputs:
- 13 safe lagged/exogenous features
- Upstream cascade feature: Price_Return_1M_Pred_Lag1 (shift 1 of Model C OOF predictions)

Methodological & Optimization Rules:
1. Strict chronological boundaries:
   - Train: <= 2022
   - Validation: 2023
   - Test: 2024-2025
2. Hyperparameter optimization and model selection strictly use Train and Validation (2023).
   The Test set (2024-2025) is NEVER used for tuning or model selection.
3. Saves canonical models to models/model_d_agri_gva.joblib,
   models/model_d_inflation.joblib, and results to results/model_d_results.json.

Run: python scripts/train_model_d_economy.py
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore")

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
PRED_C_OOF_PATH = os.path.join(RESULTS_DIR, "model_c_predictions_oof.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------
# SAFE FEATURE SCHEMA FOR MODEL D (13 features)
# ------------------------------------------------------------
SAFE_FEATURES_D = [
    "Inflation_Lag1",
    "Agri_GVA_Lag1",
    "GDP_Lag1",
    "Price_Lag1",
    "Shock_Intensity_Lag1",
    "Shock_Intensity_Lag2",
    "Trade_Share_Lag1",
    "Trade_Share_Lag2",
    "Lagged_Effective_Shock_1",
    "Lagged_Effective_Shock_2",
    "GPR",
    "INR_USD_Rate",
    "Price_Return_1M_Pred_Lag1",
]

TARGET_GVA = "Agri_GVA_Growth_Percent"
TARGET_INFLATION = "Inflation_Change_3M"

TRAIN_END_YEAR = 2022
VAL_YEAR = 2023
TEST_START_YEAR = 2024


def evaluate_model(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan
    return {"label": label, "MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def format_r2(metrics):
    val = metrics.get("R2", None)
    if val is None or np.isnan(val):
        return "N/A"
    return f"{val:.4f}"


# ============================================================
# DATA PREPARATION WITH OOF MODEL C INPUTS
# ============================================================
def load_and_prepare_data():
    """
    Load main dataset and align temporally valid Model C OOF predictions:
    Price_Return_1M_Pred_Lag1 = shift(1) of Price_Return_1M_Pred_OOF
    grouped by (Country, Trade_Type, HS4).
    """
    print("\nLoading main dataset and aligning Model C OOF prediction artifacts...")
    key_cols = ["Country", "Trade_Type", "HS4", "Year", "Month"]
    df = pd.read_csv(MAIN_CSV).sort_values(key_cols).reset_index(drop=True)

    if not os.path.exists(PRED_C_OOF_PATH):
        raise FileNotFoundError(f"Model C OOF artifact not found at {PRED_C_OOF_PATH}. Run train_model_c_price.py first.")

    pred_c_oof = pd.read_csv(PRED_C_OOF_PATH).sort_values(key_cols).reset_index(drop=True)
    assert (df[key_cols] == pred_c_oof[key_cols]).all().all(), "Key alignment mismatch between main_df and model_c_predictions_oof"
    df["Price_Return_1M_Pred_OOF"] = pred_c_oof["Price_Return_1M_Pred_OOF"]

    # Sort chronologically by series and apply shift(1)
    df = df.sort_values(key_cols).reset_index(drop=True)
    df["Price_Return_1M_Pred_Lag1"] = (
        df.groupby(["Country", "Trade_Type", "HS4"])["Price_Return_1M_Pred_OOF"]
        .shift(1)
    )

    missing_c_lag = df["Price_Return_1M_Pred_Lag1"].isna().sum()
    print(f"  Aligned Model C OOF Lag1: {len(df) - missing_c_lag:,} valid, {missing_c_lag:,} missing/cold-start")

    df["Price_Return_1M_Pred_Lag1"] = df["Price_Return_1M_Pred_Lag1"].fillna(0.0)

    for col in SAFE_FEATURES_D:
        if col not in df.columns:
            print(f"  WARNING: {col} not in columns, filling with 0.0")
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    return df


# ============================================================
# MODEL D-1: AGRI GVA GROWTH PERCENT (ANNUAL TARGET)
# ============================================================
def train_and_optimize_gva(df):
    """
    Train and optimize Model D-1 (Agri GVA Growth Percent).
    Annual macroeconomic target aggregated by calendar year.
    """
    print("\n" + "=" * 70)
    print("MODEL D-1: Agri_GVA_Growth_Percent (ANNUAL TARGET)")
    print("=" * 70)

    annual_features = df.groupby("Year")[SAFE_FEATURES_D].mean()
    annual_targets = df.groupby("Year")[TARGET_GVA].first()

    annual_df = annual_features.copy()
    annual_df[TARGET_GVA] = annual_targets
    annual_df["Previous_Year_GVA_Baseline"] = annual_targets.shift(1)
    annual_df = annual_df.dropna(subset=SAFE_FEATURES_D + [TARGET_GVA, "Previous_Year_GVA_Baseline"])

    annual_train = annual_df[annual_df.index <= TRAIN_END_YEAR].copy()
    annual_val = annual_df[annual_df.index == VAL_YEAR].copy()
    annual_test = annual_df[annual_df.index >= TEST_START_YEAR].copy()

    print(f"  Annual split: Train years={annual_train.index.tolist()} | Val years={annual_val.index.tolist()} | Test years={annual_test.index.tolist()}")

    X_train, y_train = annual_train[SAFE_FEATURES_D], annual_train[TARGET_GVA]
    X_val, y_val = annual_val[SAFE_FEATURES_D], annual_val[TARGET_GVA]
    X_test, y_test = annual_test[SAFE_FEATURES_D], annual_test[TARGET_GVA]

    # Baseline: previous annual GVA value
    bl_train = evaluate_model(y_train, annual_train["Previous_Year_GVA_Baseline"], "baseline_train")
    bl_val = evaluate_model(y_val, annual_val["Previous_Year_GVA_Baseline"], "baseline_val")
    bl_test = evaluate_model(y_test, annual_test["Previous_Year_GVA_Baseline"], "baseline_test")

    print(f"  Baseline (previous annual value): Train MAE={bl_train['MAE']:.4f} | Val MAE={bl_val['MAE']:.4f} | Test MAE={bl_test['MAE']:.4f}")

    candidate_models = [
        ("RandomForest", "RF_default", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)),
        ("RandomForest", "RF_regularized", RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
        ("LightGBM", "LGB_default", lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ("XGBoost", "XGB_regularized", xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, reg_alpha=0.5, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
    ]

    best_gva_config = None
    best_gva_val_mae = float("inf")
    gva_results = {}

    for family, name, model in candidate_models:
        model.fit(X_train, y_train)
        m_train = evaluate_model(y_train, model.predict(X_train), f"{name}_train")
        m_val = evaluate_model(y_val, model.predict(X_val), f"{name}_val")
        m_test = evaluate_model(y_test, model.predict(X_test), f"{name}_test")

        beats_val = m_val["MAE"] < bl_val["MAE"]
        beats_test = m_test["MAE"] < bl_test["MAE"]
        print(f"  [{family:<12}] {name:<16} -> Val MAE: {m_val['MAE']:.4f} ({'BEATS' if beats_val else 'LOSES TO'} bl) | Test MAE: {m_test['MAE']:.4f} | Train MAE: {m_train['MAE']:.4f}")

        gva_results[name] = {
            "family": family,
            "train": m_train,
            "val": m_val,
            "test": m_test,
            "baseline_val": bl_val,
            "baseline_test": bl_test,
            "beats_baseline_val": beats_val,
            "beats_baseline_test": beats_test,
        }

        if m_val["MAE"] < best_gva_val_mae:
            best_gva_val_mae = m_val["MAE"]
            best_gva_config = (family, name, model, m_train, m_val, m_test, beats_val, beats_test)

    best_family, best_name, best_model, best_train_m, best_val_m, best_test_m, b_val, b_test = best_gva_config
    print(f"\n  Selected GVA Winner: {best_family} ({best_name}) with Val MAE = {best_gva_val_mae:.4f}")

    canonical_model_path = os.path.join(MODELS_DIR, "model_d_agri_gva.joblib")
    oof_model_path = os.path.join(MODELS_DIR, "model_d_agri_gva_oof.joblib")
    joblib.dump(best_model, canonical_model_path)
    joblib.dump(best_model, oof_model_path)
    print(f"  Saved Model D Agri GVA Model: {canonical_model_path}")

    return {
        "target": TARGET_GVA,
        "selected_model": best_name,
        "selected_family": best_family,
        "metrics": {"train": best_train_m, "val": best_val_m, "test": best_test_m},
        "baseline": {"train": bl_train, "val": bl_val, "test": bl_test},
        "beats_baseline_val": b_val,
        "beats_baseline_test": b_test,
        "candidate_results": gva_results,
    }


# ============================================================
# MODEL D-2: INFLATION CHANGE 3M (MONTHLY TARGET)
# ============================================================
def train_and_optimize_inflation(df):
    """
    Train and optimize Model D-2 (Inflation_Change_3M).
    Monthly food inflation delta target.
    """
    print("\n" + "=" * 70)
    print("MODEL D-2: Inflation_Change_3M (MONTHLY TARGET)")
    print("=" * 70)

    train = df[df["Year"] <= TRAIN_END_YEAR]
    val = df[df["Year"] == VAL_YEAR]
    test = df[df["Year"] >= TEST_START_YEAR]

    X_train = train[SAFE_FEATURES_D]
    y_train = train[TARGET_INFLATION]
    X_val = val[SAFE_FEATURES_D]
    y_val = val[TARGET_INFLATION]
    X_test = test[SAFE_FEATURES_D]
    y_test = test[TARGET_INFLATION]

    # Baseline: 0 change (no inflation change)
    bl_val = evaluate_model(y_val, np.zeros(len(y_val)), "baseline_val")
    bl_test = evaluate_model(y_test, np.zeros(len(y_test)), "baseline_test")
    print(f"  Baseline (predict 0): Val MAE={bl_val['MAE']:.4f} | Test MAE={bl_test['MAE']:.4f}")

    candidate_models = [
        ("LightGBM", "LGB_baseline", lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ("LightGBM", "LGB_regularized", lgb.LGBMRegressor(n_estimators=250, max_depth=5, learning_rate=0.03, reg_alpha=0.5, reg_lambda=1.0, subsample=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ("RandomForest", "RF_baseline", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)),
        ("RandomForest", "RF_regularized", RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)),
        ("XGBoost", "XGB_regularized", xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.03, reg_alpha=0.5, reg_lambda=1.0, subsample=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
    ]

    best_infl_config = None
    best_infl_val_mae = float("inf")
    infl_results = {}

    for family, name, model in candidate_models:
        model.fit(X_train, y_train)
        m_train = evaluate_model(y_train, model.predict(X_train), f"{name}_train")
        m_val = evaluate_model(y_val, model.predict(X_val), f"{name}_val")
        m_test = evaluate_model(y_test, model.predict(X_test), f"{name}_test")

        beats_val = m_val["MAE"] < bl_val["MAE"]
        beats_test = m_test["MAE"] < bl_test["MAE"]
        print(f"  [{family:<12}] {name:<16} -> Val MAE: {m_val['MAE']:.4f} ({'BEATS' if beats_val else 'LOSES TO'} bl) | Test MAE: {m_test['MAE']:.4f} | Train MAE: {m_train['MAE']:.4f}")

        infl_results[name] = {
            "family": family,
            "train": m_train,
            "val": m_val,
            "test": m_test,
            "baseline_val": bl_val,
            "baseline_test": bl_test,
            "beats_baseline_val": beats_val,
            "beats_baseline_test": beats_test,
        }

        if m_val["MAE"] < best_infl_val_mae:
            best_infl_val_mae = m_val["MAE"]
            best_infl_config = (family, name, model, m_train, m_val, m_test, beats_val, beats_test)

    best_family, best_name, best_model, best_train_m, best_val_m, best_test_m, b_val, b_test = best_infl_config
    print(f"\n  Selected Inflation Winner: {best_family} ({best_name}) with Val MAE = {best_infl_val_mae:.4f}")

    canonical_model_path = os.path.join(MODELS_DIR, "model_d_inflation.joblib")
    oof_model_path = os.path.join(MODELS_DIR, "model_d_inflation_oof.joblib")
    joblib.dump(best_model, canonical_model_path)
    joblib.dump(best_model, oof_model_path)
    print(f"  Saved Model D Inflation Model: {canonical_model_path}")

    return {
        "target": TARGET_INFLATION,
        "selected_model": best_name,
        "selected_family": best_family,
        "metrics": {"train": best_train_m, "val": best_val_m, "test": best_test_m},
        "baseline": {"val": bl_val, "test": bl_test},
        "beats_baseline_val": b_val,
        "beats_baseline_test": b_test,
        "candidate_results": infl_results,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("Drishti - Tasks 5+6: Model D (Agricultural Economic Impact) OOF Retraining")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # 1. Load data with OOF Model C inputs
    df = load_and_prepare_data()

    # 2. Train and optimize GVA Model
    gva_summary = train_and_optimize_gva(df)

    # 3. Train and optimize Inflation Model
    infl_summary = train_and_optimize_inflation(df)

    # 4. Save Combined Results
    results_d_oof = {
        "task": "Tasks 5+6 - Model D Economic Impact (OOF Inputs)",
        "timestamp": datetime.now().isoformat(),
        "upstream_artifacts_used": ["results/model_c_predictions_oof.csv"],
        "safe_features": SAFE_FEATURES_D,
        "gva_model": gva_summary,
        "inflation_model": infl_summary,
        "test_set_isolation": "Verified: Test set (2024-2025) was untouched during hyperparameter search and model selection.",
    }

    results_d_path = os.path.join(RESULTS_DIR, "model_d_results.json")
    results_d_oof_path = os.path.join(RESULTS_DIR, "model_d_results_oof.json")
    with open(results_d_path, "w") as f:
        json.dump(results_d_oof, f, indent=2, default=str)
    with open(results_d_oof_path, "w") as f:
        json.dump(results_d_oof, f, indent=2, default=str)
    print(f"\nSaved Model D Results: {results_d_path}")

    print("\n" + "=" * 70)
    print("MODEL D COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
