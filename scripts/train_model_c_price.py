"""
Drishti - Task 4: Model C (Price Impact) - Walk-Forward OOF Implementation
==========================================================================
Trains Model C using temporally valid out-of-fold (OOF) predictions from:
- Model A: Trade_Return_1M_Pred_OOF
- Model B: Production_Growth_Pred_Lag1 (shift 1 of Model B OOF)

Methodological & Optimization Rules:
1. Strict chronological boundaries:
   - Train: 2019-2022 (2018 is cold-start)
   - Validation: 2023
   - Test: 2024-2025
2. Hyperparameter optimization and model selection use ONLY Train and Validation (2023).
   The Test set (2024-2025) is NEVER used for tuning or model selection.
3. Generates temporally valid walk-forward predictions for Model C:
   - Predict 2020 using data <= 2019
   - Predict 2021 using data <= 2020
   - Predict 2022 using data <= 2021
   - Predict 2023 using data <= 2022
   - Predict 2024 using data <= 2023
   - Predict 2025 using data <= 2024
   - 2018-2019 cold-start marked with Is_Out_Of_Sample = False.
4. Saves canonical model to models/model_c_price.joblib and
   predictions to results/model_c_predictions_oof.csv.

Run: python scripts/train_model_c_price.py
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
PRED_A_OOF_PATH = os.path.join(RESULTS_DIR, "model_a_predictions_oof.csv")
PRED_B_OOF_PATH = os.path.join(RESULTS_DIR, "model_b_predictions_oof.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------
# FEATURE SCHEMA FOR MODEL C (11 features)
# ------------------------------------------------------------
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
CASCADE_FEATURES_C = [
    "Trade_Return_1M_Pred",
    "Production_Growth_Pred_Lag1",
]
FEATURES_C = LAGGED_FEATURES_C + EXOGENOUS_FEATURES_C + CASCADE_FEATURES_C

PRIMARY_TARGET = "Price_Return_1M"
SECONDARY_TARGETS = ["Price_Return_3M", "Price_Volatility_3M"]

TRAIN_END_YEAR = 2022
VAL_YEAR = 2023
TEST_START_YEAR = 2024


def evaluate_model(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"label": label, "MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def compute_naive_baseline(df, target_col):
    """Naive persistence baseline for price."""
    df_sorted = df.sort_values(["Country", "Trade_Type", "HS4", "Year", "Month"]).copy()
    df_sorted["naive_pred"] = (
        df_sorted.groupby(["Country", "Trade_Type", "HS4"])[target_col]
        .shift(1)
        .fillna(0)
    )
    return df_sorted["naive_pred"]


# ============================================================
# DATA PREPARATION WITH OOF UPSTREAM INPUTS
# ============================================================
def load_and_prepare_data():
    """
    Load main dataset and align temporally valid OOF upstream predictions:
    - Model A: Trade_Return_1M_Pred_OOF
    - Model B: Production_Growth_Pred_OOF, then shift(1) by (Country, Trade_Type, HS4)
    """
    print("\nLoading main dataset and aligning OOF upstream prediction artifacts...")
    key_cols = ["Country", "Trade_Type", "HS4", "Year", "Month"]
    df = pd.read_csv(MAIN_CSV).sort_values(key_cols).reset_index(drop=True)

    # 1. Align Model A OOF Predictions
    if not os.path.exists(PRED_A_OOF_PATH):
        raise FileNotFoundError(f"Model A OOF artifact not found at {PRED_A_OOF_PATH}. Run generate_oof_predictions.py first.")

    pred_a_oof = pd.read_csv(PRED_A_OOF_PATH).sort_values(key_cols).reset_index(drop=True)
    assert (df[key_cols] == pred_a_oof[key_cols]).all().all(), "Key alignment mismatch between main_df and model_a_predictions_oof"
    df["Trade_Return_1M_Pred"] = pred_a_oof["Trade_Return_1M_Pred_OOF"]

    # 2. Align Model B OOF Predictions and Shift by 1 period
    if not os.path.exists(PRED_B_OOF_PATH):
        raise FileNotFoundError(f"Model B OOF artifact not found at {PRED_B_OOF_PATH}. Run generate_oof_predictions.py first.")

    pred_b_oof = pd.read_csv(PRED_B_OOF_PATH).sort_values(key_cols).reset_index(drop=True)
    assert (df[key_cols] == pred_b_oof[key_cols]).all().all(), "Key alignment mismatch between main_df and model_b_predictions_oof"
    df["Production_Growth_Pred_OOF"] = pred_b_oof["Production_Growth_Pred_OOF"]
    df["Has_Production_Data"] = pred_b_oof["Has_Production_Data"]

    # Sort chronologically by time series group and apply shift(1)
    df = df.sort_values(key_cols).reset_index(drop=True)
    df["Production_Growth_Pred_Lag1"] = (
        df.groupby(["Country", "Trade_Type", "HS4"])["Production_Growth_Pred_OOF"]
        .shift(1)
    )

    # Document missing counts
    missing_a = df["Trade_Return_1M_Pred"].isna().sum()
    missing_b_lag = df["Production_Growth_Pred_Lag1"].isna().sum()
    print(f"  Aligned Model A OOF: {len(df) - missing_a:,} valid, {missing_a:,} missing/cold-start")
    print(f"  Aligned Model B OOF Lag1: {len(df) - missing_b_lag:,} valid crop predictions, {missing_b_lag:,} missing/non-crop/cold-start")

    # Missing value handling policy for modeling:
    df["Trade_Return_1M_Pred"] = df["Trade_Return_1M_Pred"].fillna(0.0)
    df["Production_Growth_Pred_Lag1"] = df["Production_Growth_Pred_Lag1"].fillna(0.0)

    for col in FEATURES_C:
        df[col] = df[col].fillna(0.0)

    # Compute naive baseline column before splitting
    df["naive_pred"] = compute_naive_baseline(df, PRIMARY_TARGET)

    return df


# ============================================================
# HYPERPARAMETER OPTIMIZATION (TRAIN <= 2022 vs VAL == 2023 ONLY)
# ============================================================
def optimize_model_c(train_df, val_df):
    """
    Search candidate hyperparameters evaluating strictly on Validation (2023) MAE.
    Test set (2024-2025) is NEVER accessed during this search.
    """
    print("\n" + "=" * 70)
    print("MODEL C HYPERPARAMETER OPTIMIZATION (VALIDATION 2023 ONLY)")
    print("=" * 70)

    X_train = train_df[FEATURES_C]
    y_train = train_df[PRIMARY_TARGET]
    X_val = val_df[FEATURES_C]
    y_val = val_df[PRIMARY_TARGET]

    candidate_configs = [
        ("RandomForest", "RF_default", {
            "model_type": "RF",
            "params": {"n_estimators": 300, "max_depth": 15, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": "sqrt", "random_state": RANDOM_STATE, "n_jobs": -1}
        }),
        ("RandomForest", "RF_deep", {
            "model_type": "RF",
            "params": {"n_estimators": 400, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 3, "max_features": "sqrt", "random_state": RANDOM_STATE, "n_jobs": -1}
        }),
        ("RandomForest", "RF_regularized", {
            "model_type": "RF",
            "params": {"n_estimators": 300, "max_depth": 12, "min_samples_split": 20, "min_samples_leaf": 10, "max_features": "sqrt", "random_state": RANDOM_STATE, "n_jobs": -1}
        }),
        ("LightGBM", "LGB_baseline", {
            "model_type": "LGB",
            "params": {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.05, "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1}
        }),
        ("LightGBM", "LGB_tuned", {
            "model_type": "LGB",
            "params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1}
        }),
        ("XGBoost", "XGB_baseline", {
            "model_type": "XGB",
            "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0}
        }),
        ("XGBoost", "XGB_regularized", {
            "model_type": "XGB",
            "params": {"n_estimators": 250, "max_depth": 5, "learning_rate": 0.03, "reg_alpha": 0.1, "reg_lambda": 1.0, "subsample": 0.8, "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0}
        }),
    ]

    tuning_results = []
    best_config = None
    best_val_mae = float("inf")

    for family, name, cfg in candidate_configs:
        m_type = cfg["model_type"]
        params = cfg["params"]

        if m_type == "RF":
            model = RandomForestRegressor(**params)
        elif m_type == "LGB":
            model = lgb.LGBMRegressor(**params)
        elif m_type == "XGB":
            model = xgb.XGBRegressor(**params)

        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        train_preds = model.predict(X_train)

        val_mae = mean_absolute_error(y_val, val_preds)
        val_r2 = r2_score(y_val, val_preds)
        train_mae = mean_absolute_error(y_train, train_preds)

        print(f"  [{family:<12}] {name:<18} -> Val MAE: {val_mae:.4f} | Val R2: {val_r2:.4f} | Train MAE: {train_mae:.4f}")

        tuning_results.append({
            "family": family,
            "name": name,
            "params": {k: v for k, v in params.items() if k not in ["n_jobs", "verbose", "verbosity"]},
            "train_mae": float(train_mae),
            "val_mae": float(val_mae),
            "val_r2": float(val_r2),
        })

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_config = (family, name, cfg, model)

    print(f"\n  Selected Best Model Configuration on Validation (2023): {best_config[0]} ({best_config[1]}) with Val MAE = {best_val_mae:.4f}")
    return best_config, tuning_results


# ============================================================
# PHASE 2: GENERATE MODEL C WALK-FORWARD OOF PREDICTIONS
# ============================================================
def generate_model_c_oof(df, best_family, best_params):
    """
    Generate expanding-window walk-forward predictions for Model C across 2019..2025.
    For each prediction year Y in 2020..2025:
      Train model on Year <= Y-1
      Predict on Year == Y
    2018-2019 cold start marked accordingly.
    """
    print("\n" + "=" * 70)
    print("GENERATING MODEL C OUT-OF-FOLD (OOF) PREDICTIONS")
    print("=" * 70)

    oof_records = []
    years = sorted(df["Year"].unique())

    for pred_year in years:
        pred_mask = (df["Year"] == pred_year)
        pred_rows = df[pred_mask].copy()

        if pred_year == 2018:
            print(f"  Year {pred_year}: COLD-START (no prior data) -> {len(pred_rows):,} rows marked unavailable")
            for _, r in pred_rows.iterrows():
                oof_records.append({
                    "Year": int(r["Year"]),
                    "Month": int(r["Month"]),
                    "Country": r["Country"],
                    "Trade_Type": r["Trade_Type"],
                    "HS4": int(r["HS4"]),
                    "Price_Return_1M_Pred_OOF": np.nan,
                    "Prediction_Value": np.nan,
                    "Price_Return_1M_Actual": float(r[PRIMARY_TARGET]),
                    "Prediction_Year": int(pred_year),
                    "Training_End_Year": None,
                    "Is_Out_Of_Sample": False,
                    "Model_Name": "None (Cold-Start)",
                    "Training_Rows": 0,
                })
            continue

        train_end_year = pred_year - 1
        train_mask = (df["Year"] <= train_end_year)
        train_rows = df[train_mask]

        # Verify hard temporal rule
        assert train_end_year < pred_year, f"VIOLATION: Training_End_Year {train_end_year} >= Prediction_Year {pred_year}"

        X_train = train_rows[FEATURES_C]
        y_train = train_rows[PRIMARY_TARGET]
        X_pred = pred_rows[FEATURES_C]

        if best_family == "RandomForest":
            model = RandomForestRegressor(**best_params)
        elif best_family == "LightGBM":
            model = lgb.LGBMRegressor(**best_params)
        elif best_family == "XGBoost":
            model = xgb.XGBRegressor(**best_params)

        model.fit(X_train, y_train)
        preds = model.predict(X_pred)

        mae = np.mean(np.abs(preds - pred_rows[PRIMARY_TARGET]))
        print(f"  Year {pred_year}: Train <= {train_end_year} ({len(train_rows):,} rows) -> Predict {len(pred_rows):,} rows | OOF MAE={mae:.4f}")

        pred_rows["Price_Return_1M_Pred_OOF"] = preds
        pred_rows["Prediction_Value"] = preds

        for _, r in pred_rows.iterrows():
            oof_records.append({
                "Year": int(r["Year"]),
                "Month": int(r["Month"]),
                "Country": r["Country"],
                "Trade_Type": r["Trade_Type"],
                "HS4": int(r["HS4"]),
                "Price_Return_1M_Pred_OOF": float(r["Price_Return_1M_Pred_OOF"]),
                "Prediction_Value": float(r["Prediction_Value"]),
                "Price_Return_1M_Actual": float(r[PRIMARY_TARGET]),
                "Prediction_Year": int(pred_year),
                "Training_End_Year": int(train_end_year),
                "Is_Out_Of_Sample": True,
                "Model_Name": best_family,
                "Training_Rows": len(train_rows),
            })

    oof_df_c = pd.DataFrame(oof_records)
    out_path_c = os.path.join(RESULTS_DIR, "model_c_predictions_oof.csv")
    oof_df_c.to_csv(out_path_c, index=False)
    # Also save canonical model_c_predictions.csv
    oof_df_c.to_csv(os.path.join(RESULTS_DIR, "model_c_predictions.csv"), index=False)
    print(f"\nModel C OOF predictions artifact saved: {out_path_c} ({len(oof_df_c):,} rows)")
    return oof_df_c


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("Drishti - Task 4: Model C (Price Impact) OOF Retraining")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # 1. Load and prepare data
    df = load_and_prepare_data()

    # 2. Chronological Split
    train_df = df[df["Year"] <= TRAIN_END_YEAR].copy()
    val_df = df[df["Year"] == VAL_YEAR].copy()
    test_df = df[df["Year"] >= TEST_START_YEAR].copy()

    print(f"\nChronological Split:")
    print(f"  Train (<= 2022): {len(train_df):,} rows")
    print(f"  Val   (2023):   {len(val_df):,} rows")
    print(f"  Test  (>= 2024): {len(test_df):,} rows")

    # Compute naive baselines
    bl_val = evaluate_model(val_df[PRIMARY_TARGET], val_df["naive_pred"], "Baseline_val")
    bl_test = evaluate_model(test_df[PRIMARY_TARGET], test_df["naive_pred"], "Baseline_test")

    print(f"\nNaive Baseline:")
    print(f"  Val MAE:  {bl_val['MAE']:.4f} | Val R2:  {bl_val['R2']:.4f}")
    print(f"  Test MAE: {bl_test['MAE']:.4f} | Test R2: {bl_test['R2']:.4f}")

    # 3. Hyperparameter Optimization on Train vs Validation ONLY
    best_config, tuning_results = optimize_model_c(train_df, val_df)
    best_family, best_name, best_cfg, best_val_fitted_model = best_config
    best_params = best_cfg["params"]

    # 4. Final Fit on Train (<= 2022) and Evaluation on Frozen Test (>= 2024)
    print("\n" + "=" * 70)
    print(f"FINAL EVALUATION OF FROZEN MODEL: {best_family} ({best_name})")
    print("=" * 70)

    X_train = train_df[FEATURES_C]
    y_train = train_df[PRIMARY_TARGET]
    X_val = val_df[FEATURES_C]
    y_val = val_df[PRIMARY_TARGET]
    X_test = test_df[FEATURES_C]
    y_test = test_df[PRIMARY_TARGET]

    if best_family == "RandomForest":
        final_model = RandomForestRegressor(**best_params)
    elif best_family == "LightGBM":
        final_model = lgb.LGBMRegressor(**best_params)
    elif best_family == "XGBoost":
        final_model = xgb.XGBRegressor(**best_params)

    final_model.fit(X_train, y_train)

    train_metrics = evaluate_model(y_train, final_model.predict(X_train), f"{best_family}_train")
    val_metrics = evaluate_model(y_val, final_model.predict(X_val), f"{best_family}_val")
    test_metrics = evaluate_model(y_test, final_model.predict(X_test), f"{best_family}_test")

    print(f"  Train: MAE={train_metrics['MAE']:.4f}, RMSE={train_metrics['RMSE']:.4f}, R2={train_metrics['R2']:.4f}")
    print(f"  Val:   MAE={val_metrics['MAE']:.4f}, RMSE={val_metrics['RMSE']:.4f}, R2={val_metrics['R2']:.4f}")
    print(f"  Test:  MAE={test_metrics['MAE']:.4f}, RMSE={test_metrics['RMSE']:.4f}, R2={test_metrics['R2']:.4f}")

    beats_val = val_metrics["MAE"] < bl_val["MAE"]
    beats_test = test_metrics["MAE"] < bl_test["MAE"]
    print(f"\n  Beats Baseline Val:  {beats_val} (MAE: {val_metrics['MAE']:.4f} vs {bl_val['MAE']:.4f})")
    print(f"  Beats Baseline Test: {beats_test} (MAE: {test_metrics['MAE']:.4f} vs {bl_test['MAE']:.4f})")

    # 5. Save Model Artifacts
    canonical_model_path = os.path.join(MODELS_DIR, "model_c_price.joblib")
    oof_model_path = os.path.join(MODELS_DIR, "model_c_price_oof.joblib")
    joblib.dump(final_model, canonical_model_path)
    joblib.dump(final_model, oof_model_path)
    print(f"\n  Saved Model C Model: {canonical_model_path}")

    # 6. Generate Model C Walk-Forward OOF Predictions
    oof_df_c = generate_model_c_oof(df, best_family, best_params)

    # 7. Save Model C OOF Results Summary
    results_c_oof = {
        "task": "Task 4 - Model C Price Impact (OOF Walk-Forward)",
        "timestamp": datetime.now().isoformat(),
        "upstream_artifacts_used": [
            "results/model_a_predictions_oof.csv",
            "results/model_b_predictions_oof.csv"
        ],
        "features": FEATURES_C,
        "primary_target": PRIMARY_TARGET,
        "selected_model": best_name,
        "selected_family": best_family,
        "selected_hyperparameters": {k: v for k, v in best_params.items() if k not in ["n_jobs", "verbose", "verbosity"]},
        "baseline": {"val": bl_val, "test": bl_test},
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "beats_baseline_val": beats_val,
        "beats_baseline_test": beats_test,
        "tuning_results": tuning_results,
        "test_set_isolation": "Verified: Test set (2024-2025) was untouched during hyperparameter search and model selection.",
    }
    results_c_path = os.path.join(RESULTS_DIR, "model_c_results.json")
    results_c_oof_path = os.path.join(RESULTS_DIR, "model_c_results_oof.json")
    with open(results_c_path, "w") as f:
        json.dump(results_c_oof, f, indent=2, default=str)
    with open(results_c_oof_path, "w") as f:
        json.dump(results_c_oof, f, indent=2, default=str)
    print(f"  Saved Model C Results: {results_c_path}")

    print("\n" + "=" * 70)
    print("MODEL C COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
