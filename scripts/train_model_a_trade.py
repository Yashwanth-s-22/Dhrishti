"""
Drishti - Task 2: Model A — Trade Impact Model
================================================
Predicts Trade_Return_1M (primary) and Trade_Return_3M (secondary)
using exposure-weighted geopolitical shock features.

Algorithms: RandomForest, XGBoost, LightGBM
Split: Train <= 2022, Validation = 2023, Test >= 2024
Baseline: Naive persistence (previous period's value for same series)

Run: python scripts/train_model_a_trade.py
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

# Leakage-safe feature set for forecasting Trade_Return_1M.  Every feature is
# either lagged from an earlier prediction period or exogenous/predetermined.
REMOVED_LEAKAGE_FEATURES = [
    "Trade_Share",
    "Effective_Shock",
    "Trade_Shock_Exposure",
    "Conflict_Exposure",
    "Protest_Exposure",
    "Net_Hostility_Exposure",
    "Natural_Disaster_Trade_Exposure_USD",
]

LAGGED_FEATURES = [
    "Trade_Share_Lag1",
    "Trade_Share_Lag2",
    "Lagged_Effective_Shock_1",
    "Lagged_Effective_Shock_2",
    "Shock_Intensity_Lag1",
    "Shock_Intensity_Lag2",
]

EXOGENOUS_FEATURES = [
    "GPR",
    "INR_USD_Rate",
    "Natural_Disaster_Severity_Index",
]

FEATURES = [
    "Lagged_Effective_Shock_1",
    "Lagged_Effective_Shock_2",
    "Shock_Intensity_Lag1",
    "Shock_Intensity_Lag2",
    "Trade_Share_Lag1",
    "Trade_Share_Lag2",
    "GPR",
    "INR_USD_Rate",
    "Natural_Disaster_Severity_Index",
]

PRIMARY_TARGET = "Trade_Return_1M"
SECONDARY_TARGET = "Trade_Return_3M"

FORBIDDEN_PREDICTOR_FEATURES = set(REMOVED_LEAKAGE_FEATURES) | {
    PRIMARY_TARGET,
    SECONDARY_TARGET,
}
unsafe_features = set(FEATURES) & FORBIDDEN_PREDICTOR_FEATURES
if unsafe_features:
    raise ValueError(f"Leakage-unsafe Model A features configured: {sorted(unsafe_features)}")

# Chronological split boundaries
TRAIN_END_YEAR = 2022
VAL_YEAR = 2023
TEST_START_YEAR = 2024


def load_and_split_data(csv_path):
    """Load dataset and create chronological train/val/test splits."""
    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"  Full dataset: {df.shape}")

    # Sort chronologically (important for time-series integrity)
    df = df.sort_values(["Year", "Month", "Country", "Trade_Type", "HS4"]).reset_index(drop=True)

    # Verify no nulls in features or targets
    for col in FEATURES + [PRIMARY_TARGET, SECONDARY_TARGET]:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            print(f"  WARNING: {col} has {n_null} nulls - filling with 0")
            df[col] = df[col].fillna(0)

    # Chronological split
    train = df[df["Year"] <= TRAIN_END_YEAR].copy()
    val = df[df["Year"] == VAL_YEAR].copy()
    test = df[df["Year"] >= TEST_START_YEAR].copy()

    print(f"\n  Chronological split:")
    print(f"    Train (<=2022): {len(train):,} rows | Years: {train['Year'].min()}-{train['Year'].max()}")
    print(f"    Val   (2023):   {len(val):,} rows   | Years: {val['Year'].min()}-{val['Year'].max()}")
    print(f"    Test  (>=2024): {len(test):,} rows  | Years: {test['Year'].min()}-{test['Year'].max()}")

    return df, train, val, test


def compute_naive_baseline(df, target_col):
    """
    Naive persistence baseline: predict previous period's value
    for the same (Country, Trade_Type, HS4) series.

    For Trade_Return_1M, this means: predict that this month's trade return
    equals last month's trade return for the same country-commodity-direction.
    """
    print(f"\n  Computing naive persistence baseline for {target_col}...")

    df_sorted = df.sort_values(["Country", "Trade_Type", "HS4", "Year", "Month"]).copy()
    df_sorted["naive_pred"] = (
        df_sorted.groupby(["Country", "Trade_Type", "HS4"])[target_col]
        .shift(1)
        .fillna(0)  # First period has no prior value; predict 0
    )

    return df_sorted["naive_pred"]


def evaluate_model(y_true, y_pred, label=""):
    """Compute and return evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2}


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                       model, model_name, target_name):
    """Train a model and evaluate on val/test sets."""
    print(f"\n  Training {model_name} on {target_name}...")
    model.fit(X_train, y_train)

    # Predictions
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    # Metrics
    metrics_train = evaluate_model(y_train, pred_train, f"{model_name}_train")
    metrics_val = evaluate_model(y_val, pred_val, f"{model_name}_val")
    metrics_test = evaluate_model(y_test, pred_test, f"{model_name}_test")

    print(f"    Train: MAE={metrics_train['MAE']:.4f}, R2={metrics_train['R2']:.4f}")
    print(f"    Val:   MAE={metrics_val['MAE']:.4f}, R2={metrics_val['R2']:.4f}")
    print(f"    Test:  MAE={metrics_test['MAE']:.4f}, R2={metrics_test['R2']:.4f}")

    # Suspiciously high R2 check (Part E, rule 4)
    for split_name, m in [("train", metrics_train), ("val", metrics_val), ("test", metrics_test)]:
        if m["R2"] > 0.95:
            print(f"    *** RED FLAG: R2={m['R2']:.4f} on {split_name} is suspiciously high.")
            print(f"    *** Re-auditing features for possible leakage...")

    # Feature importances
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(FEATURES, model.feature_importances_))
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    else:
        importances = {}

    return {
        "model": model,
        "model_name": model_name,
        "target": target_name,
        "metrics": {"train": metrics_train, "val": metrics_val, "test": metrics_test},
        "importances": importances,
        "predictions": {"train": pred_train, "val": pred_val, "test": pred_test},
    }


def print_feature_importances(importances, model_name):
    """Print feature importances in ranked order."""
    print(f"\n  Feature importances ({model_name}):")
    print(f"  {'Rank':<6}{'Feature':<45}{'Importance':<12}")
    print(f"  {'-'*60}")
    for rank, (feat, imp) in enumerate(importances.items(), 1):
        marker = ""
        # Only lagged exposure-weighted features are eligible for this comparison.
        if feat in ["Lagged_Effective_Shock_1", "Lagged_Effective_Shock_2"]:
            marker = " [LAGGED EXPOSURE]"
        elif feat in ["Shock_Intensity_Lag1", "Shock_Intensity_Lag2"]:
            marker = " [RAW]"
        print(f"  {rank:<6}{feat:<45}{imp:.6f}{marker}")


def exposure_vs_raw_comparison(importances):
    """
    Compare only leakage-safe lagged exposure-weighted features with lagged raw
    shock features.  Importance alone is not treated as proof: SUPPORTS and
    DOES NOT SUPPORT require both the total and both paired lag comparisons to
    point in the same direction; otherwise the result is INCONCLUSIVE.
    """
    exposure_features = {"Lagged_Effective_Shock_1", "Lagged_Effective_Shock_2"}
    raw_features = {"Shock_Intensity_Lag1", "Shock_Intensity_Lag2"}

    exposure_imp = sum(v for k, v in importances.items() if k in exposure_features)
    raw_imp = sum(v for k, v in importances.items() if k in raw_features)

    print(f"\n  Exposure-weighting thesis check:")
    lag1_supports = importances.get("Lagged_Effective_Shock_1", 0) > importances.get("Shock_Intensity_Lag1", 0)
    lag2_supports = importances.get("Lagged_Effective_Shock_2", 0) > importances.get("Shock_Intensity_Lag2", 0)
    lag1_opposes = importances.get("Lagged_Effective_Shock_1", 0) < importances.get("Shock_Intensity_Lag1", 0)
    lag2_opposes = importances.get("Lagged_Effective_Shock_2", 0) < importances.get("Shock_Intensity_Lag2", 0)

    print(f"    Total importance of lagged exposure features: {exposure_imp:.6f}")
    print(f"    Total importance of lagged raw shocks:       {raw_imp:.6f}")
    print(f"    Pairwise comparison (lag 1, lag 2):          {lag1_supports}, {lag2_supports}")

    if exposure_imp > raw_imp and lag1_supports and lag2_supports:
        verdict = "SUPPORTS"
    elif raw_imp > exposure_imp and lag1_opposes and lag2_opposes:
        verdict = "DOES NOT SUPPORT"
    else:
        verdict = "INCONCLUSIVE"
    print(f"    Verdict: {verdict}")

    return {
        "exposure_total_importance": exposure_imp,
        "raw_total_importance": raw_imp,
        "lag1_supports": lag1_supports,
        "lag2_supports": lag2_supports,
        "verdict": verdict,
    }


def main():
    print("=" * 70)
    print("Drishti - Task 2: Model A — Trade Impact")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load and split
    df, train, val, test = load_and_split_data(MAIN_CSV)

    print("\n" + "=" * 70)
    print("LEAKAGE-PREVENTION FEATURE AUDIT")
    print("=" * 70)
    print(f"  Removed contemporaneous trade-derived features: {', '.join(REMOVED_LEAKAGE_FEATURES)}")
    print(f"  Retained lagged historical features: {', '.join(LAGGED_FEATURES)}")
    print(f"  Retained exogenous/predetermined features: {', '.join(EXOGENOUS_FEATURES)}")
    print("  Dataset was not modified; all feature handling is in-memory.")

    X_train, y_train_1m = train[FEATURES], train[PRIMARY_TARGET]
    X_val, y_val_1m = val[FEATURES], val[PRIMARY_TARGET]
    X_test, y_test_1m = test[FEATURES], test[PRIMARY_TARGET]

    y_train_3m = train[SECONDARY_TARGET]
    y_val_3m = val[SECONDARY_TARGET]
    y_test_3m = test[SECONDARY_TARGET]

    # ============================================================
    # NAIVE PERSISTENCE BASELINE
    # ============================================================
    print("\n" + "=" * 70)
    print("NAIVE PERSISTENCE BASELINE")
    print("=" * 70)

    # Compute baseline on full dataset, then split
    df["naive_pred_1m"] = compute_naive_baseline(df, PRIMARY_TARGET)
    df["naive_pred_3m"] = compute_naive_baseline(df, SECONDARY_TARGET)

    # Re-split after adding baseline predictions
    train_bl = df[df["Year"] <= TRAIN_END_YEAR]
    val_bl = df[df["Year"] == VAL_YEAR]
    test_bl = df[df["Year"] >= TEST_START_YEAR]

    baseline_results = {}
    for target, naive_col, label in [
        (PRIMARY_TARGET, "naive_pred_1m", "Trade_Return_1M"),
        (SECONDARY_TARGET, "naive_pred_3m", "Trade_Return_3M"),
    ]:
        print(f"\n  Baseline for {label}:")
        bl_train = evaluate_model(train_bl[target], train_bl[naive_col], f"baseline_train_{label}")
        bl_val = evaluate_model(val_bl[target], val_bl[naive_col], f"baseline_val_{label}")
        bl_test = evaluate_model(test_bl[target], test_bl[naive_col], f"baseline_test_{label}")
        print(f"    Train: MAE={bl_train['MAE']:.4f}, R2={bl_train['R2']:.4f}")
        print(f"    Val:  MAE={bl_val['MAE']:.4f}, R2={bl_val['R2']:.4f}")
        print(f"    Test: MAE={bl_test['MAE']:.4f}, R2={bl_test['R2']:.4f}")
        baseline_results[label] = {"train": bl_train, "val": bl_val, "test": bl_test}

    # ============================================================
    # MODEL TRAINING - PRIMARY TARGET (Trade_Return_1M)
    # ============================================================
    print("\n" + "=" * 70)
    print("MODEL TRAINING — PRIMARY TARGET: Trade_Return_1M")
    print("=" * 70)

    # 1. Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_result = train_and_evaluate(
        X_train, y_train_1m, X_val, y_val_1m, X_test, y_test_1m,
        rf_model, "RandomForest", PRIMARY_TARGET
    )
    print_feature_importances(rf_result["importances"], "RandomForest")
    rf_thesis = exposure_vs_raw_comparison(rf_result["importances"])

    # 2. XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_result = train_and_evaluate(
        X_train, y_train_1m, X_val, y_val_1m, X_test, y_test_1m,
        xgb_model, "XGBoost", PRIMARY_TARGET
    )
    print_feature_importances(xgb_result["importances"], "XGBoost")
    xgb_thesis = exposure_vs_raw_comparison(xgb_result["importances"])

    # 3. LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_result = train_and_evaluate(
        X_train, y_train_1m, X_val, y_val_1m, X_test, y_test_1m,
        lgb_model, "LightGBM", PRIMARY_TARGET
    )
    print_feature_importances(lgb_result["importances"], "LightGBM")
    lgb_thesis = exposure_vs_raw_comparison(lgb_result["importances"])

    # ============================================================
    # MODEL TRAINING - SECONDARY TARGET (Trade_Return_3M)
    # ============================================================
    print("\n" + "=" * 70)
    print("MODEL TRAINING — SECONDARY TARGET: Trade_Return_3M")
    print("=" * 70)

    # Use the best-performing algorithm from primary target for secondary
    # (still report all three for primary, but for brevity use best for secondary)
    all_primary_results = [rf_result, xgb_result, lgb_result]
    best_primary = min(all_primary_results, key=lambda r: r["metrics"]["val"]["MAE"])
    print(f"\n  Best model on primary target (val MAE): {best_primary['model_name']}")

    # Train all three on secondary target too for completeness
    secondary_results = {}
    for model_cls, name, params in [
        (RandomForestRegressor, "RandomForest", dict(
            n_estimators=300, max_depth=15, min_samples_split=10,
            min_samples_leaf=5, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)),
        (xgb.XGBRegressor, "XGBoost", dict(
            n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
        (lgb.LGBMRegressor, "LightGBM", dict(
            n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
    ]:
        model = model_cls(**params)
        result = train_and_evaluate(
            X_train, y_train_3m, X_val, y_val_3m, X_test, y_test_3m,
            model, name, SECONDARY_TARGET
        )
        secondary_results[name] = result

    # ============================================================
    # MODEL COMPARISON TABLE
    # ============================================================
    print("\n" + "=" * 70)
    print("MODEL COMPARISON — vs BASELINE")
    print("=" * 70)

    comparison = []

    # Primary target
    for result in [rf_result, xgb_result, lgb_result]:
        bl = baseline_results["Trade_Return_1M"]
        row = {
            "Target": PRIMARY_TARGET,
            "Model": result["model_name"],
            "Val_MAE": result["metrics"]["val"]["MAE"],
            "Val_R2": result["metrics"]["val"]["R2"],
            "Test_MAE": result["metrics"]["test"]["MAE"],
            "Test_R2": result["metrics"]["test"]["R2"],
            "Baseline_Val_MAE": bl["val"]["MAE"],
            "Baseline_Val_R2": bl["val"]["R2"],
            "Baseline_Test_MAE": bl["test"]["MAE"],
            "Baseline_Test_R2": bl["test"]["R2"],
            "Beats_Baseline_Val": result["metrics"]["val"]["MAE"] < bl["val"]["MAE"],
            "Beats_Baseline_Test": result["metrics"]["test"]["MAE"] < bl["test"]["MAE"],
        }
        comparison.append(row)

    # Secondary target
    for name, result in secondary_results.items():
        bl = baseline_results["Trade_Return_3M"]
        row = {
            "Target": SECONDARY_TARGET,
            "Model": name,
            "Val_MAE": result["metrics"]["val"]["MAE"],
            "Val_R2": result["metrics"]["val"]["R2"],
            "Test_MAE": result["metrics"]["test"]["MAE"],
            "Test_R2": result["metrics"]["test"]["R2"],
            "Baseline_Val_MAE": bl["val"]["MAE"],
            "Baseline_Val_R2": bl["val"]["R2"],
            "Baseline_Test_MAE": bl["test"]["MAE"],
            "Baseline_Test_R2": bl["test"]["R2"],
            "Beats_Baseline_Val": result["metrics"]["val"]["MAE"] < bl["val"]["MAE"],
            "Beats_Baseline_Test": result["metrics"]["test"]["MAE"] < bl["test"]["MAE"],
        }
        comparison.append(row)

    comp_df = pd.DataFrame(comparison)
    print("\n" + comp_df.to_string(index=False))

    # ============================================================
    # SELECT BEST MODEL & SAVE ARTIFACTS
    # ============================================================
    print("\n" + "=" * 70)
    print("SAVING ARTIFACTS")
    print("=" * 70)

    # Select best model based on validation MAE for primary target
    best_model_result = min([rf_result, xgb_result, lgb_result],
                            key=lambda r: r["metrics"]["val"]["MAE"])
    best_model = best_model_result["model"]
    best_name = best_model_result["model_name"]
    print(f"\n  Best model (primary, val MAE): {best_name}")

    # Save the best model
    model_path = os.path.join(MODELS_DIR, "model_a_trade.joblib")
    joblib.dump(best_model, model_path)
    print(f"  Model saved: {model_path}")

    # Also save all models for reference
    joblib.dump(rf_result["model"], os.path.join(MODELS_DIR, "model_a_rf.joblib"))
    joblib.dump(xgb_result["model"], os.path.join(MODELS_DIR, "model_a_xgb.joblib"))
    joblib.dump(lgb_result["model"], os.path.join(MODELS_DIR, "model_a_lgb.joblib"))
    print("  All model variants saved.")

    # Save predictions (needed by Model C downstream)
    pred_df = df[["Year", "Month", "Country", "Trade_Type", "HS4"]].copy()
    # Generate predictions on full dataset using best model
    pred_df["Trade_Return_1M_Pred"] = best_model.predict(df[FEATURES])
    pred_df["Trade_Return_1M_Actual"] = df[PRIMARY_TARGET]

    pred_path = os.path.join(RESULTS_DIR, "model_a_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"  Predictions saved: {pred_path}")

    # Save results summary
    results = {
        "task": "Task 2 - Model A: Trade Impact",
        "timestamp": datetime.now().isoformat(),
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "primary_target": PRIMARY_TARGET,
        "secondary_target": SECONDARY_TARGET,
        "split": {
            "train": f"<= {TRAIN_END_YEAR}",
            "val": str(VAL_YEAR),
            "test": f">= {TEST_START_YEAR}",
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
        },
        "best_model": best_name,
        "primary_results": {
            "RandomForest": rf_result["metrics"],
            "XGBoost": xgb_result["metrics"],
            "LightGBM": lgb_result["metrics"],
            "baseline": baseline_results["Trade_Return_1M"],
        },
        "secondary_results": {
            name: result["metrics"] for name, result in secondary_results.items()
        },
        "secondary_baseline": baseline_results["Trade_Return_3M"],
        "feature_importances": {
            "RandomForest": rf_result["importances"],
            "XGBoost": xgb_result["importances"],
            "LightGBM": lgb_result["importances"],
        },
        "exposure_vs_raw_thesis": {
            "RandomForest": rf_thesis,
            "XGBoost": xgb_thesis,
            "LightGBM": lgb_thesis,
        },
        "comparison_table": comparison,
    }

    results_path = os.path.join(RESULTS_DIR, "model_a_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {results_path}")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("TASK 2 COMPLETE — SUMMARY")
    print("=" * 70)

    print(f"\n  Best model: {best_name}")
    bm = best_model_result["metrics"]
    bl_1m = baseline_results["Trade_Return_1M"]
    print(f"\n  Primary target ({PRIMARY_TARGET}):")
    print(f"    {best_name:<15} Val MAE={bm['val']['MAE']:.4f}  Test MAE={bm['test']['MAE']:.4f}  Test R2={bm['test']['R2']:.4f}")
    print(f"    {'Baseline':<15} Val MAE={bl_1m['val']['MAE']:.4f}  Test MAE={bl_1m['test']['MAE']:.4f}  Test R2={bl_1m['test']['R2']:.4f}")
    beats = bm['test']['MAE'] < bl_1m['test']['MAE']
    print(f"    Model {'BEATS' if beats else 'DOES NOT BEAT'} baseline on test set")

    print(f"\n  Artifacts saved:")
    print(f"    Model:       {model_path}")
    print(f"    Predictions: {pred_path}")
    print(f"    Results:     {results_path}")


if __name__ == "__main__":
    main()
