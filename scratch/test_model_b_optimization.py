"""
Drishti - Model B Optimization Experiment
=========================================
Strictly evaluates on Train (<=2021) vs Validation (2022).
Test set (>=2023) is untouched.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.linear_model import Ridge, HuberRegressor
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MAIN_CSV = os.path.join(DATA_DIR, "Drishti_Cascade_Final_With_EMDAT.csv")
CROP_CSV = os.path.join(DATA_DIR, "Crop_Production_Final.csv")

import sys
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))
from train_model_b_agriculture import (
    build_mapping, aggregate_crop_production, expand_to_monthly,
    merge_with_main, compute_production_risk, clip_target_for_training_and_evaluation,
    TARGET_CLIP_LIMIT, RANDOM_STATE
)

# 1. Load and prepare data
mapping_df = build_mapping()
crop_df = pd.read_csv(CROP_CSV)
national_df = aggregate_crop_production(crop_df, mapping_df)
monthly_crop_df = expand_to_monthly(national_df)
main_df = pd.read_csv(MAIN_CSV)
merged_df = merge_with_main(main_df, monthly_crop_df, mapping_df)
merged_df = compute_production_risk(merged_df)

prod_df = merged_df[merged_df["Has_Production_Data"]].copy()

# Add seasonal features
prod_df["Season_Kharif"] = prod_df["Month"].isin([6, 7, 8, 9, 10]).astype(int)
prod_df["Season_Rabi"] = prod_df["Month"].isin([11, 12, 1, 2, 3]).astype(int)
prod_df["Season_Summer"] = prod_df["Month"].isin([3, 4, 5]).astype(int)

# One-hot encode HS4
hs4_dummies = pd.get_dummies(prod_df["HS4"], prefix="HS4", drop_first=False).astype(int)
prod_df = pd.concat([prod_df, hs4_dummies], axis=1)

# Base features
BASE_FEATURES = [
    "Lagged_Effective_Shock_1",
    "Lagged_Effective_Shock_2",
    "Shock_Intensity_Lag1",
    "Shock_Intensity_Lag2",
    "Trade_Share_Lag1",
    "Trade_Share_Lag2",
    "GPR",
    "INR_USD_Rate",
    "Natural_Disaster_Severity_Index",
    "Season_Kharif",
    "Season_Rabi",
    "Season_Summer",
]

# Additional safe exogenous / lagged disaster / weather features
DISASTER_FEATURES = [
    "Natural_Disaster_Count",
    "Natural_Disaster_Flood_Count",
    "Natural_Disaster_Drought_Count",
    "Natural_Disaster_Affected_Population",
]

MACRO_FEATURES = [
    "Inflation_Lag1",
    "Agri_GVA_Lag1",
    "GDP_Lag1",
    "Price_Lag1",
]

HS4_COLS = hs4_dummies.columns.tolist()

FEATURE_SETS = {
    "Base (Current)": BASE_FEATURES,
    "Base + HS4_Dummies": BASE_FEATURES + HS4_COLS,
    "Base + Disasters + Macro": BASE_FEATURES + DISASTER_FEATURES + MACRO_FEATURES,
    "Base + HS4 + Disasters + Macro": BASE_FEATURES + HS4_COLS + DISASTER_FEATURES + MACRO_FEATURES,
}

# Fill NAs
for col in BASE_FEATURES + DISASTER_FEATURES + MACRO_FEATURES + HS4_COLS:
    if col in prod_df.columns:
        prod_df[col] = prod_df[col].fillna(0.0)

# Splits
train_df = prod_df[prod_df["Year"] <= 2021].copy()
val_df = prod_df[prod_df["Year"] == 2022].copy()

train_t1 = train_df.dropna(subset=["Production_YoY_National"]).copy()
val_t1 = val_df.dropna(subset=["Production_YoY_National"]).copy()

y_train_raw = train_t1["Production_YoY_National"].replace([np.inf, -np.inf], np.nan).fillna(0)
y_train_c = y_train_raw.clip(-TARGET_CLIP_LIMIT, TARGET_CLIP_LIMIT)

y_val_raw = val_t1["Production_YoY_National"].replace([np.inf, -np.inf], np.nan).fillna(0)
y_val_c = y_val_raw.clip(-TARGET_CLIP_LIMIT, TARGET_CLIP_LIMIT)

bl_val_mae = mean_absolute_error(y_val_c, np.zeros(len(y_val_c)))
print(f"\nTarget 1: Production_YoY_National | Val Baseline (Predict 0) MAE: {bl_val_mae:.4f}")

print("\n" + "=" * 80)
print("TESTING CANDIDATE MODELS & FEATURE SETS ON PRODUCTION_YOY_NATIONAL")
print("=" * 80)

for fset_name, fcols in FEATURE_SETS.items():
    print(f"\n--- Feature Set: {fset_name} ({len(fcols)} features) ---")
    X_train = train_t1[fcols]
    X_val = val_t1[fcols]

    models = [
        ("RF_default", RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)),
        ("RF_regularized", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=15, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)),
        ("RF_shallow", RandomForestRegressor(n_estimators=250, max_depth=5, min_samples_leaf=30, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)),
        ("LGB_L2_default", lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ("LGB_L1_mae", lgb.LGBMRegressor(objective="regression_l1", n_estimators=200, max_depth=6, learning_rate=0.03, min_child_samples=30, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ("LGB_Huber", lgb.LGBMRegressor(objective="huber", n_estimators=200, max_depth=6, learning_rate=0.03, min_child_samples=30, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ("XGB_L1_mae", xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=200, max_depth=5, learning_rate=0.03, reg_alpha=1.0, reg_lambda=2.0, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
        ("XGB_Huber", xgb.XGBRegressor(objective="reg:pseudohubererror", n_estimators=200, max_depth=5, learning_rate=0.03, reg_alpha=1.0, reg_lambda=2.0, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
        ("Ridge_reg", Ridge(alpha=100.0)),
        ("Huber_reg", HuberRegressor(max_iter=1000, alpha=100.0)),
    ]

    for m_name, model in models:
        try:
            model.fit(X_train, y_train_c)
            preds_val = model.predict(X_val)
            val_mae = mean_absolute_error(y_val_c, preds_val)
            val_r2 = r2_score(y_val_c, preds_val)
            train_mae = mean_absolute_error(y_train_c, model.predict(X_train))
            beats = val_mae < bl_val_mae
            status = "BEATS" if beats else "LOSES TO"
            diff = bl_val_mae - val_mae
            print(f"  [{status:<8}] {m_name:<18} -> Val MAE: {val_mae:6.2f} (diff: {diff:+6.2f}) | Val R2: {val_r2:+.4f} | Train MAE: {train_mae:6.2f}")
        except Exception as e:
            print(f"  [ERROR   ] {m_name:<18} -> {e}")
