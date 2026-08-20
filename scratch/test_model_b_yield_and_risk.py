"""
Drishti - Model B Yield and Risk Optimization Experiment
========================================================
Strictly evaluates on Train (<=2021) vs Validation (2022).
Test set (>=2023) is untouched.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, classification_report
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

# 1. Prepare data
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

# Features
FEATURES_B = [
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

for col in FEATURES_B:
    prod_df[col] = prod_df[col].fillna(0.0)

# Splits
train_df = prod_df[prod_df["Year"] <= 2021].copy()
val_df = prod_df[prod_df["Year"] == 2022].copy()

# ----------------------------------------------------
# TARGET 2: YIELD_YOY_NATIONAL
# ----------------------------------------------------
print("=" * 80)
print("EXPERIMENT: TARGET 2 - YIELD_YOY_NATIONAL")
print("=" * 80)
train_t2 = train_df.dropna(subset=["Yield_YoY_National"]).copy()
val_t2 = val_df.dropna(subset=["Yield_YoY_National"]).copy()

y_train_y = clip_target_for_training_and_evaluation(train_t2["Yield_YoY_National"], "Yield_YoY", "train")
y_val_y = clip_target_for_training_and_evaluation(val_t2["Yield_YoY_National"], "Yield_YoY", "val")

bl_yield_val_mae = mean_absolute_error(y_val_y, np.zeros(len(y_val_y)))
print(f"Val Baseline (Predict 0) MAE: {bl_yield_val_mae:.4f}")

X_train_y = train_t2[FEATURES_B]
X_val_y = val_t2[FEATURES_B]

yield_models = [
    ("RF_default", RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)),
    ("RF_regularized", RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=20, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)),
    ("LGB_default", lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
    ("LGB_L1", lgb.LGBMRegressor(objective="regression_l1", n_estimators=200, max_depth=5, learning_rate=0.03, min_child_samples=30, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
    ("XGB_L1", xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=200, max_depth=4, learning_rate=0.03, reg_alpha=2.0, reg_lambda=2.0, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
]

for m_name, model in yield_models:
    model.fit(X_train_y, y_train_y)
    preds = model.predict(X_val_y)
    val_mae = mean_absolute_error(y_val_y, preds)
    beats = val_mae < bl_yield_val_mae
    print(f"  [{'BEATS' if beats else 'LOSES TO':<8}] {m_name:<16} -> Val MAE: {val_mae:.4f} (bl={bl_yield_val_mae:.4f})")

# ----------------------------------------------------
# TARGET 3: PRODUCTION_RISK (CLASSIFICATION)
# ----------------------------------------------------
print("\n" + "=" * 80)
print("EXPERIMENT: TARGET 3 - PRODUCTION_RISK CLASSIFIER")
print("=" * 80)
risk_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
train_t3 = train_df.dropna(subset=["Production_Risk"]).copy()
val_t3 = val_df.dropna(subset=["Production_Risk"]).copy()

y_train_r = train_t3["Production_Risk"].map(risk_map)
y_val_r = val_t3["Production_Risk"].map(risk_map)

# Majority class baseline
majority_class = y_train_r.mode()[0]
maj_val_preds = np.full(len(y_val_r), majority_class)
bl_risk_acc = accuracy_score(y_val_r, maj_val_preds)
bl_risk_f1 = f1_score(y_val_r, maj_val_preds, average="weighted", zero_division=0)
print(f"Val Baseline (Majority Class {majority_class}) -> Accuracy: {bl_risk_acc:.4f} | F1: {bl_risk_f1:.4f}")

X_train_r = train_t3[FEATURES_B]
X_val_r = val_t3[FEATURES_B]

risk_models = [
    ("RF_default", RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)),
    ("RF_balanced", RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ("LGB_default", lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
    ("LGB_tuned", lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
    ("XGB_default", xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
    ("XGB_regularized", xgb.XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.03, reg_alpha=1.0, reg_lambda=2.0, subsample=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
]

for m_name, model in risk_models:
    model.fit(X_train_r, y_train_r)
    preds = model.predict(X_val_r)
    acc = accuracy_score(y_val_r, preds)
    f1 = f1_score(y_val_r, preds, average="weighted", zero_division=0)
    beats = acc > bl_risk_acc
    print(f"  [{'BEATS' if beats else 'LOSES TO':<8}] {m_name:<16} -> Val Acc: {acc:.4f} ({acc*100:.1f}%) | Val F1: {f1:.4f} | (vs bl Acc {bl_risk_acc:.4f})")
