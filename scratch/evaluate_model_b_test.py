"""
Drishti - Model B Test Evaluation & Comparison
=============================================
Evaluates the best validation-selected models on the frozen Test set (>=2023).
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

# 1. Load data
mapping_df = build_mapping()
crop_df = pd.read_csv(CROP_CSV)
national_df = aggregate_crop_production(crop_df, mapping_df)
monthly_crop_df = expand_to_monthly(national_df)
main_df = pd.read_csv(MAIN_CSV)
merged_df = merge_with_main(main_df, monthly_crop_df, mapping_df)
merged_df = compute_production_risk(merged_df)

prod_df = merged_df[merged_df["Has_Production_Data"]].copy()

prod_df["Season_Kharif"] = prod_df["Month"].isin([6, 7, 8, 9, 10]).astype(int)
prod_df["Season_Rabi"] = prod_df["Month"].isin([11, 12, 1, 2, 3]).astype(int)
prod_df["Season_Summer"] = prod_df["Month"].isin([3, 4, 5]).astype(int)

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

train_df = prod_df[prod_df["Year"] <= 2021].copy()
val_df = prod_df[prod_df["Year"] == 2022].copy()
test_df = prod_df[prod_df["Year"] >= 2023].copy()

print(f"Data counts: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

# --- TARGET 1: Production_YoY_National ---
train_t1 = train_df.dropna(subset=["Production_YoY_National"])
val_t1 = val_df.dropna(subset=["Production_YoY_National"])
test_t1 = test_df.dropna(subset=["Production_YoY_National"])

y_train_p = clip_target_for_training_and_evaluation(train_t1["Production_YoY_National"], "Prod_YoY", "train")
y_val_p = clip_target_for_training_and_evaluation(val_t1["Production_YoY_National"], "Prod_YoY", "val")
y_test_p = clip_target_for_training_and_evaluation(test_t1["Production_YoY_National"], "Prod_YoY", "test")

bl_p_val = mean_absolute_error(y_val_p, np.zeros(len(y_val_p)))
bl_p_test = mean_absolute_error(y_test_p, np.zeros(len(y_test_p)))

m_p_xgb = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=200, max_depth=5, learning_rate=0.03, reg_alpha=1.0, reg_lambda=2.0, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
m_p_xgb.fit(train_t1[FEATURES_B], y_train_p)

m_p_rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
m_p_rf.fit(train_t1[FEATURES_B], y_train_p)

m_p_lgb = lgb.LGBMRegressor(objective="regression_l1", n_estimators=200, max_depth=6, learning_rate=0.03, min_child_samples=30, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
m_p_lgb.fit(train_t1[FEATURES_B], y_train_p)

print("\n--- Production_YoY_National ---")
print(f"  Baseline (predict 0) : Val MAE = {bl_p_val:.4f} | Test MAE = {bl_p_test:.4f}")
for name, m in [("XGB_L1", m_p_xgb), ("RF_default", m_p_rf), ("LGB_L1", m_p_lgb)]:
    v_mae = mean_absolute_error(y_val_p, m.predict(val_t1[FEATURES_B]))
    t_mae = mean_absolute_error(y_test_p, m.predict(test_t1[FEATURES_B]))
    t_r2 = r2_score(y_test_p, m.predict(test_t1[FEATURES_B]))
    print(f"  {name:<12}       : Val MAE = {v_mae:.4f} (diff: {bl_p_val-v_mae:+.4f}) | Test MAE = {t_mae:.4f} (diff: {bl_p_test-t_mae:+.4f}, R2={t_r2:.4f})")

# --- TARGET 2: Yield_YoY_National ---
train_t2 = train_df.dropna(subset=["Yield_YoY_National"])
val_t2 = val_df.dropna(subset=["Yield_YoY_National"])
test_t2 = test_df.dropna(subset=["Yield_YoY_National"])

y_train_y = clip_target_for_training_and_evaluation(train_t2["Yield_YoY_National"], "Yield_YoY", "train")
y_val_y = clip_target_for_training_and_evaluation(val_t2["Yield_YoY_National"], "Yield_YoY", "val")
y_test_y = clip_target_for_training_and_evaluation(test_t2["Yield_YoY_National"], "Yield_YoY", "test")

bl_y_val = mean_absolute_error(y_val_y, np.zeros(len(y_val_y)))
bl_y_test = mean_absolute_error(y_test_y, np.zeros(len(y_test_y)))

m_y_xgb = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=200, max_depth=4, learning_rate=0.03, reg_alpha=2.0, reg_lambda=2.0, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
m_y_xgb.fit(train_t2[FEATURES_B], y_train_y)

m_y_rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
m_y_rf.fit(train_t2[FEATURES_B], y_train_y)

m_y_lgb = lgb.LGBMRegressor(objective="regression_l1", n_estimators=200, max_depth=5, learning_rate=0.03, min_child_samples=30, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
m_y_lgb.fit(train_t2[FEATURES_B], y_train_y)

print("\n--- Yield_YoY_National ---")
print(f"  Baseline (predict 0) : Val MAE = {bl_y_val:.4f} | Test MAE = {bl_y_test:.4f}")
for name, m in [("XGB_L1", m_y_xgb), ("RF_default", m_y_rf), ("LGB_L1", m_y_lgb)]:
    v_mae = mean_absolute_error(y_val_y, m.predict(val_t2[FEATURES_B]))
    t_mae = mean_absolute_error(y_test_y, m.predict(test_t2[FEATURES_B]))
    t_r2 = r2_score(y_test_y, m.predict(test_t2[FEATURES_B]))
    print(f"  {name:<12}       : Val MAE = {v_mae:.4f} (diff: {bl_y_val-v_mae:+.4f}) | Test MAE = {t_mae:.4f} (diff: {bl_y_test-t_mae:+.4f}, R2={t_r2:.4f})")

# --- TARGET 3: Production_Risk (Classification) ---
risk_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
train_t3 = train_df.dropna(subset=["Production_Risk"])
val_t3 = val_df.dropna(subset=["Production_Risk"])
test_t3 = test_df.dropna(subset=["Production_Risk"])

y_train_r = train_t3["Production_Risk"].map(risk_map)
y_val_r = val_t3["Production_Risk"].map(risk_map)
y_test_r = test_t3["Production_Risk"].map(risk_map)

majority_class = y_train_r.mode()[0]
bl_r_val_acc = accuracy_score(y_val_r, np.full(len(y_val_r), majority_class))
bl_r_test_acc = accuracy_score(y_test_r, np.full(len(y_test_r), majority_class))

m_r_lgb = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
m_r_lgb.fit(train_t3[FEATURES_B], y_train_r)

m_r_xgb = xgb.XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.03, reg_alpha=1.0, reg_lambda=2.0, subsample=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
m_r_xgb.fit(train_t3[FEATURES_B], y_train_r)

m_r_rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
m_r_rf.fit(train_t3[FEATURES_B], y_train_r)

print("\n--- Production_Risk (Classification) ---")
print(f"  Baseline (Majority Class) : Val Acc = {bl_r_val_acc:.4f} | Test Acc = {bl_r_test_acc:.4f}")
for name, m in [("LGB_tuned", m_r_lgb), ("XGB_regularized", m_r_xgb), ("RF_default", m_r_rf)]:
    v_acc = accuracy_score(y_val_r, m.predict(val_t3[FEATURES_B]))
    t_acc = accuracy_score(y_test_r, m.predict(test_t3[FEATURES_B]))
    t_f1 = f1_score(y_test_r, m.predict(test_t3[FEATURES_B]), average="weighted", zero_division=0)
    print(f"  {name:<16}       : Val Acc = {v_acc:.4f} ({v_acc*100:.1f}%) | Test Acc = {t_acc:.4f} ({t_acc*100:.1f}%), F1={t_f1:.4f}")
