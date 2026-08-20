"""
Drishti - Model B Investigation & Optimization Experiment
=========================================================
Strictly evaluates on Train (<=2021) and Validation (2022) only.
Test set (>=2023) is untouched during model search.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.linear_model import Ridge, HuberRegressor, QuantileRegressor
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MAIN_CSV = os.path.join(DATA_DIR, "Drishti_Cascade_Final_With_EMDAT.csv")
CROP_CSV = os.path.join(DATA_DIR, "Crop_Production_Final.csv")

# Load scripts/train_model_b_agriculture.py functions
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

# Add seasonal dummies
prod_df["Season_Kharif"] = prod_df["Month"].isin([6, 7, 8, 9, 10]).astype(int)
prod_df["Season_Rabi"] = prod_df["Month"].isin([11, 12, 1, 2, 3]).astype(int)
prod_df["Season_Summer"] = prod_df["Month"].isin([3, 4, 5]).astype(int)

# Inspect available columns
print("\nAll columns in prod_df:")
print(prod_df.columns.tolist())

# Splits
train_df = prod_df[prod_df["Year"] <= 2021].copy()
val_df = prod_df[prod_df["Year"] == 2022].copy()
test_df = prod_df[prod_df["Year"] >= 2023].copy()

print(f"\nSplit sizes: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

# Let's inspect target stats
y_train_raw = train_df["Production_YoY_National"].dropna()
y_val_raw = val_df["Production_YoY_National"].dropna()
print(f"\nTrain YoY: count={len(y_train_raw)}, mean={y_train_raw.mean():.2f}, median={y_train_raw.median():.2f}, std={y_train_raw.std():.2f}, min={y_train_raw.min():.2f}, max={y_train_raw.max():.2f}")
print(f"Val YoY: count={len(y_val_raw)}, mean={y_val_raw.mean():.2f}, median={y_val_raw.median():.2f}, std={y_val_raw.std():.2f}, min={y_val_raw.min():.2f}, max={y_val_raw.max():.2f}")

# Baselines on Val
y_train_c = clip_target_for_training_and_evaluation(train_df["Production_YoY_National"].dropna(), "YoY", "train")
val_valid = val_df.dropna(subset=["Production_YoY_National"]).copy()
y_val_c = clip_target_for_training_and_evaluation(val_valid["Production_YoY_National"], "YoY", "val")

bl_zero_mae = mean_absolute_error(y_val_c, np.zeros(len(y_val_c)))
bl_mean_mae = mean_absolute_error(y_val_c, np.full(len(y_val_c), y_train_c.mean()))
bl_median_mae = mean_absolute_error(y_val_c, np.full(len(y_val_c), y_train_c.median()))

print(f"\nBaselines on Val:")
print(f"  Predict 0:      MAE = {bl_zero_mae:.4f}")
print(f"  Predict Mean:   MAE = {bl_mean_mae:.4f} (train mean={y_train_c.mean():.2f})")
print(f"  Predict Median: MAE = {bl_median_mae:.4f} (train median={y_train_c.median():.2f})")
