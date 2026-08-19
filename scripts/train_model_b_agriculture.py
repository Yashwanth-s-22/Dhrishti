"""
Drishti - Task 3: Crop-to-HS4 Bridge + Model B (Agricultural Impact)
=====================================================================
1. Use the reviewed crop-to-HS4 mapping (in memory)
2. Aggregate crop production to national level
3. Expand season-level to month-level
4. Merge into main dataset
5. Train Model B on production targets

Run: python scripts/train_model_b_agriculture.py
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, r2_score, mean_squared_error,
                             accuracy_score, classification_report, f1_score)
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
CROP_CSV = os.path.join(DATA_DIR, "Crop_Production_Final.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# STEP 1: BUILD CROP-TO-HS4 MAPPING
# ============================================================
# Manual mapping verified by inspecting both datasets.
# Only crops with clear, unambiguous matches to HS chapters 1-20 are included.
# Rationale for each match documented inline.
#
# SCOPE LIMITATION: Cotton (lint) is HS ch.52, outside our ch.1-20 data.
# Many crops (arecanut, guar seed, niger seed, etc.) have no clear HS4 match
# in this dataset's commodity descriptions.

CROP_HS4_MAPPING = [
    # ---- Cereals (HS ch.10) ----
    {"Crop": "Rice", "HS4": 1006, "Match_Quality": "exact",
     "Notes": "HS 1006 = Rice; semi/wholly milled"},
    {"Crop": "Wheat", "HS4": 1001, "Match_Quality": "exact",
     "Notes": "HS 1001 = Wheat and meslin"},
    {"Crop": "Maize", "HS4": 1005, "Match_Quality": "exact",
     "Notes": "HS 1005 = Maize (corn)"},
    {"Crop": "Barley", "HS4": 1003, "Match_Quality": "exact",
     "Notes": "HS 1003 = Barley"},
    {"Crop": "Jowar", "HS4": 1007, "Match_Quality": "exact",
     "Notes": "HS 1007 = Grain sorghum; Jowar is sorghum"},
    {"Crop": "Bajra", "HS4": 1008, "Match_Quality": "close",
     "Notes": "HS 1008 = Buckwheat, millet and canary seeds; Bajra is pearl millet"},

    # ---- Pulses (HS ch.7, dried legumes at 0713) ----
    {"Crop": "Gram", "HS4": 713, "Match_Quality": "close",
     "Notes": "HS 0713 = Dried leguminous vegetables; chickpea/gram is a major subcategory"},
    {"Crop": "Arhar/Tur", "HS4": 713, "Match_Quality": "close",
     "Notes": "HS 0713 covers pigeon pea (arhar/tur)"},
    {"Crop": "Moong(Green Gram)", "HS4": 713, "Match_Quality": "close",
     "Notes": "HS 0713 covers mung bean"},
    {"Crop": "Urad", "HS4": 713, "Match_Quality": "close",
     "Notes": "HS 0713 covers urad (black gram)"},
    {"Crop": "Masoor", "HS4": 713, "Match_Quality": "close",
     "Notes": "HS 0713 covers lentils (masoor)"},

    # ---- Vegetables (HS ch.7) ----
    {"Crop": "Onion", "HS4": 703, "Match_Quality": "exact",
     "Notes": "HS 0703 = Onions, shallots, garlic, leeks"},
    {"Crop": "Garlic", "HS4": 703, "Match_Quality": "exact",
     "Notes": "HS 0703 covers garlic (same HS4 as onion)"},
    {"Crop": "Potato", "HS4": 701, "Match_Quality": "exact",
     "Notes": "HS 0701 = Potatoes; fresh or chilled"},
    {"Crop": "Sweet potato", "HS4": 714, "Match_Quality": "exact",
     "Notes": "HS 0714 = Manioc, sweet potatoes and similar roots"},
    {"Crop": "Tapioca", "HS4": 714, "Match_Quality": "exact",
     "Notes": "HS 0714 = Manioc (cassava/tapioca)"},

    # ---- Oilseeds (HS ch.12) ----
    {"Crop": "Soyabean", "HS4": 1201, "Match_Quality": "exact",
     "Notes": "HS 1201 = Soya beans"},
    {"Crop": "Groundnut", "HS4": 1202, "Match_Quality": "exact",
     "Notes": "HS 1202 = Ground-nuts (peanuts)"},
    {"Crop": "Sesamum", "HS4": 1207, "Match_Quality": "close",
     "Notes": "HS 1207 = Other oil seeds; includes sesame"},
    {"Crop": "Linseed", "HS4": 1204, "Match_Quality": "exact",
     "Notes": "HS 1204 = Linseed (flax seed)"},
    {"Crop": "Rapeseed &Mustard", "HS4": 1205, "Match_Quality": "exact",
     "Notes": "HS 1205 = Rape or colza seeds"},
    {"Crop": "Sunflower", "HS4": 1206, "Match_Quality": "exact",
     "Notes": "HS 1206 = Sunflower seeds"},
    {"Crop": "Castor seed", "HS4": 1207, "Match_Quality": "close",
     "Notes": "HS 1207 = Other oil seeds; includes castor"},
    {"Crop": "Safflower", "HS4": 1207, "Match_Quality": "close",
     "Notes": "HS 1207 = Other oil seeds; includes safflower"},

    # ---- Spices (HS ch.9) ----
    {"Crop": "Black pepper", "HS4": 904, "Match_Quality": "exact",
     "Notes": "HS 0904 = Pepper of the genus Piper"},
    {"Crop": "Dry chillies", "HS4": 904, "Match_Quality": "close",
     "Notes": "HS 0904 also covers capsicum (chillies, dried)"},
    {"Crop": "Turmeric", "HS4": 910, "Match_Quality": "exact",
     "Notes": "HS 0910 = Ginger, saffron, turmeric, thyme, curry"},
    {"Crop": "Ginger", "HS4": 910, "Match_Quality": "exact",
     "Notes": "HS 0910 covers ginger"},
    {"Crop": "Dry Ginger", "HS4": 910, "Match_Quality": "exact",
     "Notes": "HS 0910 covers dried ginger"},
    {"Crop": "Coriander", "HS4": 909, "Match_Quality": "exact",
     "Notes": "HS 0909 = Seeds of anise, cumin, coriander, etc."},

    # ---- Sugar (HS ch.17) ----
    {"Crop": "Sugarcane", "HS4": 1701, "Match_Quality": "close",
     "Notes": "HS 1701 = Cane or beet sugar; sugarcane is the raw input"},

    # ---- Fruits (HS ch.8) ----
    {"Crop": "Banana", "HS4": 803, "Match_Quality": "exact",
     "Notes": "HS 0803 = Bananas; including plantains, fresh or dried"},
    {"Crop": "Cashewnut", "HS4": 801, "Match_Quality": "exact",
     "Notes": "HS 0801 = Coconuts, Brazil nuts and cashew nuts"},

    # ---- Fibers (HS ch.12/ch.53 — jute straddles) ----
    {"Crop": "Jute", "HS4": 1209, "Match_Quality": "approximate",
     "Notes": "HS 1209 = Seeds; used for sowing. Jute fiber is ch.53 (not in dataset). Using seed proxy."},

    # ---- Tobacco (HS ch.24 — outside ch.1-20 scope!) ----
    # Tobacco excluded: HS ch.24 is NOT in our dataset

    # ---- Cotton ----
    # Cotton(lint) excluded: HS ch.52 is NOT in our dataset (ch.1-20 only)
]

# Crops explicitly excluded and why:
EXCLUDED_CROPS = {
    "Cotton(lint)": "HS ch.52, outside dataset scope (ch.1-20)",
    "Tobacco": "HS ch.24, outside dataset scope (ch.1-20)",
    "Arecanut": "No clear HS4 match in ch.1-20",
    "Cowpea(Lobia)": "Could map to HS 0713 but too small a crop",
    "Guar seed": "No clear HS4 match",
    "Horse-gram": "No clear HS4 match",
    "Khesari": "No clear HS4 match",
    "Mesta": "Fiber crop, no HS4 in ch.1-20",
    "Moth": "No clear HS4 match",
    "Niger seed": "Too small, no clear HS4",
    "Oilseeds total": "Aggregate category, not a single crop",
    "Other Cereals": "Aggregate/residual category",
    "Other Kharif pulses": "Aggregate category",
    "Other Rabi pulses": "Aggregate category",
    "Other Summer Pulses": "Aggregate category",
    "Peas & beans (Pulses)": "Could map to 0713 but overlaps with other pulse crops already mapped",
    "Ragi": "Finger millet — could share 1008 with bajra but too ambiguous",
    "Small millets": "Could share 1008 with bajra but too ambiguous",
    "Sannhamp": "Fiber crop (sun hemp), no HS4 in ch.1-20",
    "other oilseeds": "Aggregate category",
}


def build_mapping():
    """Build the reviewed crop-to-HS4 mapping without modifying any dataset."""
    mapping_df = pd.DataFrame(CROP_HS4_MAPPING)

    print(f"  Crop-to-HS4 mapping: {len(mapping_df)} matches from 54 crops")
    print(f"  Match quality: {mapping_df['Match_Quality'].value_counts().to_dict()}")
    print(f"  Excluded: {len(EXCLUDED_CROPS)} crops")
    print("  Mapping retained in memory; no dataset/CSV is written.")

    return mapping_df


# ============================================================
# STEP 2: AGGREGATE CROP PRODUCTION TO NATIONAL LEVEL
# ============================================================

def aggregate_crop_production(crop_df, mapping_df):
    """
    Aggregate crop production to national level and keep state-level too.
    Group by (Crop, Season, Crop_Year), sum Area/Production, weighted-avg Yield.
    """
    print("\n  Aggregating crop production...")

    # Filter to only mapped crops
    mapped_crops = mapping_df["Crop"].unique()
    crop_mapped = crop_df[crop_df["Crop"].isin(mapped_crops)].copy()
    print(f"  Rows with mapped crops: {len(crop_mapped):,} / {len(crop_df):,}")

    # Drop rows where Production is null (0.5% of data)
    crop_mapped = crop_mapped.dropna(subset=["Production_Tonnes"])

    # --- National-level aggregation ---
    national = crop_mapped.groupby(["Crop", "Season", "Crop_Year", "Start_Year", "End_Year",
                                     "Season_Start_Month", "Season_End_Month", "Season_Months",
                                     "Season_Crosses_Calendar_Year"]).agg(
        Area_Ha_National=("Area_Ha", "sum"),
        Production_Tonnes_National=("Production_Tonnes", "sum"),
        # Weighted average yield: total production / total area
    ).reset_index()

    national["Yield_National"] = (
        national["Production_Tonnes_National"] / national["Area_Ha_National"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # Compute national YoY change
    national = national.sort_values(["Crop", "Season", "Crop_Year"])
    grp = national.groupby(["Crop", "Season"])
    national["Production_YoY_National"] = grp["Production_Tonnes_National"].pct_change() * 100
    national["Yield_YoY_National"] = grp["Yield_National"].pct_change() * 100

    # 3-year rolling mean for production deviation
    national["Production_3Y_Mean_National"] = (
        grp["Production_Tonnes_National"]
        .transform(lambda x: x.rolling(3, min_periods=2).mean())
    )
    national["Production_Deviation_National"] = (
        national["Production_Tonnes_National"] - national["Production_3Y_Mean_National"]
    )

    print(f"  National aggregation: {len(national)} rows")

    # --- State-level aggregation (keep for regional risk later) ---
    state_level = crop_mapped.groupby(["State", "Crop", "Season", "Crop_Year"]).agg(
        Area_Ha_State=("Area_Ha", "sum"),
        Production_Tonnes_State=("Production_Tonnes", "sum"),
    ).reset_index()
    state_level["Yield_State"] = (
        state_level["Production_Tonnes_State"] / state_level["Area_Ha_State"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    print(f"  State-level aggregation: {len(state_level)} rows (retained in memory)")

    return national


# ============================================================
# STEP 3: EXPAND SEASON-LEVEL TO MONTH-LEVEL
# ============================================================

def expand_to_monthly(national_df):
    """
    Expand season-level rows to monthly using Season_Months.
    Seasonal production is temporally distributed across constituent months
    using an equal-allocation assumption; these are estimated monthly values,
    not observed monthly production.
    """
    print("\n  Expanding season to monthly granularity...")
    print("  Seasonal production is temporally distributed across constituent months using an equal-allocation assumption; these are estimated monthly values, not observed monthly production.")

    monthly_rows = []
    for _, row in national_df.iterrows():
        season_months_str = str(row["Season_Months"])
        try:
            months = [int(m.strip()) for m in season_months_str.split(",") if m.strip()]
        except (ValueError, AttributeError):
            continue

        n_months = len(months)
        if n_months == 0:
            continue

        # Determine year for each month
        start_year = int(row["Start_Year"])
        crosses = row["Season_Crosses_Calendar_Year"]

        for m in months:
            # If season crosses calendar year and month is early (Jan-Jun),
            # it belongs to End_Year; otherwise Start_Year
            if crosses and m <= 6:
                year = int(row["End_Year"])
            else:
                year = start_year

            monthly_row = {
                "Crop": row["Crop"],
                "Season": row["Season"],
                "Crop_Year": row["Crop_Year"],
                "Year": year,
                "Month": m,
                "Area_Ha_National": row["Area_Ha_National"] / n_months,
                "Production_Tonnes_National": row["Production_Tonnes_National"] / n_months,
                "Yield_National": row["Yield_National"],  # yield is a rate, don't split
                "Production_YoY_National": row["Production_YoY_National"],
                "Yield_YoY_National": row["Yield_YoY_National"],
                "Production_Deviation_National": row.get("Production_Deviation_National", np.nan),
            }
            monthly_rows.append(monthly_row)

    monthly_df = pd.DataFrame(monthly_rows)
    print(f"  Monthly expansion: {len(monthly_df)} rows from {len(national_df)} season rows")
    return monthly_df


# ============================================================
# STEP 4: MERGE INTO MAIN DATASET
# ============================================================

def merge_with_main(main_df, monthly_crop_df, mapping_df):
    """
    Merge crop production data into main dataset via HS4 bridge.
    main.HS4 -> mapping.HS4 -> mapping.Crop -> crop_monthly.Crop
    Joined on (Crop, Year, Month).
    """
    print("\n  Merging crop production into main dataset...")

    # Add HS4 to monthly crop data via mapping
    crop_with_hs4 = monthly_crop_df.merge(
        mapping_df[["Crop", "HS4"]],
        on="Crop",
        how="left"
    )

    # Some crops map to same HS4 (e.g., Gram/Arhar/Moong/Urad/Masoor -> 713)
    # Aggregate by (HS4, Year, Month) for those
    crop_by_hs4 = crop_with_hs4.groupby(["HS4", "Year", "Month"]).agg(
        Production_Tonnes_National=("Production_Tonnes_National", "sum"),
        Area_Ha_National=("Area_Ha_National", "sum"),
        Yield_National=("Yield_National", "mean"),
        Production_YoY_National=("Production_YoY_National", "mean"),
        Yield_YoY_National=("Yield_YoY_National", "mean"),
        Production_Deviation_National=("Production_Deviation_National", "sum"),
    ).reset_index()

    # Left join to main dataset
    merged = main_df.merge(
        crop_by_hs4,
        on=["HS4", "Year", "Month"],
        how="left"
    )

    # Add Has_Production_Data flag
    merged["Has_Production_Data"] = merged["Production_Tonnes_National"].notna()

    print(f"  Merged shape: {merged.shape}")
    print(f"  Has_Production_Data: {merged['Has_Production_Data'].sum():,} / {len(merged):,} " +
          f"({merged['Has_Production_Data'].mean()*100:.1f}%)")
    print("  NOTE: A national agricultural observation can repeat across Country/Trade_Type rows.")
    print("  These are trade-level rows enriched with national agricultural context, not independent agricultural observations.")

    return merged


# ============================================================
# STEP 5: COMPUTE PRODUCTION RISK CATEGORIES
# ============================================================

def compute_production_risk(merged_df):
    """
    Derive Production_Risk (Low/Med/High/Critical) from
    Production_Deviation_From_3Y_Mean using quantile bucketing.
    """
    print("\n  Computing Production_Risk categories...")

    prod_data = merged_df[merged_df["Has_Production_Data"]].copy()
    dev = prod_data["Production_Deviation_National"]

    # Quantile bucketing (negative deviation = higher risk)
    # Lower deviation (more negative) = Critical risk
    q25, q50, q75 = dev.quantile([0.25, 0.50, 0.75])

    def assign_risk(d):
        if pd.isna(d):
            return np.nan
        if d <= q25:
            return "Critical"  # Bottom 25% (worst production shortfall)
        elif d <= q50:
            return "High"
        elif d <= q75:
            return "Medium"
        else:
            return "Low"  # Top 25% (production surplus)

    merged_df["Production_Risk"] = merged_df["Production_Deviation_National"].apply(assign_risk)

    risk_dist = merged_df["Production_Risk"].value_counts()
    print(f"  Risk distribution:\n{risk_dist.to_string()}")
    print(f"  Quantile boundaries: q25={q25:.0f}, q50={q50:.0f}, q75={q75:.0f}")

    return merged_df


# ============================================================
# STEP 6: TRAIN MODEL B
# ============================================================

LAGGED_FEATURES_B = [
    "Lagged_Effective_Shock_1",
    "Lagged_Effective_Shock_2",
    "Shock_Intensity_Lag1",
    "Shock_Intensity_Lag2",
    "Trade_Share_Lag1",
    "Trade_Share_Lag2",
]

EXOGENOUS_FEATURES_B = [
    "GPR",
    "INR_USD_Rate",
    "Natural_Disaster_Severity_Index",
]

# These target-derived/current agricultural outcomes are labels or inputs to a
# label only; they must never be predictors for Model B.
CURRENT_AGRICULTURAL_OUTCOMES = {
    "Production_Tonnes_National",
    "Yield_National",
    "Production_YoY_National",
    "Yield_YoY_National",
    "Production_Deviation_National",
    "Production_Risk",
}

FEATURES_B = LAGGED_FEATURES_B + EXOGENOUS_FEATURES_B
unsafe_features = set(FEATURES_B) & CURRENT_AGRICULTURAL_OUTCOMES
if unsafe_features:
    raise ValueError(f"Current agricultural outcomes cannot be Model B features: {sorted(unsafe_features)}")

TARGET_CLIP_LIMIT = 500


def evaluate_model(y_true, y_pred, label=""):
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2}


def clip_target_for_training_and_evaluation(target, target_name, split_name):
    """Limit unstable percentage changes caused by very small prior-year bases."""
    n_clipped = ((target < -TARGET_CLIP_LIMIT) | (target > TARGET_CLIP_LIMIT)).sum()
    print(
        f"    {target_name} ({split_name}) is clipped to +/-{TARGET_CLIP_LIMIT}% "
        f"for training/evaluation: {n_clipped:,} of {len(target):,} rows clipped."
    )
    return target.clip(-TARGET_CLIP_LIMIT, TARGET_CLIP_LIMIT)


def train_model_b(merged_df):
    """
    Train Model B on Has_Production_Data == True rows.
    Targets: Production_YoY_National, Yield_YoY_National
    Also trains Production_Risk classifier.

    SCOPE LIMITATION: No fertilizer/input-cost feature is derivable
    from this dataset (no HS ch.31 data).
    """
    print("\n" + "=" * 70)
    print("MODEL B TRAINING — Agricultural Impact")
    print("=" * 70)

    # Filter to rows with production data
    prod_df = merged_df[merged_df["Has_Production_Data"]].copy()
    print(f"\n  Training data: {len(prod_df):,} rows with production data")
    print("  Training rows are trade-level rows with national agricultural context; repeated national outcomes are not independent agricultural observations.")
    print("  Production_Risk is derived from Production_Deviation_National and is a supervised target, never an input feature.")
    print("  Current realized agricultural outcomes are excluded from predictors; only lagged, exogenous/predetermined, and calendar features are used.")
    print("  +/-500% target clipping is retained because percentage changes can become unstable with very small prior-year bases; clipping applies to both training and evaluation.")

    # One-hot encode Season from main dataset
    # Season info comes through the merge — use the Crop_Year's season
    # Actually, we need to add Season from the crop data.
    # Since we lost Season in the HS4 aggregation, let's use Month as a proxy
    # (months map to seasons deterministically for Indian agriculture)
    # Kharif: Jun-Oct (months 6-10), Rabi: Nov-Mar (months 11,12,1,2,3),
    # Summer: Mar-May (months 3-5), etc.
    prod_df["Season_Kharif"] = prod_df["Month"].isin([6, 7, 8, 9, 10]).astype(int)
    prod_df["Season_Rabi"] = prod_df["Month"].isin([11, 12, 1, 2, 3]).astype(int)
    prod_df["Season_Summer"] = prod_df["Month"].isin([3, 4, 5]).astype(int)

    season_features = ["Season_Kharif", "Season_Rabi", "Season_Summer"]
    all_features = FEATURES_B + season_features

    # Handle NaN in features
    for col in all_features:
        if col in prod_df.columns:
            prod_df[col] = prod_df[col].fillna(0)

    # --- Chronological split ---
    # Model B window is narrower: crop data through 2022-23
    # Start_Year for 2022-23 crop year is 2022
    # So we can use 2022 as latest training year
    # But some months of 2022-23 crop year fall in 2023 (Rabi months Jan-Mar)
    # Use: Train = Year <= 2021, Val = Year == 2022, Test = Year >= 2023
    # This is the narrower split per Task 3 spec

    train_b = prod_df[prod_df["Year"] <= 2021].copy()
    val_b = prod_df[prod_df["Year"] == 2022].copy()
    # Test: limited since crop data only goes to 2022-23
    test_b = prod_df[prod_df["Year"] >= 2023].copy()

    print(f"\n  Model B chronological split (narrower due to crop data):")
    print(f"    Train (<=2021): {len(train_b):,} rows")
    print(f"    Val   (2022):   {len(val_b):,} rows")
    print(f"    Test  (>=2023): {len(test_b):,} rows")

    if len(test_b) == 0:
        print("    WARNING: No test data for 2023+ (crop data ends at 2022-23)")
        print("    The prescribed chronological split is retained; test metrics will be unavailable.")

    results_b = {}

    # ---- TARGET 1: Production_YoY_National ----
    target1 = "Production_YoY_National"
    print(f"\n  --- Target: {target1} ---")

    for split_df, name in [(train_b, "train"), (val_b, "val"), (test_b, "test")]:
        n_valid = split_df[target1].notna().sum()
        print(f"    {name}: {n_valid} non-null target rows / {len(split_df)} total")

    # Drop rows where target is NaN (cold-start periods)
    train_t1 = train_b.dropna(subset=[target1])
    val_t1 = val_b.dropna(subset=[target1])
    test_t1 = test_b.dropna(subset=[target1])

    if len(train_t1) > 0 and len(val_t1) > 0:
        X_train = train_t1[all_features]
        y_train = clip_target_for_training_and_evaluation(train_t1[target1], target1, "train")
        X_val = val_t1[all_features]
        y_val = clip_target_for_training_and_evaluation(val_t1[target1], target1, "validation")
        X_test = test_t1[all_features] if len(test_t1) > 0 else pd.DataFrame()
        y_test = (clip_target_for_training_and_evaluation(test_t1[target1], target1, "test")
                  if len(test_t1) > 0 else pd.Series(dtype=float))

        # Naive baseline: predict 0 (no change)
        bl_val = evaluate_model(y_val, np.zeros(len(y_val)), "baseline_val")
        bl_test = evaluate_model(y_test, np.zeros(len(y_test)), "baseline_test") if len(y_test) > 0 else None

        print(f"    Baseline (predict 0): Val MAE={bl_val['MAE']:.2f}, R2={bl_val['R2']:.4f}")

        # Train models
        for ModelCls, name, params in [
            (RandomForestRegressor, "RandomForest", dict(
                n_estimators=200, max_depth=12, min_samples_leaf=5,
                random_state=RANDOM_STATE, n_jobs=-1)),
            (lgb.LGBMRegressor, "LightGBM", dict(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ]:
            model = ModelCls(**params)
            model.fit(X_train, y_train)

            pred_val = model.predict(X_val)
            metrics_val = evaluate_model(y_val, pred_val, f"{name}_val")

            pred_test = model.predict(X_test) if len(X_test) > 0 else np.array([])
            metrics_test = evaluate_model(y_test, pred_test, f"{name}_test") if len(y_test) > 0 else None

            beats = metrics_val["MAE"] < bl_val["MAE"]
            print(f"    {name}: Val MAE={metrics_val['MAE']:.2f}, R2={metrics_val['R2']:.4f} " +
                  f"| {'BEATS' if beats else 'LOSES TO'} baseline")

            if metrics_test:
                print(f"      Test MAE={metrics_test['MAE']:.2f}, R2={metrics_test['R2']:.4f}")

            results_b[f"{target1}_{name}"] = {
                "metrics_val": metrics_val,
                "metrics_test": metrics_test,
                "baseline_val": bl_val,
                "baseline_test": bl_test,
                "beats_baseline": beats,
            }

            # Save best model for this target
            if name == "LightGBM":
                joblib.dump(model, os.path.join(MODELS_DIR, "model_b_production_yoy.joblib"))

    # ---- TARGET 2: Yield_YoY_National ----
    target2 = "Yield_YoY_National"
    print(f"\n  --- Target: {target2} ---")

    train_t2 = train_b.dropna(subset=[target2])
    val_t2 = val_b.dropna(subset=[target2])
    test_t2 = test_b.dropna(subset=[target2])

    if len(train_t2) > 0 and len(val_t2) > 0:
        X_train = train_t2[all_features]
        y_train = clip_target_for_training_and_evaluation(train_t2[target2], target2, "train")
        X_val = val_t2[all_features]
        y_val = clip_target_for_training_and_evaluation(val_t2[target2], target2, "validation")
        X_test = test_t2[all_features] if len(test_t2) > 0 else pd.DataFrame()
        y_test = (clip_target_for_training_and_evaluation(test_t2[target2], target2, "test")
                  if len(test_t2) > 0 else pd.Series(dtype=float))

        bl_val = evaluate_model(y_val, np.zeros(len(y_val)), "baseline_val")
        print(f"    Baseline (predict 0): Val MAE={bl_val['MAE']:.2f}, R2={bl_val['R2']:.4f}")

        for ModelCls, name, params in [
            (RandomForestRegressor, "RandomForest", dict(
                n_estimators=200, max_depth=12, min_samples_leaf=5,
                random_state=RANDOM_STATE, n_jobs=-1)),
            (lgb.LGBMRegressor, "LightGBM", dict(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ]:
            model = ModelCls(**params)
            model.fit(X_train, y_train)

            pred_val = model.predict(X_val)
            metrics_val = evaluate_model(y_val, pred_val, f"{name}_val")

            pred_test = model.predict(X_test) if len(X_test) > 0 else np.array([])
            metrics_test = evaluate_model(y_test, pred_test, f"{name}_test") if len(y_test) > 0 else None

            beats = metrics_val["MAE"] < bl_val["MAE"]
            print(f"    {name}: Val MAE={metrics_val['MAE']:.2f}, R2={metrics_val['R2']:.4f} " +
                  f"| {'BEATS' if beats else 'LOSES TO'} baseline")

            results_b[f"{target2}_{name}"] = {
                "metrics_val": metrics_val,
                "metrics_test": metrics_test,
                "baseline_val": bl_val,
                "beats_baseline": beats,
            }

            if name == "LightGBM":
                joblib.dump(model, os.path.join(MODELS_DIR, "model_b_yield_yoy.joblib"))

    # ---- TARGET 3: Production_Risk (Classification) ----
    target3 = "Production_Risk"
    print(f"\n  --- Target: {target3} (Classification) ---")

    train_t3 = train_b.dropna(subset=[target3])
    val_t3 = val_b.dropna(subset=[target3])
    test_t3 = test_b.dropna(subset=[target3])

    if len(train_t3) > 0 and len(val_t3) > 0:
        risk_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        y_train = train_t3[target3].map(risk_map)
        y_val = val_t3[target3].map(risk_map)
        y_test = test_t3[target3].map(risk_map) if len(test_t3) > 0 else pd.Series()

        X_train = train_t3[all_features]
        X_val = val_t3[all_features]
        X_test = test_t3[all_features] if len(test_t3) > 0 else pd.DataFrame()

        # Drop any rows where mapping failed
        valid_train = y_train.notna()
        valid_val = y_val.notna()
        X_train, y_train = X_train[valid_train], y_train[valid_train].astype(int)
        X_val, y_val = X_val[valid_val], y_val[valid_val].astype(int)

        # Baseline: most frequent class
        most_frequent = y_train.mode()[0]
        bl_acc = accuracy_score(y_val, np.full(len(y_val), most_frequent))
        print(f"    Baseline (most frequent class={most_frequent}): Acc={bl_acc:.4f}")

        for ModelCls, name, params in [
            (RandomForestClassifier, "RandomForest", dict(
                n_estimators=200, max_depth=12, min_samples_leaf=5,
                random_state=RANDOM_STATE, n_jobs=-1)),
            (lgb.LGBMClassifier, "LightGBM", dict(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ]:
            model = ModelCls(**params)
            model.fit(X_train, y_train)

            pred_val = model.predict(X_val)
            acc = accuracy_score(y_val, pred_val)
            f1 = f1_score(y_val, pred_val, average="macro")

            beats = acc > bl_acc
            print(f"    {name}: Acc={acc:.4f}, F1={f1:.4f} | {'BEATS' if beats else 'LOSES TO'} baseline")

            results_b[f"{target3}_{name}"] = {
                "accuracy": acc,
                "f1_macro": f1,
                "baseline_accuracy": bl_acc,
                "beats_baseline": beats,
            }

            if name == "LightGBM":
                joblib.dump(model, os.path.join(MODELS_DIR, "model_b_production_risk.joblib"))

    return results_b


def main():
    print("=" * 70)
    print("Drishti - Task 3: Crop-to-HS4 Bridge + Model B")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    print("\n  SCOPE LIMITATION: No fertilizer/input-cost features derivable")
    print("  from this dataset (HS ch.31 absent). Noted for methods section.")

    # Step 1: Build mapping
    print("\n--- STEP 1: Crop-to-HS4 Mapping ---")
    mapping_df = build_mapping()

    # Step 2: Load and aggregate crop production
    print("\n--- STEP 2: Aggregate Crop Production ---")
    crop_df = pd.read_csv(CROP_CSV)
    national_df = aggregate_crop_production(crop_df, mapping_df)

    # Step 3: Expand to monthly
    print("\n--- STEP 3: Season-to-Month Expansion ---")
    monthly_crop = expand_to_monthly(national_df)

    print("  Monthly agricultural estimates retained in memory; no dataset/CSV is written.")

    # Step 4: Load main dataset and merge
    print("\n--- STEP 4: Merge with Main Dataset ---")
    main_df = pd.read_csv(MAIN_CSV)
    merged_df = merge_with_main(main_df, monthly_crop, mapping_df)

    # Step 5: Compute production risk
    merged_df = compute_production_risk(merged_df)

    print("\n  Merged trade/agriculture data retained in memory; no dataset/CSV is written.")
    print(f"  Shape: {merged_df.shape}")

    # Step 6: Train Model B
    results_b = train_model_b(merged_df)

    # Save results
    results_path = os.path.join(RESULTS_DIR, "model_b_results.json")
    with open(results_path, "w") as f:
        json.dump(results_b, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    # Generate predictions for downstream (Model C cascade input)
    print("\n  Generating production growth model predictions for the existing Model C artifact interface...")
    print("  NOTE: This artifact may include in-sample predictions; it is not an out-of-sample cascade prediction file.")
    prod_model = joblib.load(os.path.join(MODELS_DIR, "model_b_production_yoy.joblib"))

    season_features = ["Season_Kharif", "Season_Rabi", "Season_Summer"]
    merged_df["Season_Kharif"] = merged_df["Month"].isin([6, 7, 8, 9, 10]).astype(int)
    merged_df["Season_Rabi"] = merged_df["Month"].isin([11, 12, 1, 2, 3]).astype(int)
    merged_df["Season_Summer"] = merged_df["Month"].isin([3, 4, 5]).astype(int)

    all_features = FEATURES_B + season_features
    for col in all_features:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0)

    # Predict on rows with production data
    pred_mask = merged_df["Has_Production_Data"]
    if pred_mask.sum() > 0:
        merged_df.loc[pred_mask, "Production_Growth_Pred"] = prod_model.predict(
            merged_df.loc[pred_mask, all_features]
        )
    merged_df["Production_Growth_Pred"] = merged_df["Production_Growth_Pred"].fillna(0)

    pred_path = os.path.join(RESULTS_DIR, "model_b_predictions.csv")
    merged_df[["Year", "Month", "Country", "Trade_Type", "HS4",
               "Has_Production_Data", "Production_Growth_Pred",
               "Production_Risk"]].to_csv(pred_path, index=False)
    print(f"  Predictions saved: {pred_path}")

    print("\n" + "=" * 70)
    print("TASK 3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
