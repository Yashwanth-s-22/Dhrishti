"""
Drishti - Model B Deep Audit Script
===================================
Gathers exact empirical statistics across all 15 audit dimensions.
"""

import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MAIN_CSV = os.path.join(DATA_DIR, "Drishti_Cascade_Final_With_EMDAT.csv")
CROP_CSV = os.path.join(DATA_DIR, "Crop_Production_Final.csv")

import sys
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))
from train_model_b_agriculture import (
    CROP_HS4_MAPPING, EXCLUDED_CROPS, build_mapping,
    aggregate_crop_production, expand_to_monthly, merge_with_main,
    compute_production_risk, TARGET_CLIP_LIMIT
)

print("=" * 80)
print("RUNNING MODEL B DEEP METHODOLOGICAL AUDIT CALCULATIONS")
print("=" * 80)

# 1. Mapping stats
mapping_df = pd.DataFrame(CROP_HS4_MAPPING)
crop_raw = pd.read_csv(CROP_CSV)
total_source_crops = crop_raw["Crop"].nunique()
mapped_crops_count = len(mapping_df)
excluded_crops_count = len(EXCLUDED_CROPS)

print(f"\n1. Crop Mapping:")
print(f"  Total unique crops in raw dataset: {total_source_crops}")
print(f"  Mapped crops: {mapped_crops_count}")
print(f"  Excluded crops: {excluded_crops_count}")
print(f"  Match qualities: {mapping_df['Match_Quality'].value_counts().to_dict()}")

# 2. Raw vs Aggregated vs Monthly rows
national_df = aggregate_crop_production(crop_raw, mapping_df)
monthly_crop = expand_to_monthly(national_df)
main_df = pd.read_csv(MAIN_CSV)
merged_df = merge_with_main(main_df, monthly_crop, mapping_df)
merged_df = compute_production_risk(merged_df)

print(f"\n2. Data Volume and Coverage:")
print(f"  Raw crop rows: {len(crop_raw):,}")
print(f"  Rows with mapped crops: {len(crop_raw[crop_raw['Crop'].isin(mapping_df['Crop'])]):,}")
print(f"  National seasonal aggregated rows: {len(national_df):,}")
print(f"  Expanded monthly crop rows: {len(monthly_crop):,}")
print(f"  Main trade dataset rows: {len(main_df):,}")
print(f"  Merged trade dataset with production data: {merged_df['Has_Production_Data'].sum():,} ({merged_df['Has_Production_Data'].mean()*100:.2f}%)")

# 3. Independent vs Repeated observations
prod_df = merged_df[merged_df["Has_Production_Data"]].copy()
unique_agri_obs = prod_df[["HS4", "Year", "Month"]].drop_duplicates()
print(f"\n3. Pseudo-Replication / Repeated Observations:")
print(f"  Total trade rows with production data: {len(prod_df):,}")
print(f"  Unique (HS4, Year, Month) agricultural cells: {len(unique_agri_obs):,}")
print(f"  Average trade rows per agricultural cell: {len(prod_df) / len(unique_agri_obs):.2f}")
print(f"  Unique season/crop national observations: {len(national_df):,}")

# 4. Clipping stats across splits
train_df = prod_df[prod_df["Year"] <= 2021]
val_df = prod_df[prod_df["Year"] == 2022]
test_df = prod_df[prod_df["Year"] >= 2023]

print(f"\n4. Chronological Splits & Clipping Stats:")
for name, split in [("Train (<=2021)", train_df), ("Val (2022)", val_df), ("Test (>=2023)", test_df)]:
    p_nonnull = split["Production_YoY_National"].dropna()
    p_clipped = ((p_nonnull < -TARGET_CLIP_LIMIT) | (p_nonnull > TARGET_CLIP_LIMIT)).sum()
    y_nonnull = split["Yield_YoY_National"].dropna()
    y_clipped = ((y_nonnull < -TARGET_CLIP_LIMIT) | (y_nonnull > TARGET_CLIP_LIMIT)).sum()
    print(f"  Split: {name}")
    print(f"    Total rows: {len(split):,}")
    print(f"    Prod_YoY non-null: {len(p_nonnull):,} | Clipped: {p_clipped:,} ({p_clipped/len(p_nonnull)*100:.2f}%)")
    print(f"    Yield_YoY non-null: {len(y_nonnull):,} | Clipped: {y_clipped:,} ({y_clipped/len(y_nonnull)*100:.2f}%)")

# 5. Crop year range
print(f"\n5. Crop Data Temporal Range:")
print(f"  Crop_Year values in crop data: {sorted(crop_raw['Crop_Year'].unique())}")
print(f"  Years in monthly crop data: {sorted(monthly_crop['Year'].unique())}")
print(f"  Years in main trade dataset: {sorted(main_df['Year'].unique())}")
